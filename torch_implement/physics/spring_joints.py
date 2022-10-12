"""Joints connect bodies and constrain their movement.

These joint definitions are from the original Brax release, and constrain joints
via acceleration-level, springy updates.  We find these types of joints to be
simple to implement, fairly fast to simulate, but difficult to tune, typically
requiring significant iteration on the damping constants to get acceptable
motion quality. All future environments will be built with the newer,
position-level updates defined in joints.py which are infinitely stiff by
default.
"""

from typing import Any, List, Tuple

from torch_math import math
import torch_math.array_ops as arrop

from physics import bodies
from physics import torch_config_pb2
from physics import joints
from physics.torch_base import P, QP, vec_to_arr
import numpy as np


class Joint(joints.Joint):
  """A spring joint connects two bodies and constrains their movement.

  This constraint is determined by axes that define how to bodies may move in
  relation to one-another.
  """

  __pytree_ignore__ = ('index', 'dof')

  def __init__(self,
               joints: List[torch_config_pb2.Joint],
               body: bodies.Body,
               spring_damping_coeff: float = 2.0):
    """Creates a Joint that connects two bodies and constrains their movement.

    Args:
      joints: list of joints (all of the same type) to batch together
      body: batched body that contain the parents and children of each joint
      spring_damping_coeff: coefficient for setting default spring damping
    """
    # springy-joint specific fields
    self.stiffness = np.array([j.stiffness for j in joints])
    self.spring_damping = np.array([
        j.spring_damping if j.HasField('spring_damping') else
        spring_damping_coeff * arrop.sqrt(j.stiffness) for j in joints
    ])
    self.limit_strength = np.array([
        j.limit_strength if j.HasField('limit_strength') else j.stiffness
        for j in joints
    ])
    self.angular_damping = np.array([j.angular_damping for j in joints])
    self.limit = np.array([[[i.min, i.max]
                            for i in j.angle_limit]
                           for j in joints]) / 180.0 * np.pi
    
    # TODO: the jp.take here is substitutes by the take of the body
    self.body_p = body.take([body.index[j.parent] for j in joints])
    self.body_c = body.take([body.index[j.child] for j in joints])
    # self.body_p = jp.take(body, [body.index[j.parent] for j in joints])
    # self.body_c = jp.take(body, [body.index[j.child] for j in joints])
    
    self.off_p = np.array([vec_to_arr(j.parent_offset) for j in joints])
    self.off_c = np.array([vec_to_arr(j.child_offset) for j in joints])
    self.index = {j.name: i for i, j in enumerate(joints)}
    self.dof = len(joints[0].angle_limit)
    v_rot = arrop.torchVmap(math.rotate, include=[True, False])
    relative_quats = arrop.array(
        [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joints])
    self.axis_c = np.array([
        v_rot(arrop.eye(3), math.euler_to_quat(vec_to_arr(j.rotation)))
        for j in joints
    ])
    self.axis_p = jp.array(
        [v_rot(j, r) for j, r in zip(self.axis_c, relative_quats)])
    self.free_dofs = None  # property used by pbd joints

  def apply(self, qp: QP) -> P:
    """Returns impulses to constrain and align bodies connected by a joint.

    Only called by springy joints.

    Args:
      qp: State data for system

    Returns:
      dP: Impulses on all bodies to maintain joint constraints
    """
    qp_p = np.take(qp, self.body_p.idx)
    qp_c = np.take(qp, self.body_c.idx)
    dp_p, dp_c = arrop.torchVmap(type(self).apply_reduced)(self, qp_p, qp_c)

    # sum together all impulse contributions across parents and children
    body_idx = arrop.concatenate((self.body_p.idx, self.body_c.idx))
    dp_vel = arrop.concatenate((dp_p.vel, dp_c.vel))  # pytype: disable=attribute-error
    dp_ang = arrop.concatenate((dp_p.ang, dp_c.ang))  # pytype: disable=attribute-error
    dp_vel = arrop.segment_sum(dp_vel, body_idx, qp.pos.shape[0])
    dp_ang = arrop.segment_sum(dp_ang, body_idx, qp.pos.shape[0])

    return P(vel=dp_vel, ang=dp_ang)


class Revolute(Joint):
  """A springy revolute joint constrains two bodies around a single axis.

  Constructs a revolute joint where the parent's local x-axis is constrained
  to point in the same direction as the child's local x-axis.  This construction
  follows the line of nodes convention shared by the universal and spherical
  joints for x-y'-z'' intrinsic euler angles.
  """

  def __init__(self, joints: List[torch_config_pb2.Joint], body: bodies.Body):
    super().__init__(joints, body, .5)

  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = qp_p.to_world(self.off_p)
    pos_c, vel_c = qp_c.to_world(self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to align their axes
    (axis,), (angle,) = self.axis_angle(qp_p, qp_c)
    axis_c = math.rotate(self.axis_c[0], qp_c.rot)
    torque = self.stiffness * arrop.cross(axis, axis_c)

    # torque the bodies to stay within angle limits
    dang = arrop.where(angle < self.limit[0][0], self.limit[0][0] - angle, 0)
    dang = arrop.where(angle > self.limit[0][1], self.limit[0][1] - angle, dang)
    torque -= self.limit_strength * axis * dang

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dang_p = self.body_p.inertia * torque
    dang_c = -self.body_c.inertia * torque
    dp_p = dp_p.replace(ang=dp_p.ang + dang_p)
    dp_c = dp_c.replace(ang=dp_c.ang + dang_c)

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    axis_p = math.rotate(self.axis_p[0], qp_p.rot)
    ref_p = math.rotate(self.axis_p[2], qp_p.rot)
    ref_c = math.rotate(self.axis_c[2], qp_c.rot)
    # algebraically the same as the calculation in `Spherical`, but simpler
    # because child local-x and parent local-x are constrained to be the same
    psi = math.signed_angle(axis_p, ref_p, ref_c)
    return (axis_p,), (psi,)


# @pytree.register
class Universal(Joint):
  """A springy universal joint constrains two bodies around two axes.

  Constructs a universal joint defined as the first two degrees of freedom
  of a spherical joint.  See `SpringSpherical` for details.
  """

  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = qp_p.to_world(self.off_p)
    pos_c, vel_c = qp_c.to_world(self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to align to a joint plane
    (axis_1, axis_2), angles = self.axis_angle(qp_p, qp_c)
    axis_c_proj = axis_2 - arrop.dot(axis_2, axis_1) * axis_1
    axis_c_proj = axis_c_proj / arrop.norm(axis_c_proj)
    torque = (self.limit_strength / 5.) * arrop.cross(axis_c_proj, axis_2)

    # torque the bodies to stay within angle limits
    axis, angle = np.array((axis_1, axis_2)), np.array(angles)
    dang = arrop.where(angle < self.limit[:, 0], self.limit[:, 0] - angle, 0)
    dang = arrop.where(angle > self.limit[:, 1], self.limit[:, 1] - angle, dang)
    torque -= self.limit_strength * arrop.sum(arrop.torchVmap(arrop.multiply)(axis, dang), 0)

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dang_p = self.body_p.inertia * torque
    dang_c = -self.body_c.inertia * torque
    dp_p = dp_p.replace(ang=dp_p.ang + dang_p)
    dp_c = dp_c.replace(ang=dp_c.ang + dang_c)

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    v_rot = arrop.torchVmap(math.rotate, include=[True, False])
    axis_p_rotated = v_rot(self.axis_p, qp_p.rot)
    axis_c_rotated = v_rot(self.axis_c, qp_c.rot)
    axis_1_p = axis_p_rotated[0]
    axis_2_p = axis_p_rotated[1]
    axis_1_c = axis_c_rotated[0]
    axis_2_c = axis_c_rotated[1]
    axis_3_c = axis_c_rotated[2]

    line_of_nodes = arrop.cross(axis_3_c, axis_1_p)
    line_of_nodes = line_of_nodes / (1e-10 + arrop.norm(line_of_nodes))
    y_n_normal = axis_1_p
    psi = math.signed_angle(y_n_normal, axis_2_p, line_of_nodes)
    axis_1_p_in_xz_c = arrop.dot(axis_1_p, axis_1_c) * axis_1_c + arrop.dot(
        axis_1_p, axis_2_c) * axis_2_c
    axis_1_p_in_xz_c = axis_1_p_in_xz_c / (1e-10 +
                                           arrop.norm(axis_1_p_in_xz_c))
    theta = arrop.safe_arccos(arrop.clip(arrop.dot(axis_1_p_in_xz_c, axis_1_p), -1,
                                   1)) * arrop.sign(arrop.dot(axis_1_p, axis_3_c))
    axis = (axis_1_p, axis_2_c)
    angle = (psi, theta)

    return axis, angle



class Spherical(Joint):
  """A springy spherical joint constrains two bodies around three axes.

  Constructs a spherical joint which returns intrinsic euler angles in the
    x-y'-z'' convention between the parent and child.  Uses the line of nodes
    construction described in section 3.2.3.2 here:
    https://www.sedris.org/wg8home/Documents/WG80485.pdf
  """

  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[P, P]:
    """Returns calculated impulses in compressed joint space."""
    pos_p, vel_p = qp_p.to_world(self.off_p)
    pos_c, vel_c = qp_c.to_world(self.off_c)

    # push the bodies towards their offsets
    # TODO: cap this damping so that it can't overcorrect
    impulse = (pos_p - pos_c) * self.stiffness + self.spring_damping * (
        vel_p - vel_c)
    dp_p = self.body_p.impulse(qp_p, -impulse, pos_p)
    dp_c = self.body_c.impulse(qp_c, impulse, pos_c)

    # torque the bodies to stay within angle limits
    axes, angles = self.axis_angle(qp_p, qp_c)
    axis, angle = np.array(axes), np.array(angles)
    dang = arrop.where(angle < self.limit[:, 0], self.limit[:, 0] - angle, 0)
    dang = arrop.where(angle > self.limit[:, 1], self.limit[:, 1] - angle, dang)
    torque = -self.limit_strength * arrop.sum(arrop.torchVmap(arrop.multiply)(axis, dang), 0)

    # damp the angular motion
    torque -= self.angular_damping * (qp_p.ang - qp_c.ang)

    dp_p = dp_p.replace(ang=dp_p.ang + self.body_p.inertia * torque)
    dp_c = dp_c.replace(ang=dp_c.ang + -self.body_c.inertia * torque)

    return dp_p, dp_c

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    v_rot = arrop.torchVmap(math.rotate, include=[True, False])
    axis_p_rotated = v_rot(self.axis_p, qp_p.rot)
    axis_c_rotated = v_rot(self.axis_c, qp_c.rot)
    axis_1_p = axis_p_rotated[0]
    axis_2_p = axis_p_rotated[1]
    axis_1_c = axis_c_rotated[0]
    axis_2_c = axis_c_rotated[1]
    axis_3_c = axis_c_rotated[2]

    line_of_nodes = arrop.cross(axis_3_c, axis_1_p)
    line_of_nodes = line_of_nodes / (1e-10 + arrop.norm(line_of_nodes))
    y_n_normal = axis_1_p
    psi = math.signed_angle(y_n_normal, axis_2_p, line_of_nodes)
    axis_1_p_in_xz_c = arrop.dot(axis_1_p, axis_1_c) * axis_1_c + arrop.dot(
        axis_1_p, axis_2_c) * axis_2_c
    axis_1_p_in_xz_c = axis_1_p_in_xz_c / (1e-10 +
                                           arrop.norm(axis_1_p_in_xz_c))
    ang_between_1_p_xz_c = arrop.dot(axis_1_p_in_xz_c, axis_1_p)
    theta = arrop.safe_arccos(arrop.clip(ang_between_1_p_xz_c, -1, 1)) * arrop.sign(
        arrop.dot(axis_1_p, axis_3_c))
    yc_n_normal = -axis_3_c
    phi = math.signed_angle(yc_n_normal, axis_2_c, line_of_nodes)
    axis = (axis_1_p, axis_2_c, axis_3_c)
    angle = (psi, theta, phi)

    return axis, angle


def get(config: torch_config_pb2.Config, body: bodies.Body) -> List[Joint]:
  """Creates all joints given a config."""
  joints = {}
  for joint in config.joints:
    dof = len(joint.angle_limit)
    springy = joint.stiffness > 0
    if springy:
      if dof not in joints:
        joints[dof] = []
      joints[dof].append(joint)

  # ensure stable order for joint application: dof
  joints = sorted(joints.items(), key=lambda kv: kv[0])
  ret = []

  for dof, v in joints:
    if dof == 1:
      ret.append(Revolute(v, body))
    elif dof == 2:
      ret.append(Universal(v, body))
    elif dof == 3:
      ret.append(Spherical(v, body))
    else:
      raise RuntimeError(f'invalid number of joint limits: {dof}')

  return ret
