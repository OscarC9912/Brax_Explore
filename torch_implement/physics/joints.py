"""Joints connect bodies and constrain their movement."""

import abc
import numpy as np
from numpy import *
from numpy import array
from typing import Any, List, Tuple, Union, Optional

from torch_math.array_ops import *
import torch_math.math as math
from physics import bodies
from physics import torch_config_pb2
from physics.torch_base import P, Q, QP, vec_to_arr
import torch_math.array_ops as arrop
import copy


class Joint(abc.ABC):
  """A joint connects two bodies and constrains their movement.

  This constraint is determined by axes that define how to bodies may move in
  relation to one-another.
  """

  # __pytree_ignore__ = ('index', 'dof', 'free_dofs')
  
  
  def take(self, i):
    """Return a new sliced Joint Object."""
    new_joint = copy.deepcopy(self)  # prevent aliasing
    
    new_joint.angular_damping = np.take(self.angular_damping, i)
    new_joint.scale_pos = np.take(self.scale_pos, i)
    new_joint.scale_ang = np.take(self.scale_ang, i)
    new_joint.limit = np.take(self.limit, i)
    new_joint.off_p = np.take(self.off_p, i)
    new_joint.off_c = np.take(self.off_c, i)
    new_joint.axis_c = np.take(self.axis_c, i)
    new_joint.axis_p = np.take(self.axis_p, i)
    
    return new_joint
  

  def __init__(self,
               joints: List[torch_config_pb2.Joint],
               body: bodies.Body,
               solver_scale_pos: float = 0.6,
               solver_scale_ang: float = 0.25,
               free_dofs: Optional[List[int]] = None):
    """Creates a Joint that connects two bodies and constrains their movement.

    Args:
      joints: list of joints (all of the same type) to batch together
      body: batched body that contain the parents and children of each joint
      solver_scale_pos: Magnitude of jacobi update for position based updates
      solver_scale_ang: Magnitude of jacobi update for angular position based
        update
      free_dofs: Number of free (not-fixed) degrees of freedom per joint
    """
    self.angular_damping = array([j.angular_damping for j in joints])
    self.scale_pos = array([solver_scale_pos for j in joints])
    self.scale_ang = array([solver_scale_ang for j in joints])

    self.limit = array([[[i.min, i.max]
                            for i in j.angle_limit]
                           for j in joints]) / 180.0 * np.pi
    
    
    # ==== Apply the newer version of the take function ======
    # self.body_p = take(body, [body.index[j.parent] for j in joints])
    # self.body_c = take(body, [body.index[j.child] for j in joints])

    self.body_p = body.take([body.index[j.parent] for j in joints])
    self.body_c = body.take([body.index[j.child] for j in joints])
    
    self.off_p = array([vec_to_arr(j.parent_offset) for j in joints])
    self.off_c = array([vec_to_arr(j.child_offset) for j in joints])
    self.index = {j.name: i for i, j in enumerate(joints)}
    self.dof = len(joints[0].angle_limit)


    # HERE we deal with the torch function 
    v_rot = torchVmap(math.rotate, include=[True, False])
    # if v_rot is None:
    #   def map_vrot(vec: ndarray, quat: ndarray):
    #     length = len(vec.shape)
    #     out = []
    #     for i in range(length):
    #       out.append(math.rotate(vec[i], quat))
    #     return out
    #   v_rot = map_vrot
    # HERE the thing ends


    relative_quats = array(
        [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joints])
    

    self.axis_c = np.array([
        v_rot(eye(3), math.euler_to_quat(vec_to_arr(j.rotation)))
        for j in joints
    ])

    self.axis_p = np.array(
        [v_rot(j, r) for j, r in zip(self.axis_c, relative_quats)])

    self.free_dofs = free_dofs

  def apply(self, qp: QP) -> Q:
    """Returns position-based update to constrain bodies connected by a joint.

    Args:
      qp: State data for system

    Returns:
      dQ: Change in position and quaternions to enforce joint constraint.
    """

    qp_p = np.take(qp, self.body_p.idx)
    qp_c = np.take(qp, self.body_c.idx)
    dq_p, dq_c = arrop.torchVmap(type(self).apply_reduced)(self, qp_p, qp_c)  # pytype: disable=attribute-error

    # sum together all impulse contributions across parents and children
    body_idx = arrop.concatenate((self.body_p.idx, self.body_c.idx))
    dq_pos = arrop.concatenate((dq_p.pos, dq_c.pos))  # pytype: disable=attribute-error
    dq_rot = arrop.concatenate((dq_p.rot, dq_c.rot))  # pytype: disable=attribute-error
    dq_pos = arrop.segment_sum(dq_pos, body_idx, qp.pos.shape[0])
    dq_rot = arrop.segment_sum(dq_rot, body_idx, qp.rot.shape[0])

    return Q(pos=dq_pos, rot=dq_rot)

  # TODO: replace this function call with a more efficient version
  def damp(self, qp: QP) -> P:
    """Returns an acceleration-level damping term for the joint.

    Args:
      qp: State data for system

    Returns:
      dP: Change in velocity from joint damping.
    """

    def damp_reduced(self, qp_p, qp_c):
      torque = -1. * self.angular_damping * (qp_p.ang - qp_c.ang)
      dang_p = self.body_p.inertia * torque
      dang_c = -self.body_c.inertia * torque
      return dang_p, dang_c

    qp_p = np.take(qp, self.body_p.idx)
    qp_c = np.take(qp, self.body_c.idx)
    dang_p, dang_c = torchVmap(damp_reduced)(self, qp_p, qp_c)  # pytype: disable=attribute-error

    # sum together all impulse contributions across parents and children
    body_idx = arrop.concatenate((self.body_p.idx, self.body_c.idx))
    dq_ang = arrop.concatenate((dang_p, dang_c))  # pytype: disable=attribute-error
    dq_ang = arrop.segment_sum(dq_ang, body_idx, qp.ang.shape[0])

    return P(vel=np.zeros_like(dq_ang), ang=dq_ang)

  def apply_angle_update(self, qp_p: QP, qp_c: QP,
                         dq: ndarray) -> Tuple[Q, Q]:
    """Calculates a position based angular update."""

    th = arrop.norm(dq)
    n = dq / (th + 1e-6)

    # ignoring inertial effects for now
    w1 = dot(n, self.body_p.inertia * n)
    w2 = dot(n, self.body_c.inertia * n)

    dlambda = -th / (w1 + w2 + 1e-6)
    p = -dlambda * n

    dq_p_pos = np.zeros_like(p)
    dq_p_rot = .5 * math.vec_quat_mul(self.body_p.inertia * p, qp_p.rot)

    dq_c_pos = np.zeros_like(p)
    dq_c_rot = -.5 * math.vec_quat_mul(self.body_c.inertia * p, qp_c.rot)

    dq_p = Q(pos=self.scale_ang * dq_p_pos, rot=self.scale_ang * dq_p_rot)
    dq_c = Q(pos=self.scale_ang * dq_c_pos, rot=self.scale_ang * dq_c_rot)
    return dq_p, dq_c

  def apply_position_update(self, qp_p: QP, qp_c: QP, pos_p: ndarray,
                            pos_c: ndarray) -> Tuple[Q, Q]:
    """Calculates a position based positional update.

    Args:
      qp_p: First body participating in update
      qp_c: Second body participating in update:
      pos_p: World space location on first body
      pos_c: World space location on second body
    Returns: A position based update that adjusts qp_p and qp_c so that pos_p
      and pos_c are the same in world space.
    """

    dx = pos_p - pos_c
    pos_p = pos_p - qp_p.pos
    pos_c = pos_c - qp_c.pos

    c = norm(dx)
    n = dx / (c + 1e-6)

    # only treating spherical inertias
    cr1 = cross(pos_p, n)
    w1 = (1. / self.body_p.mass) + dot(cr1, self.body_p.inertia * cr1)

    cr2 = cross(pos_c, n)
    w2 = (1. / self.body_c.mass) + dot(cr2, self.body_c.inertia * cr2)

    dlambda = -c / (w1 + w2 + 1e-6)
    p = dlambda * n

    dq_p_pos = p / self.body_p.mass
    dq_p_rot = .5 * math.vec_quat_mul(self.body_p.inertia * cross(pos_p, p),
                                      qp_p.rot)

    dq_c_pos = -p / self.body_c.mass
    dq_c_rot = -.5 * math.vec_quat_mul(self.body_c.inertia * cross(pos_c, p),
                                       qp_c.rot)

    dq_p = Q(pos=self.scale_pos * dq_p_pos, rot=self.scale_pos * dq_p_rot)
    dq_c = Q(pos=self.scale_pos * dq_c_pos, rot=self.scale_pos * dq_c_rot)

    return dq_p, dq_c

  def angle_vel(self, qp: QP) -> Tuple[Any, Any]:
    """Returns joint angle and velocity.

    Args:
      qp: State data for system

    Returns:
      angle: n-tuple of joint angles where n = # DoF of the joint
      vel: n-tuple of joint velocities where n = # DoF of the joint
    """

    def op(joint, qp_p, qp_c):
      axes, angles = joint.axis_angle(qp_p, qp_c)
      vels = tuple([dot(qp_p.ang - qp_c.ang, axis) for axis in axes])
      return array(angles), array(vels)

    qp_p = np.take(qp, self.body_p.idx)
    qp_c = np.take(qp, self.body_c.idx)
    
    angles, vels = op(self, qp_p, qp_c)
    angles, vels = angles.reshape(-1), vels.reshape(-1)

    if self.free_dofs is not None:
      # constructs a flat list of indices that don't have [0, 0] angle limits
      dof_indices = arrop.concatenate([
          arange(i * self.dof, i * self.dof + dw)
          for i, dw in enumerate(self.free_dofs)
      ])
      
      # TODO: check if the take works here?
      angles, vels = np.take((angles, vels), dof_indices)
    return angles, vels

  @abc.abstractmethod
  def apply_reduced(self, qp_p: QP,
                    qp_c: QP) -> Union[Tuple[P, P], Tuple[Q, Q]]:
    """Returns impulses to constrain and align bodies connected by a joint.

    Operates in reduced joint space.

    Args:
      qp_p: Joint parent state data
      qp_c: Joint child state data

    Returns:
      dp_p: Joint parent impulse
      dp_c: Joint child impulse
    """

  @abc.abstractmethod
  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint.

    vmap across axis_angle to get all joints.

    Args:
      qp_p: State for parent body
      qp_c: State for child body

    Returns:
      axis: n-tuple of joint axes where n = # DoF of the joint
      angle: n-tuple of joint angles where n = # DoF of the joint
    """



class Revolute(Joint):
  """A revolute joint constrains two bodies around a single axis.

  Constructs a revolute joint where the parent's local x-axis is constrained
  to point in the same direction as the child's local x-axis.  This construction
  follows the line of nodes convention shared by the universal and spherical
  joints for x-y'-z'' intrinsic euler angles.
  """

  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[Q, Q]:
    """Constructs the position based constraint update for a parent and child."""

    # positional constraints

    pos_p, _ = qp_p.to_world(self.off_p)
    pos_c, _ = qp_c.to_world(self.off_c)

    dq_p, dq_c = self.apply_position_update(qp_p, qp_c, pos_p, pos_c)

    # angular constraints

    axis = math.rotate(self.axis_p[0], qp_p.rot)
    ref_p = math.rotate(self.axis_p[2], qp_p.rot)
    ref_c = math.rotate(self.axis_c[2], qp_c.rot)

    psi = math.signed_angle(axis, ref_p, ref_c)
    axis_c = math.rotate(self.axis_c[0], qp_c.rot)

    dq_1 = arrop.cross(axis, axis_c)

    # limit constraints

    ph = arrop.clip(psi, self.limit[0][0], self.limit[0][1])
    fixrot = math.quat_rot_axis(axis, ph)
    n1 = math.rotate(ref_p, fixrot)
    dq_2 = arrop.cross(n1, ref_c)

    v_apply = arrop.torchVmap(self.apply_angle_update, [False, False, True])
    dq_p_ang, dq_c_ang = v_apply(qp_p, qp_c, np.array([dq_1, dq_2]))

    dq_p += Q(
        pos=dq_p_ang.pos[0] + dq_p_ang.pos[1],
        rot=dq_p_ang.rot[0] + dq_p_ang.rot[1])

    dq_c += Q(
        pos=dq_c_ang.pos[0] + dq_c_ang.pos[1],
        rot=dq_c_ang.rot[0] + dq_c_ang.rot[1])

    return dq_p, dq_c  # pytype: disable=bad-return-type

  def axis_angle(self, qp_p: QP, qp_c: QP) -> Tuple[Any, Any]:
    """Returns axes and angles of a single joint."""
    axis_p = math.rotate(self.axis_p[0], qp_p.rot)
    ref_p = math.rotate(self.axis_p[2], qp_p.rot)
    ref_c = math.rotate(self.axis_c[2], qp_c.rot)
    # algebraically the same as the calculation in `Spherical`, but simpler
    # because child local-x and parent local-x are constrained to be the same
    psi = math.signed_angle(axis_p, ref_p, ref_c)
    return (axis_p,), (psi,)


class Spherical(Joint):
  """A spherical joint constrains two bodies around three axes.

  Constructs a spherical joint which returns intrinsic euler angles in the
    x-y'-z'' convention between the parent and child.  Uses the line of nodes
    construction described in section 3.2.3.2 here:
    https://www.sedris.org/wg8home/Documents/WG80485.pdf
  """

  def apply_reduced(self, qp_p: QP, qp_c: QP) -> Tuple[Q, Q]:

    # positional constraints

    pos_p, _ = qp_p.to_world(self.off_p)
    pos_c, _ = qp_c.to_world(self.off_c)

    dq_p, dq_c = self.apply_position_update(qp_p, qp_c, pos_p, pos_c)

    # angular constraints

    def limit_angle(n, n_1, n_2, lim_num):

      ph = math.signed_angle(n, n_1, n_2)
      mask = arrop.where(ph < self.limit[lim_num][0], 1., 0.)
      mask = arrop.where(ph > self.limit[lim_num][1], 1., mask)
      ph = arrop.clip(ph, self.limit[lim_num][0], self.limit[lim_num][1])
      fixrot = math.quat_rot_axis(n, ph)
      n1 = math.rotate(n_1, fixrot)
      dq = arrop.cross(n1, n_2) * mask
      dq_p_ang, dq_c_ang = self.apply_angle_update(qp_p, qp_c, dq)

      return dq_p_ang, dq_c_ang

    v_rot = arrop.torchVmap(math.rotate, include=[True, False])
    axis_p_rotated = v_rot(self.axis_p, qp_p.rot)
    axis_c_rotated = v_rot(self.axis_c, qp_c.rot)
    axis_1_p = axis_p_rotated[0]
    axis_2_p = axis_p_rotated[1]
    axis_1_c = axis_c_rotated[0]
    axis_2_c = axis_c_rotated[1]
    axis_3_c = axis_c_rotated[2]

    line_of_nodes = arrop.cross(axis_3_c, axis_1_p)
    line_of_nodes = line_of_nodes / (1e-6 + arrop.norm(line_of_nodes))
    y_n_normal = axis_1_p
    axis_1_p_in_xz_c = arrop.dot(axis_1_p, axis_1_c) * axis_1_c + arrop.dot(
        axis_1_p, axis_2_c) * axis_2_c
    axis_1_p_in_xz_c = axis_1_p_in_xz_c / (1e-6 +
                                           arrop.norm(axis_1_p_in_xz_c))
    axis_2_normal = arrop.cross(axis_1_p_in_xz_c, axis_1_p)
    axis_2_normal = axis_2_normal / (1e-6 + arrop.norm(axis_2_normal))
    yc_n_normal = -axis_3_c

    dq_p_ang_1, dq_c_ang_1 = limit_angle(y_n_normal, axis_2_p, line_of_nodes, 0)
    dq_p_ang_2, dq_c_ang_2 = limit_angle(
        -axis_2_normal * arrop.sign(arrop.dot(axis_1_p, axis_3_c)), axis_1_p,
        axis_1_p_in_xz_c, 1)
    dq_p_ang_3, dq_c_ang_3 = limit_angle(-yc_n_normal, line_of_nodes, axis_2_c,
                                         2)

    dq_p += dq_p_ang_1 + dq_p_ang_2 + dq_p_ang_3
    dq_c += dq_c_ang_1 + dq_c_ang_2 + dq_c_ang_3

    return dq_p, dq_c  # pytype: disable=bad-return-type

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

  dofs = {len(j.angle_limit) for j in config.joints}
  # check to see if we should make all joints spherical
  sphericalize = len(dofs) > 1  # multiple joint types present
  sphericalize |= 2 in dofs  # ... or 2-dof joints present
  sphericalize &= config.dynamics_mode == 'pbd'  # only sphericalize if pbd
  for joint in config.joints:
    dof = len(joint.angle_limit)
    if config.dynamics_mode == 'pbd':
      free_dofs = dof
      while sphericalize and dof < 3:
        joint.angle_limit.add()
        dof += 1
      if dof not in joints:
        joints[dof] = {'joint': [], 'free_dofs': []}
      joints[dof]['joint'].append(joint)
      joints[dof]['free_dofs'].append(free_dofs)

  # ensure stable order for joint application: dof
  joints = sorted(joints.items(), key=lambda kv: kv[0])
  ret = []

  solver_scale_pos = config.solver_scale_pos or .6
  solver_scale_ang = config.solver_scale_ang or .2

  for dof, v in joints:
    if dof == 1:
      ret.append(
          Revolute(
              v['joint'],
              body,
              free_dofs=None,
              solver_scale_pos=solver_scale_pos,
              solver_scale_ang=solver_scale_ang))
    elif dof == 2:
      ret.append(
          Spherical(
              v['joint'],
              body,
              free_dofs=None,
              solver_scale_pos=solver_scale_pos,
              solver_scale_ang=solver_scale_ang))
    elif dof == 3:
      ret.append(
          Spherical(
              v['joint'],
              body,
              free_dofs=v.get('free_dofs'),
              solver_scale_pos=solver_scale_pos,
              solver_scale_ang=solver_scale_ang))
    else:
      raise RuntimeError(f'invalid number of joint limits: {dof}')

  return ret
