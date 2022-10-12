"""Numerical integrators."""

import abc
from typing import Optional

import torch_math.array_ops as arrop
import numpy as np
import torch_math.math as math
from physics import torch_config_pb2
from physics.torch_base import P, Q, QP, vec_to_arr


class Euler(abc.ABC):
  """Base integrator class."""

  def __init__(self, config: torch_config_pb2.Config):
    """Creates an integrator.

    Args:
      config: brax config
    """
    self.pos_mask = 1. * arrop.logical_not(
        np.array([vec_to_arr(b.frozen.position) for b in config.bodies]))
    self.rot_mask = 1. * arrop.logical_not(
        np.array([vec_to_arr(b.frozen.rotation) for b in config.bodies]))
    self.quat_mask = 1. * arrop.logical_not(
        np.array([[0.] + list(vec_to_arr(b.frozen.rotation))
                  for b in config.bodies]))
    self.dt = config.dt / config.substeps
    self.gravity = vec_to_arr(config.gravity)
    self.velocity_damping = config.velocity_damping
    self.angular_damping = config.angular_damping

  def kinetic(self, qp: QP) -> QP:
    """Performs a kinetic integration step.

    Args:
      qp: State data to be integrated

    Returns:
      State data advanced by one kinematic integration step.
    """

    def op(qp, pos_mask, rot_mask) -> QP:
      pos = qp.pos + qp.vel * self.dt * pos_mask
      rot_at_ang_quat = math.ang_to_quat(qp.ang * rot_mask) * 0.5 * self.dt
      rot = qp.rot + math.quat_mul(rot_at_ang_quat, qp.rot)
      rot = rot / arrop.norm(rot)
      return QP(pos, rot, qp.vel, qp.ang)

    return op(qp, self.pos_mask, self.rot_mask)

  def update(self,
             qp: QP,
             acc_p: Optional[P] = None,
             vel_p: Optional[P] = None,
             pos_q: Optional[Q] = None) -> QP:
    """Performs an arg dependent integrator step.

    Args:
      qp: State data to be integrated
      acc_p: Acceleration level updates to apply to qp
      vel_p: Velocity level updates to apply to qp
      pos_q: Position level updates to apply to qp

    Returns:
      State data advanced by one potential integration step.
    """

    def op_acc(qp, dp, pos_mask, rot_mask) -> QP:
      vel = arrop.exp(self.velocity_damping * self.dt) * qp.vel
      vel += (dp.vel + self.gravity) * self.dt
      vel *= pos_mask
      ang = arrop.exp(self.angular_damping * self.dt) * qp.ang
      ang += dp.ang * self.dt
      ang *= rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)


    def op_vel(qp, dp, pos_mask, rot_mask) -> QP:
      vel = (qp.vel + dp.vel) * pos_mask
      ang = (qp.ang + dp.ang) * rot_mask
      return QP(pos=qp.pos, rot=qp.rot, vel=vel, ang=ang)


    def op_pos(qp, dq, pos_mask, rot_mask) -> QP:
      qp = QP(
          pos=qp.pos + dq.pos * pos_mask,
          rot=qp.rot + dq.rot * rot_mask,
          ang=qp.ang,
          vel=qp.vel)
      return qp

    if acc_p:
      return op_acc(qp, acc_p, self.pos_mask, self.rot_mask)
    elif vel_p:
      return op_vel(qp, vel_p, self.pos_mask, self.rot_mask)
    elif pos_q:
      return op_pos(qp, pos_q, self.pos_mask, self.quat_mask)
    else:
      # no-op
      return qp

  def velocity_projection(self, qp: QP, qp_prev: QP) -> QP:
    """Performs the position based dynamics velocity projection step.

    The velocity and angular velocity must respect the spatial and quaternion
    distance (respectively) between qp and qpold.

    Args:
      qp: The current qp
      qp_prev: The qp at the previous timestep

    Returns:
      qp with velocities pinned to respect the distance traveled since qpold
    """

    def op(qp, qp_prev, pos_mask, rot_mask) -> QP:
      new_rot = qp.rot / arrop.norm(qp.rot)
      vel = ((qp.pos - qp_prev.pos) / self.dt) * pos_mask
      dq = math.relative_quat(qp_prev.rot, new_rot)
      ang = 2. * dq[1:] / self.dt
      scale = arrop.where(dq[0] >= 0., 1., -1.) * rot_mask
      ang = scale * ang * rot_mask
      return QP(pos=qp.pos, vel=vel, rot=new_rot, ang=ang)

    return op(qp, qp_prev, self.pos_mask, self.rot_mask)
