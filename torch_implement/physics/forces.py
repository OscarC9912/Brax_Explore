"""Force applied to bodies."""
from typing import List, Tuple, Union

import torch_math.array_ops as arrop
from physics import bodies
from physics import torch_config_pb2
from physics.torch_base import P, QP
import numpy as np


class Thruster:
  """Applies a force to a body."""

  def __init__(self, forces: List[torch_config_pb2.Force], body: bodies.Body,
               act_index: List[Tuple[int, int]]):
    """Thruster applies linear force to a body given a 3d act array.

    Args:
      forces: list of forces (all of the same type) to batch together
      body: (batched) bodfies for this force to act upon
      act_index: indices from the act array that drive this force
    """
    
    # self.body = jp.take(body, [body.index[f.body] for f in forces])
    self.body = body.take([body.index[f.body] for f in forces])
    
    
    self.strength = np.array([f.strength for f in forces])
    self.act_index = np.array(act_index)

  def apply_reduced(self, force: np.ndarray) -> np.ndarray:
    dvel = force * self.strength / self.body.mass
    return dvel, np.zeros_like(dvel)

  def apply(self, qp: QP, force_data: np.ndarray) -> P:
    """Applies a force to a body.

    Args:
      qp: State data for system
      force_data: Data specifying the force to apply to a body.

    Returns:
      dP: The impulses that result from apply a force to the body.
    """

    force_data = np.take(force_data, self.act_index)
    dvel, dang = arrop.torchVmap(type(self).apply_reduced)(self, force_data)

    # sum together all impulse contributions to all parts effected by forces
    dvel = arrop.segment_sum(dvel, self.body.idx, qp.pos.shape[0])
    dang = arrop.segment_sum(dang, self.body.idx, qp.pos.shape[0])

    return P(vel=dvel, ang=dang)


class Twister:
  """Applies a torque to a body."""

  def __init__(self, forces: List[torch_config_pb2.Force], body: bodies.Body,
               act_index: List[Tuple[int, int]]):
    """Twister applies torque to a single body.

    Args:
      forces: list of forces (all of the same type) to batch together
      body: (batched) bodfies for this force to act upon
      act_index: indices from the act array that drive this force
    """
    
    # self.body = jp.take(body, [body.index[f.body] for f in forces])
    self.body = body.take([body.index[f.body] for f in forces])
    
    
    self.strength = np.array([f.strength for f in forces])
    self.act_index = np.array(act_index)

  def apply_reduced(self, torque: np.ndarray) -> np.ndarray:
    dang = torque * self.strength / self.body.mass
    return np.zeros_like(dang), dang

  def apply(self, qp: QP, force_data: np.ndarray) -> P:
    """Applies a force to a body.

    Args:
      qp: State data for system
      force_data: Data specifying the force to apply to a body.

    Returns:
      dP: The impulses that result from apply a force to the body.
    """
    force_data = np.take(force_data, self.act_index)
    dvel, dang = arrop.torchVmap(type(self).apply_reduced)(self, force_data)

    # sum together all impulse contributions to all parts effected by forces
    dvel = arrop.segment_sum(dvel, self.body.idx, qp.pos.shape[0])
    dang = arrop.segment_sum(dang, self.body.idx, qp.pos.shape[0])

    return P(vel=dvel, ang=dang)


def get(config: torch_config_pb2.Config,
        body: bodies.Body) -> List[Union[Thruster, Twister]]:
  """Creates all forces given a config and actuators."""
  # by convention, force indices are after actuator indices
  # get the next available index after actuator indices
  dofs = {j.name: len(j.angle_limit) for j in config.joints}
  current_index = sum([dofs[a.joint] for a in config.actuators])

  thrusters, thruster_indices = [], []
  twisters, twister_indices = [], []
  for f in config.forces:
    act_index = tuple(range(current_index, current_index + 3))
    current_index += 3
    if f.WhichOneof('type') == 'thruster':
      thrusters.append(f)
      thruster_indices.append(act_index)
    elif f.WhichOneof('type') == 'twister':
      twisters.append(f)
      twister_indices.append(act_index)
    else:
      raise ValueError(f'unknown force type: {f.WhichOneof("type")}')

  forces = []
  if thrusters:
    forces.append(Thruster(thrusters, body, thruster_indices))
  if twisters:
    forces.append(Twister(twisters, body, twister_indices))

  return forces
