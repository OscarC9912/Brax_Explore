"""Contains Env that Torch Script Needs to Use"""

from typing import Any, Dict, Optional
import numpy as np
from numpy import ndarray
import abc
from google.protobuf import text_format

from physics.torch_base import QP
from physics.torch_system import torchSystem
from physics.torch_config_pb2 import Config

class torchState:
    """Environment state for training and inference."""
    qp: QP
    obs: ndarray
    reward: ndarray
    done: ndarray
    metrics: Dict[str, ndarray]
    info: Dict[str, Any]
    
    def __init__(self, qp: QP, obs: ndarray, reward: ndarray, done: ndarray, metrics: Dict[str, ndarray]):
        self.qp = qp
        self.obs = obs
        self.reward = reward
        self.done = done
        self.metrics = metrics
        
        
    def replace(self, changes: Dict[Any, Any]) -> None:
        """Replace the attribute of the state by new one if there's any in changes """
        for arg in changes:
            if arg == 'qp':
                self.qp = changes['qp']
            elif arg == 'obs':
                self.obs = changes['obs']
            elif arg == 'reward':
                self.reward = changes['reward']
            elif arg == 'done':
                self.done = changes['done']
            elif arg == 'metrics':
                self.metrics = changes['metrics']
                
                
                
class Env(abc.ABC):
  """API for driving a brax system for training and inference."""

  def __init__(self, config: Optional[str]):
    if config:
      config = text_format.Parse(config, Config())
      self.sys = torchSystem(config)

  @abc.abstractmethod
  def reset(self, rng: ndarray) -> torchState:
    """Resets the environment to an initial state."""

  @abc.abstractmethod
  def step(self, state: torchState, action: ndarray) -> torchState:
    """Run one timestep of the environment's dynamics."""

  @property
  def observation_size(self) -> int:
    """The size of the observation vector returned in step and reset."""
    rng = np.random_prngkey(0)
    reset_state = self.unwrapped.reset(rng)
    return reset_state.obs.shape[-1]

  @property
  def action_size(self) -> int:
    """The size of the action vector expected by step."""
    return self.sys.num_joint_dof + self.sys.num_forces_dof

  @property
  def unwrapped(self) -> 'Env':
    return self
        
        