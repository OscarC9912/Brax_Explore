import gym
import numpy as np
from google.protobuf import text_format

import ant_util as ant_util
import physics.torch_config_pb2 as Config
from physics.torch_system import torchSystem
from physics.torch_base import P, Q, QP, Info
import torch_math.array_ops as arrop
from torch_env.torchEnv import torchState
from torch_env.torchEnv import Env


class AntTorch(Env):  # this was originally gym.Env
    """
    torch Implemented: Ant Environment
    """

    def __init__(self,
                ctrl_cost_weight=0.5,
                use_contact_forces=False,
                contact_cost_weight=5e-4,
                healthy_reward=1.0,
                terminate_when_unhealthy=True,
                healthy_z_range=(0.2, 1.0),
                reset_noise_scale=0.1,
                exclude_current_positions_from_observation=True,
                legacy_spring=False,
                **kwargs):
        
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Add the following paramters for the state
        # config = text_format.Parse(config,  Config())
        self.state = torchState
        
        self.sys = torchSystem(config)


    def reset(self, rng: np.ndarray):
        """
        Reset the environment to the initial state
        """
        rng, rng1, rng2 = ant_util.random_split(rng, 3)
        
        qpos = self.sys.default_angle() + self._noise(rng1)
        qvel = self._noise(rng2)
        
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        obs = self._get_obs(qp, self.sys.info(qp))
        reward, done, zero = np.zeros(3)
        
        metrics = {
          'reward_forward': zero,
          'reward_survive': zero,
          'reward_ctrl': zero,
          'reward_contact': zero,
          'x_position': zero,
          'y_position': zero,
          'distance_from_origin': zero,
          'x_velocity': zero,
          'y_velocity': zero,
          'forward_reward': zero,
        }
        
        self.state = torchState(qp, obs, reward, done, metrics)
        return self.state


    def step(self, action: np.ndarray):
        """Run one timestep of the environment's dynamics."""
        # now we can access the state by self
        qp, info = self.sys.step(self.state.qp, action)
        
        velocity = (qp.pos[0] - self.state.qp.pos[0]) / self.sys.config.dt
        forward_reward = velocity[0]
        
        min_z, max_z = self._healthy_z_range
        is_healthy = arrop.where(qp.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = arrop.where(qp.pos[0, 2] > max_z, x=0.0, y=is_healthy)
        if self._terminate_when_unhealthy:
          healthy_reward = self._healthy_reward
        else:
          healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * arrop.sum(arrop.square(action))
        contact_cost = (self._contact_cost_weight *
                        arrop.sum(arrop.square(arrop.clip(info.contact.vel, -1, 1))))
        obs = self._get_obs(qp, info)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
        self.state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=qp.pos[0, 0],
            y_position=qp.pos[0, 1],
            distance_from_origin=arrop.norm(qp.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
        )
        
        changes_dict = {'qp': qp, 'obs': obs, 'reward': reward, 'done': done}
        self.state.replace(changes_dict)

        return self.state


    def _get_obs(self, qp: QP, info: Info) -> np.ndarray:
      """Observe ant body position and velocities."""
      joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

      # qpos: position and orientation of the torso and the joint angles.
      if self._exclude_current_positions_from_observation:
        qpos = [qp.pos[0, 2:], qp.rot[0], joint_angle]
      else:
        qpos = [qp.pos[0], qp.rot[0], joint_angle]

      # qvel: velocity of the torso and the joint angle velocities.
      qvel = [qp.vel[0], qp.ang[0], joint_vel]

      # external contact forces:
      # delta velocity (3,), delta ang (3,) * 10 bodies in the system
      if self._use_contact_forces:
        cfrc = [
            arrop.clip(info.contact.vel, -1, 1),
            arrop.clip(info.contact.ang, -1, 1)
        ]
        # flatten bottom dimension
        cfrc = [arrop.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]
      else:
        cfrc = []

      return arrop.concatenate(qpos + qvel + cfrc)

    def _noise(self, rng):
      low, hi = -self._reset_noise_scale, self._reset_noise_scale
      return arrop.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)



_SYSTEM_CONFIG = """
  bodies {
    name: "$ Torso"
    colliders {
      capsule {
        radius: 0.25
        length: 0.5
        end: 1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 10
  }
  bodies {
    name: "Aux 1"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 4"
    colliders {
      rotation { x: 90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 2"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 7"
    colliders {
      rotation { x: 90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 3"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 10"
    colliders {
      rotation { x: -90 y: 45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Aux 4"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.4428427219390869
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "$ Body 13"
    colliders {
      rotation { x: -90 y: -45 }
      capsule {
        radius: 0.08
        length: 0.7256854176521301
        end: -1
      }
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
  }
  bodies {
    name: "Ground"
    colliders {
      plane {}
    }
    inertia { x: 1.0 y: 1.0 z: 1.0 }
    mass: 1
    frozen { all: true }
  }
  joints {
    name: "hip_1"
    parent_offset { x: 0.2 y: 0.2 }
    child_offset { x: -0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 1"
    angle_limit { min: -30.0 max: 30.0 }
    rotation { y: -90 }
    angular_damping: 20
  }
  joints {
    name: "ankle_1"
    parent_offset { x: 0.1 y: 0.1 }
    child_offset { x: -0.2 y: -0.2 }
    parent: "Aux 1"
    child: "$ Body 4"
    rotation: { z: 135 }
    angle_limit {
      min: 30.0
      max: 70.0
    }
    angular_damping: 20
  }
  joints {
    name: "hip_2"
    parent_offset { x: -0.2 y: 0.2 }
    child_offset { x: 0.1 y: -0.1 }
    parent: "$ Torso"
    child: "Aux 2"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_2"
    parent_offset { x: -0.1 y: 0.1 }
    child_offset { x: 0.2 y: -0.2 }
    parent: "Aux 2"
    child: "$ Body 7"
    rotation { z: 45 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_3"
    parent_offset { x: -0.2 y: -0.2 }
    child_offset { x: 0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 3"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_3"
    parent_offset { x: -0.1 y: -0.1 }
    child_offset {
      x: 0.2
      y: 0.2
    }
    parent: "Aux 3"
    child: "$ Body 10"
    rotation { z: 135 }
    angle_limit { min: -70.0 max: -30.0 }
    angular_damping: 20
  }
  joints {
    name: "hip_4"
    parent_offset { x: 0.2 y: -0.2 }
    child_offset { x: -0.1 y: 0.1 }
    parent: "$ Torso"
    child: "Aux 4"
    rotation { y: -90 }
    angle_limit { min: -30.0 max: 30.0 }
    angular_damping: 20
  }
  joints {
    name: "ankle_4"
    parent_offset { x: 0.1 y: -0.1 }
    child_offset { x: -0.2 y: 0.2 }
    parent: "Aux 4"
    child: "$ Body 13"
    rotation { z: 45 }
    angle_limit { min: 30.0 max: 70.0 }
    angular_damping: 20
  }
  actuators {
    name: "hip_1"
    joint: "hip_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_1"
    joint: "ankle_1"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_2"
    joint: "hip_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_2"
    joint: "ankle_2"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_3"
    joint: "hip_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_3"
    joint: "ankle_3"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "hip_4"
    joint: "hip_4"
    strength: 350.0
    torque {}
  }
  actuators {
    name: "ankle_4"
    joint: "ankle_4"
    strength: 350.0
    torque {}
  }
  friction: 1.0
  gravity { z: -9.8 }
  angular_damping: -0.05
  collide_include {
    first: "$ Torso"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 4"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 7"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 10"
    second: "Ground"
  }
  collide_include {
    first: "$ Body 13"
    second: "Ground"
  }
  dt: 0.05
  substeps: 10
  dynamics_mode: "pbd"
  """

_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "$ Torso"
  colliders {
    capsule {
      radius: 0.25
      length: 0.5
      end: 1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 10
}
bodies {
  name: "Aux 1"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 4"
  colliders {
    rotation { x: 90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 2"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 7"
  colliders {
    rotation { x: 90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 3"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 10"
  colliders {
    rotation { x: -90 y: 45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Aux 4"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.4428427219390869
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "$ Body 13"
  colliders {
    rotation { x: -90 y: -45 }
    capsule {
      radius: 0.08
      length: 0.7256854176521301
      end: -1
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}
bodies {
  name: "Ground"
  colliders {
    plane {}
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
  frozen { all: true }
}
joints {
  name: "$ Torso_Aux 1"
  parent_offset { x: 0.2 y: 0.2 }
  child_offset { x: -0.1 y: -0.1 }
  parent: "$ Torso"
  child: "Aux 1"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  angle_limit { min: -30.0 max: 30.0 }
  rotation { y: -90 }
}
joints {
  name: "Aux 1_$ Body 4"
  parent_offset { x: 0.1 y: 0.1 }
  child_offset { x: -0.2 y: -0.2 }
  parent: "Aux 1"
  child: "$ Body 4"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation: { z: 135 }
  angle_limit {
    min: 30.0
    max: 70.0
  }
}
joints {
  name: "$ Torso_Aux 2"
  parent_offset { x: -0.2 y: 0.2 }
  child_offset { x: 0.1 y: -0.1 }
  parent: "$ Torso"
  child: "Aux 2"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 2_$ Body 7"
  parent_offset { x: -0.1 y: 0.1 }
  child_offset { x: 0.2 y: -0.2 }
  parent: "Aux 2"
  child: "$ Body 7"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 45 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "$ Torso_Aux 3"
  parent_offset { x: -0.2 y: -0.2 }
  child_offset { x: 0.1 y: 0.1 }
  parent: "$ Torso"
  child: "Aux 3"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 3_$ Body 10"
  parent_offset { x: -0.1 y: -0.1 }
  child_offset {
    x: 0.2
    y: 0.2
  }
  parent: "Aux 3"
  child: "$ Body 10"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 135 }
  angle_limit { min: -70.0 max: -30.0 }
}
joints {
  name: "$ Torso_Aux 4"
  parent_offset { x: 0.2 y: -0.2 }
  child_offset { x: -0.1 y: 0.1 }
  parent: "$ Torso"
  child: "Aux 4"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { y: -90 }
  angle_limit { min: -30.0 max: 30.0 }
}
joints {
  name: "Aux 4_$ Body 13"
  parent_offset { x: 0.1 y: -0.1 }
  child_offset { x: -0.2 y: 0.2 }
  parent: "Aux 4"
  child: "$ Body 13"
  stiffness: 18000.0
  angular_damping: 20
  spring_damping: 80
  rotation { z: 45 }
  angle_limit { min: 30.0 max: 70.0 }
}
actuators {
  name: "$ Torso_Aux 1"
  joint: "$ Torso_Aux 1"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 1_$ Body 4"
  joint: "Aux 1_$ Body 4"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 2"
  joint: "$ Torso_Aux 2"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 2_$ Body 7"
  joint: "Aux 2_$ Body 7"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 3"
  joint: "$ Torso_Aux 3"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 3_$ Body 10"
  joint: "Aux 3_$ Body 10"
  strength: 350.0
  torque {}
}
actuators {
  name: "$ Torso_Aux 4"
  joint: "$ Torso_Aux 4"
  strength: 350.0
  torque {}
}
actuators {
  name: "Aux 4_$ Body 13"
  joint: "Aux 4_$ Body 13"
  strength: 350.0
  torque {}
}
friction: 1.0
gravity { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "$ Torso"
  second: "Ground"
}
collide_include {
  first: "$ Body 4"
  second: "Ground"
}
collide_include {
  first: "$ Body 7"
  second: "Ground"
}
collide_include {
  first: "$ Body 10"
  second: "Ground"
}
collide_include {
  first: "$ Body 13"
  second: "Ground"
}
dt: 0.05
substeps: 10
dynamics_mode: "legacy_spring"
"""






# if __name__ == '__main__':
#     print(1)

#     # import torch
#     # import html

#     # # env = gym.make('Ant-v4')

#     # env = AntTorch()
#     # a = torch.zeros(env.action_space)
 
#     # for i in range(10):
#     #     qp = env.step(a)
#     #     # print(qp)


#     # with open('index.html', 'w') as f:
#     #   f.write(html.render(env.sys, [qp]))


