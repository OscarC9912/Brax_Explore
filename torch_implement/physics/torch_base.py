import copy
from typing import Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import torch
from physics import torch_config_pb2
from torch_math import math
import torch_io.mesh as mesh

# === Implementation for the state object to use ===
class Q(object):
  """Coordinates: position and rotation.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
  """
  
  pos: np.ndarray
  rot: np.ndarray

  def __add__(self, o):

    if isinstance(o, P):
      return QP(self.pos, self.rot, o.vel, o.ang)
    elif isinstance(o, Q):
      return Q(self.pos + o.pos, self.rot + o.rot)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, o.vel, o.ang)
    else:
      raise ValueError('add only supported for P, Q, QP')


class P(object):
  """Time derivatives: velocity and angular velocity.

  Attributes:
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  vel: np.ndarray
  ang: np.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return P(self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(o.pos, o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(o.pos, o.rot, self.vel + o.vel, self.ang + o.ang)
    else:
      raise ValueError('add only supported for P, Q, QP')

  def __mul__(self, o):
    return P(self.vel * o, self.ang * o)



class QP(object):
  """A coordinate and time derivative frame for a brax body.

  Attributes:
    pos: Location of center of mass.
    rot: Rotation about center of mass, represented as a quaternion.
    vel: Velocity.
    ang: Angular velocity about center of mass.
  """
  pos: np.ndarray
  rot: np.ndarray
  vel: np.ndarray
  ang: np.ndarray

  def __add__(self, o):
    if isinstance(o, P):
      return QP(self.pos, self.rot, self.vel + o.vel, self.ang + o.ang)
    elif isinstance(o, Q):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel, self.ang)
    elif isinstance(o, QP):
      return QP(self.pos + o.pos, self.rot + o.rot, self.vel + o.vel,
                self.ang + o.ang)
    else:
      raise ValueError('add only supported for P, Q, QP')

  def __mul__(self, o):
    return QP(self.pos * o, self.rot * o, self.vel * o, self.ang * o)

  def zero(cls, shape=()):
    return cls(
        pos=np.zeros(shape + (3,)),
        rot=np.tile(np.array([1., 0., 0., 0]), reps=shape + (1,)),
        vel=np.zeros(shape + (3,)),
        ang=np.zeros(shape + (3,)))

  def to_world(self, rpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns world information about a point relative to a part.

    Args:
      rpos: Point relative to center of mass of part.

    Returns:
      A 2-tuple containing:
        * World-space coordinates of rpos
        * World-space velocity of rpos
    """
    rpos_off = math.rotate(rpos, self.rot)
    rvel = math.cross(self.ang, rpos_off)
    return (self.pos + rpos_off, self.vel + rvel)

  def world_velocity(self, pos: np.ndarray) -> np.ndarray:
    """Returns the velocity of the point on a rigidbody in world space.

    Args:
      pos: World space position which to use for velocity calculation.
    """
    return self.vel + math.cross(self.ang, pos - self.pos)


class Info(object):
  """Auxilliary data calculated during the dynamics of each physics step.

  Attributes:
    contact: External contact forces applied at a step
    joint: Joint constraint forces applied at a step
    actuator: Actuator forces applied at a step
  """
  contact: P
  joint: Union[P, Q]
  actuator: P


      












def validate_config(
    config: torch_config_pb2.Config,
    resource_paths: Optional[Sequence[str]] = None) -> torch_config_pb2.Config:
  """Validate and normalize config settings for use in systems."""
  config = copy.deepcopy(config)

  if config.dt <= 0:
    raise ValueError('config.dt must be positive')

  if config.substeps == 0:
    config.substeps = 1

  def find_dupes(objs):
    names = set()
    for obj in objs:
      if obj.name in names:
        raise RuntimeError(f'duplicate name in config: {obj.name}')
      names.add(obj.name)

  find_dupes(config.bodies)
  find_dupes(config.joints)
  find_dupes(config.actuators)
  find_dupes(config.mesh_geometries)

  if config.dynamics_mode == 'legacy_spring':
    if any(j.stiffness == 0 for j in config.joints):
      raise ValueError(
          'joint.stiffness must be >0 when dynamics_mode == legacy_spring')
  elif config.dynamics_mode == 'pbd':
    if any(j.stiffness != 0 for j in config.joints):
      raise ValueError('joint.stiffness is invalid when dynamics_mode == pbd')
    if config.baumgarte_erp:
      raise ValueError('baumgarte_erp is invalid when dynamics_mode == pbd')
  elif any(j.stiffness != 0 for j in config.joints):
    config.dynamics_mode = 'legacy_spring'
    warnings.warn('dynamics_mode not specified, but joint.stiffness >0. '
                  'Setting dynamics_mode="legacy_spring".')
  else:
    config.dynamics_mode = 'pbd'
    warnings.warn(
        'dynamics_mode either not specified or not recognized, defaulting to '
        '"pbd".  If you wish to preserve legacy behavior used in previous '
        'versions of Brax, set dynamics_mode="legacy_spring".'
    )

  # Load the meshes.
  if resource_paths is None:
    resource_paths = ['']
  for i in range(len(config.mesh_geometries)):
    path = config.mesh_geometries[i].path
    if not path:
      continue
    mesh_geom = mesh.load(config.mesh_geometries[i].name, path, resource_paths)
    config.mesh_geometries[i].CopyFrom(mesh_geom)

  # TODO: more config validation

  # reify all frozen dimensions in the system
  allvec = torch_config_pb2.Vector3(x=1.0, y=1.0, z=1.0)
  frozen = config.frozen
  if frozen.all:
    frozen.position.CopyFrom(allvec)
    frozen.rotation.CopyFrom(allvec)
  if all([
      frozen.position.x, frozen.position.y, frozen.position.z,
      frozen.rotation.x, frozen.rotation.y, frozen.rotation.z
  ]):
    config.frozen.all = True
  for b in config.bodies:
    inertia = b.inertia
    if inertia.x == 0 and inertia.y == 0 and inertia.z == 0:
      b.inertia.x, b.inertia.y, b.inertia.z = 1, 1, 1

    b.frozen.position.x = b.frozen.position.x or frozen.position.x
    b.frozen.position.y = b.frozen.position.y or frozen.position.y
    b.frozen.position.z = b.frozen.position.z or frozen.position.z
    b.frozen.rotation.x = b.frozen.rotation.x or frozen.rotation.x
    b.frozen.rotation.y = b.frozen.rotation.y or frozen.rotation.y
    b.frozen.rotation.z = b.frozen.rotation.z or frozen.rotation.z
    if b.frozen.all:
      b.frozen.position.CopyFrom(allvec)
      b.frozen.rotation.CopyFrom(allvec)
    if all([
        b.frozen.position.x, b.frozen.position.y, b.frozen.position.z,
        b.frozen.rotation.x, b.frozen.rotation.y, b.frozen.rotation.z
    ]):
      b.frozen.all = True

    # insert material properties to colliders
    for c in b.colliders:
      if not c.HasField('material'):
        c.material.friction = config.friction
        c.material.elasticity = config.elasticity

  frozen.all = all(b.frozen.all for b in config.bodies)

  return config

def vec_to_arr(vec: torch_config_pb2.Vector3) -> np.ndarray:
    return np.array([vec.x, vec.y, vec.z])




if __name__ == '__main__':

    obj = Q()

    obj.pos