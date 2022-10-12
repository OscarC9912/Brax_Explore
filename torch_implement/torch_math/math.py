# Include all the Math Algorithm and Array Operations
from turtle import shape
import numpy as np
import torch
from numpy import ndarray



Vector3 = np.ndarray
Quaternion = np.ndarray

def cross(x: ndarray, y: ndarray):
    """
    return the cross product of two array
    """
    return np.dot(x, y)


def rotate(vec: np.ndarray, quat: np.ndarray):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise AssertionError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
  item2 = 2 * s * np.cross(u, vec)
  r = r + item2
  return r


def amin(x: ndarray):
    return amin(x) 
  
  

def euler_to_quat(v: ndarray) -> ndarray:
  """Converts euler rotations in degrees to quaternion."""
  # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
  c1, c2, c3 = np.cos(v * np.pi / 360)
  s1, s2, s3 = np.sin(v * np.pi / 360)
  w = c1 * c2 * c3 - s1 * s2 * s3
  x = s1 * c2 * c3 + c1 * s2 * s3
  y = c1 * s2 * c3 - s1 * c2 * s3
  z = c1 * c2 * s3 + s1 * s2 * c3
  return np.array([w, x, y, z])


def vec_quat_mul(u: ndarray, v: ndarray) -> ndarray:
  """Multiplies a vector and a quaternion.

  This is a convenience method for multiplying two quaternions when
  one of the quaternions has a 0-value w-part, i.e.:
  quat_mul([0.,a,b,c], [d,e,f,g])

  It is slightly more efficient than constructing a 0-w-part quaternion
  from the vector.

  Args:
    u: (3,) vector representation of the quaternion (0.,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return np.array([
      -u[0] * v[1] - u[1] * v[2] - u[2] * v[3],
      u[0] * v[0] + u[1] * v[3] - u[2] * v[2],
      -u[0] * v[3] + u[1] * v[0] + u[2] * v[1],
      u[0] * v[2] - u[1] * v[1] + u[2] * v[0],
  ])
  
  
  
  
  
def ang_to_quat(ang: Vector3):
  """Converts angular velocity to a quaternion.

  Args:
    ang: (3,) angular velocity

  Returns:
    A rotation quaternion.
  """
  return np.array([0, ang[0], ang[1], ang[2]])


def quat_mul(u: Quaternion, v: Quaternion) -> Quaternion:
  """Multiplies two quaternions.

  Args:
    u: (4,) quaternion (w,x,y,z)
    v: (4,) quaternion (w,x,y,z)

  Returns:
    A quaternion u * v.
  """
  return np.array([
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])
  
  
def relative_quat(q1: Quaternion, q2: Quaternion) -> Quaternion:
  """Returns the relative quaternion from q1 to q2."""
  return quat_mul(q2, quat_inv(q1))


def quat_inv(q: Quaternion) -> Quaternion:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * np.array([1, -1, -1, -1])

