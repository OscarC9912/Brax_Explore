from sqlite3 import OperationalError
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union
import builtins
import numpy as np
from numpy import ndarray
import torch

import jax
from jax import core

pi = np.pi
inf = np.inf
float32 = np.float32
int32 = np.int32


F = TypeVar('F', bound=Callable)


def _in_jit() -> bool:
  """Returns true if currently inside a jax.jit call or jit is disabled."""
  if jax.config.jax_disable_jit:
    return True
  return core.cur_sublevel().level > 0


def torchVmap(fun: F, include: Optional[Sequence[bool]] = None) -> F:
  """Creates a function which maps ``fun`` over argument axes."""
  if _in_jit():
    in_axes = 0
    if include:
        in_axes = [0 if inc else None for inc in include]
    return torch.vmap(fun, in_axes=in_axes)


Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def norm(x: ndarray,
         axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Returns the array norm."""
  x = torch.from_numpy(x)
  return torch.linalg.norm(x)


def index_update(x: ndarray, idx: ndarray, y: ndarray) -> ndarray:
  """Pure equivalent of x[idx] = y."""
  x = np.copy(x)
  x[idx] = y
  return x


def safe_norm(x: ndarray,
              axis: Optional[Union[Tuple[int, ...], int]] = None) -> ndarray:
  """Calculates a linalg.norm(x) that's safe for gradients at x=0.
  Avoids a poorly defined gradient for jnp.linal.norm(0) see
  https://github.com/google/jax/issues/3058 for details
  Args:
    x: A jnp.array
    axis: The axis along which to compute the norm
  Returns:
    Norm of the array x.
  """
  x = torch.from_numpy(x)
  n = torch.linalg.norm(x, axis=axis)
  return n


def any(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether any array element along a given axis evaluates to True."""
  a = torch.from_numpy(a)
  return torch.any(a, axis=axis)


def all(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Test whether all array elements along a given axis evaluate to True."""
  a = torch.from_numpy(a)
  return torch.all(a, axis=axis)


def mean(a: ndarray, axis: Optional[int] = None) -> ndarray:
  """Compute the arithmetic mean along the specified axis."""
  a = torch.from_numpy(a)
  return torch.mean(a, axis=axis)


def arange(start: int, stop: int) -> ndarray:
  """Return evenly spaced values within a given interval."""
  return torch.arange(start, stop)


def dot(x: ndarray, y: ndarray) -> ndarray:
  """Returns dot product of two arrays."""
  x = torch.from_numpy(x)
  y = torch.from_numpy(y)
  return torch.dot(x, y)


def square(x: ndarray) -> ndarray:
  """Return the element-wise square of the input."""
  x = torch.from_numpy(x)
  return torch.square(x)



def cross(x: ndarray, y: ndarray) -> ndarray:
  """Returns cross product of two arrays."""
  x = torch.from_numpy(x)
  y = torch.from_numpy(y)
  return torch.cross(x, y)



def arccos(x: ndarray) -> ndarray:
  """Trigonometric inverse cosine, element-wise."""
  x = torch.from_numpy(x)
  return torch.arccos(x)



def safe_arccos(x: ndarray) -> ndarray:
  """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
  x = torch.from_numpy(x)
  return torch.arccos(x)


def logical_not(x: ndarray) -> ndarray:
  """Returns the truth value of NOT x element-wise."""
  x = torch.from_numpy(x)
  return torch.logical_not(x)


def multiply(x1: ndarray, x2: ndarray) -> ndarray:
  """Multiply arguments element-wise."""
  x1 = torch.from_numpy(x1)
  x2 = torch.from_numpy(x2)
  return torch.multiply(x1, x2)


def amin(x: ndarray) -> ndarray:
  """Returns the minimum along a given axis."""
  x = torch.from_numpy(x)
  return torch.amin(x)


def amax(x: ndarray) -> ndarray:
  """Returns the maximum along a given axis."""
  x = torch.from_numpy(x)
  return torch.amax(x)


def exp(x: ndarray) -> ndarray:
  """Returns the exponential of all elements in the input array."""
  x = torch.from_numpy(x)
  return torch.exp(x)


def sign(x: ndarray) -> ndarray:
  """Returns an element-wise indication of the sign of a number."""
  x = torch.from_numpy(x)
  return torch.sign(x)

def sum(a: ndarray, axis: Optional[int] = None):
  """Returns sum of array elements over a given axis."""
  a = torch.from_numpy(a)
  return torch.sum(a, axis=axis)


def random_prngkey(seed: int) -> ndarray:
  """Returns a PRNG key given a seed."""
  rng = np.random.default_rng(seed)
  return rng.integers(low=0, high=2**32, dtype='uint32', size=2)


def random_uniform(rng: ndarray,
                   shape: Tuple[int, ...] = (),
                   low: Optional[float] = 0.0,
                   high: Optional[float] = 1.0) -> ndarray:
  """Sample uniform random values in [low, high) with given shape/dtype."""
  return np.random.default_rng(rng).uniform(size=shape, low=low, high=high)




def top_k(operand: ndarray, k: int) -> ndarray:
  """Returns top k values and their indices along the last axis of operand."""
  operand = torch.from_numpy(operand)
  value, ind = torch.topk(operand, k)
  return value, ind


def concatenate(x: Sequence[ndarray], axis=0) -> ndarray:
  """Join a sequence of arrays along an existing axis."""
  x = torch.from_numpy(x)
  return torch.concatenate(x)


def sqrt(x: ndarray) -> ndarray:
  """Returns the non-negative square-root of an array, element-wise."""
  x = torch.from_numpy(x)
  return torch.sqrt(x)


def where(condition: ndarray, x: ndarray, y: ndarray) -> ndarray:
  """Return elements chosen from `x` or `y` depending on `condition`."""
  x = torch.from_numpy(x)
  y = torch.from_numpy(y)
  z = torch.from_numpy(z)
  return torch.where(condition, x, y)


def diag(v: ndarray, k: int = 0) -> ndarray:
  """Extract a diagonal or construct a diagonal array."""
  v = torch.from_numpy(v)
  return torch.diag(v, k)


def clip(a: ndarray, a_min: ndarray, a_max: ndarray) -> ndarray:
  """Clip (limit) the values in an array."""
  a = torch.from_numpy(a)
  a_min = torch.from_numpy(a_min)
  a_max = torch.from_numpy(a_max)
  return torch.clip(a, a_min, a_max)


def eye(n: int) -> ndarray:
  """Return a 2-D array with ones on the diagonal and zeros elsewhere."""
  return torch.eye(n)


def reshape(a: ndarray, newshape: Union[Tuple[int, ...], int]) -> ndarray:
  """Gives a new shape to an array without changing its data."""
  a = torch.from_numpy(a)
  return torch.reshape(a, newshape)



def segment_sum(data: ndarray,
              segment_ids: ndarray,
              num_segments: Optional[int] = None) -> ndarray:
  """Computes the sum within segments of an array."""
  if num_segments is None:
    num_segments = np.amax(segment_ids) + 1
  s = np.zeros((num_segments,) + data.shape[1:])
  np.add.at(s, segment_ids, data)   # TODO: do it with torch
  return s
