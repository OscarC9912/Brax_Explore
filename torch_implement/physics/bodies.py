"""Functionality brax body in torch implementation"""

from torch_math.math import *
from physics import torch_config_pb2
from physics.torch_base import P, Q, QP, vec_to_arr
import numpy as np


# @torchtree.register
class Body:
    """A body is a solid, non-deformable object with some mass and shape.

    Attributes:
        idx: Index of where body is found in the system.
        inertia: (3, 3) Inverse Inertia matrix represented in body frame.
        mass: Mass of the body.
        active: whether the body is effected by physics calculations
        index: name->index dict for looking up body names
    """
    
    def take(self, i):
        import copy
        new_body = copy.deepcopy(self)
        new_body.idx = np.take(self.idx, i)
        new_body.inertia = np.take(self.inertia, i)
        new_body.mass = np.take(self.mass, i)
        new_body.active = np.take(self.active, i)
        return new_body

    def __init__(self, config: torch_config_pb2.Config):
        self.idx = np.arange(0, len(config.bodies))
        self.inertia = 1. / np.array([vec_to_arr(b.inertia) for b in config.bodies])
        self.mass = np.array([b.mass for b in config.bodies])
        self.active = np.array(
            [0.0 if b.frozen.all else 1.0 for b in config.bodies])
        self.index = {b.name: i for i, b in enumerate(config.bodies)}
        
        
        
    def impulse(self, qp: QP, impulse: np.ndarray, pos: np.ndarray) -> P:
        """Calculates updates to state information based on an impulse.

        Args:
        qp: State data of the system
        impulse: Impulse vector
        pos: Location of the impulse relative to the body's center of mass

        Returns:
        dP: An impulse to apply to this body
        """
        dvel = impulse / self.mass
        dang = self.inertia * cross(pos - qp.pos, impulse)
        return P(vel=dvel, ang=dang)
    


def min_z(qp: QP, body: torch_config_pb2.Body) -> float:
    """Returns the lowest z of all the colliders in a body."""
    if not body.colliders:
        return 0.0

    result = float('inf')

    for col in body.colliders:
        if col.HasField('sphere'):
            sphere_pos = rotate(vec_to_arr(col.position), qp.rot)
            z = qp.pos[2] + sphere_pos[2] - col.sphere.radius
            result = amin(jp.array([result, z]))
        elif col.HasField('capsule'):
            rot = euler_to_quat(vec_to_arr(col.rotation))
            axis = rotate(np.array([0., 0., 1.]), rot)
            length = col.capsule.length / 2 - col.capsule.radius
            for end in (-1, 1):
                sphere_pos = vec_to_arr(col.position) + end * axis * length
                sphere_pos = rotate(sphere_pos, qp.rot)
                z = qp.pos[2] + sphere_pos[2] - col.capsule.radius
                result = amin(jp.array([result, z]))
        elif col.HasField('box'):
            corners = [(i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1)
                    for i in range(8)]
            corners = np.array(corners, dtype=float)
            for corner in corners:
                corner = corner * vec_to_arr(col.box.halfsize)
                rot = euler_to_quat(vec_to_arr(col.rotation))
                corner = rotate(corner, rot)
                corner = corner + vec_to_arr(col.position)
                corner = rotate(corner, qp.rot) + qp.pos
                result = amin(np.array([result, corner[2]]))
        else:
            # ignore planes and other stuff
            result = amin(np.array([result, 0.0]))

    return result

    
        