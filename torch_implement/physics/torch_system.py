"""A torch environment system"""

import numpy as np
import math
from typing import Optional, Sequence, Tuple

from physics import torch_config_pb2
from physics.torch_base import validate_config
from physics.torch_base import vec_to_arr
from physics.torch_base import *
from physics import bodies
from physics import colliders
from physics import joints
from physics import spring_joints
from physics import actuators
from physics import forces
from physics import integrators

import torch_math.array_ops as arrop



class torchSystem:
    """A torch environment system"""
    
    
    def __init__(self, 
                 config: torch_config_pb2.Config,
                 resource_paths: Optional[Sequence[str]] = None) -> None:

        config = validate_config(config, resource_paths=resource_paths)

        self.config = config
        self.num_actuators = len(config.actuators)
        self.num_joint_dof = np.sum(len(j.angle_limit) for j in config.joints)   # Note: TODO:  here is still under numpy
        self.num_bodies = len(config.bodies)
        self.body = bodies.Body(config)
        self.colliders = colliders.get(config, self.body)
        self.num_joints = len(config.joints)        
        self.joints = joints.get(config, self.body) + spring_joints.get(
        config, self.body)
        self.actuators = actuators.get(config, self.joints)
        self.forces = forces.get(config, self.body)
        self.num_forces_dof = arrop.sum(f.act_index.shape[-1] for f in self.forces)
        self.integrator = integrators.Euler(config)
    

    def default_angle(self, default_index: int = 0) -> np.ndarray:
        """Returns the default joint angles for the system"""
        if not self.config.joints:
            return np.array([])
        dofs = {}
        for j in self.config.joints:
            dofs[j.name] = sum([l.min != 0 or l.max != 0 for l in j.angle_limit])
        angles = {}
        
        # check overrides in config defaults
        if default_index < len(self.config.defaults):
            defaults = self.config.defaults[default_index]
            for ja in defaults.angles:
                angles[ja.name] = vec_to_arr(ja.angle)[:dofs[ja.name]] * np.pi / 180

        # set remaining joint angles set from angle limits, and add jitter
        for joint in self.config.joints:
            if joint.name not in angles:
                dof = dofs[joint.name]
                angles[joint.name] = np.array([
                    (l.min + l.max) * np.pi / 360 for l in joint.angle_limit
                ][:dof])
        
        return arrop.concatenate([angles[j.name] for j in self.config.joints])



    def default_qp(self,
                    default_index: int = 0,
                    joint_angle: Optional[np.ndarray] = None,
                    joint_velocity: Optional[np.ndarray] = None) -> QP:
        """Returns a default state for the system."""
        qp = QP.zero(shape=(self.num_bodies,))

        # set any default qps from the config
        default = None
        if default_index < len(self.config.defaults):
            default = self.config.defaults[default_index]
        for dqp in default.qps:
            body_i = self.body.index[dqp.name]
            pos = np.index_update(qp.pos, body_i, vec_to_arr(dqp.pos))
            rot = np.index_update(qp.rot, body_i,
                                math.euler_to_quat(vec_to_arr(dqp.rot)))
            vel = np.index_update(qp.vel, body_i, vec_to_arr(dqp.vel))
            ang = np.index_update(qp.ang, body_i, vec_to_arr(dqp.ang))
            qp = qp.replace(pos=pos, rot=rot, vel=vel, ang=ang)

        # build joints and joint -> array lookup, and order by depth
        if joint_angle is None:
            joint_angle = self.default_angle(default_index)
        if joint_velocity is None:
            joint_velocity = np.zeros_like(joint_angle)
        joint_idxs = []
        for j in self.config.joints:
            beg = joint_idxs[-1][1][1] if joint_idxs else 0
            dof = arrop.sum([l.min != 0 or l.max != 0 for l in j.angle_limit])
            joint_idxs.append((j, (beg, beg + dof)))
        lineage = {j.child: j.parent for j in self.config.joints}
        depth = {}
        for child, parent in lineage.items():
            depth[child] = 1
            while parent in lineage:
                parent = lineage[parent]
                depth[child] += 1
        joint_idxs = sorted(joint_idxs, key=lambda x: depth.get(x[0].parent, 0))
        joint = [j for j, _ in joint_idxs]

        if joint:
            # convert joint_angle and joint_vel to 3dof
            takes = []
            beg = 0
            for j, (beg, end) in joint_idxs:
                arr = list(range(beg, end))
                arr.extend([self.num_joint_dof] * (3 - len(arr)))
                takes.extend(arr)
            takes = np.array(takes, dtype=int)

            def to_3dof(a):
                a = np.concatenate([a, np.array([0.0])])
                a = np.take(a, takes)
                a = np.reshape(a, (self.num_joints, 3))
                return a

            joint_angle = to_3dof(joint_angle)
            joint_velocity = to_3dof(joint_velocity)

            # build local rot and ang per joint
            joint_rot = np.array(
                [math.euler_to_quat(vec_to_arr(j.rotation)) for j in joint])
            joint_ref = np.array(
                [math.euler_to_quat(vec_to_arr(j.reference_rotation)) for j in joint])

            def local_rot_ang(_, x):
                angles, vels, rot, ref = x
                axes = np.vmap(math.rotate, [True, False])(np.eye(3), rot)
                ang = np.dot(axes.T, vels).T
                rot = ref
                for axis, angle in zip(axes, angles):
                    # these are euler intrinsic rotations, so the axes are rotated too:
                    axis = math.rotate(axis, rot)
                    next_rot = math.quat_rot_axis(axis, angle)
                    rot = math.quat_mul(next_rot, rot)
                return (), (rot, ang)

        xs = (joint_angle, joint_velocity, joint_rot, joint_ref)
        _, (local_rot, local_ang) = np.scan(local_rot_ang, (), xs, len(joint))

        # update qp in depth order
        joint_body = np.array([
            (self.body.index[j.parent], self.body.index[j.child]) for j in joint
        ])
        joint_off = np.array([(vec_to_arr(j.parent_offset),
                                vec_to_arr(j.child_offset)) for j in joint])

        def set_qp(carry, x):
            qp, = carry
            (body_p, body_c), (off_p, off_c), local_rot, local_ang = x
            world_rot = math.quat_mul(qp.rot[body_p], local_rot)
            local_pos = off_p - math.rotate(off_c, local_rot)
            world_pos = qp.pos[body_p] + math.rotate(local_pos, qp.rot[body_p])
            world_ang = math.rotate(local_ang, qp.rot[body_p])
            pos = arrop.index_update(qp.pos, body_c, world_pos)
            rot = arrop.index_update(qp.rot, body_c, world_rot)
            ang = arrop.index_update(qp.ang, body_c, world_ang)
            qp = qp.replace(pos=pos, rot=rot, ang=ang)
            return (qp,), ()

        xs = (joint_body, joint_off, local_rot, local_ang)
        
        # TODO: define the scan function
        (qp,), () = arrop.scan(set_qp, (qp,), xs, len(joint))

        # any trees that have no body qp overrides in the config are moved above
        # the xy plane.  this convenience operation may be removed in the future.
        fixed = {j.child for j in joint}
        if default:
            fixed |= {qp.name for qp in default.qps}
        root_idx = {
            b.name: [i]
            for i, b in enumerate(self.config.bodies)
            if b.name not in fixed
        }
        for j in joint:
            parent = j.parent
            while parent in lineage:
                parent = lineage[parent]
            if parent in root_idx:
                root_idx[parent].append(self.body.index[j.child])

        for children in root_idx.values():
            zs = np.array([
                bodies.min_z(np.take(qp, c), self.config.bodies[c]) for c in children
            ])
            min_z = np.amin(zs)
            children = np.array(children)
            pos = np.take(qp.pos, children) - min_z * np.array([0., 0., 1.])
            pos = arrop.index_update(qp.pos, children, pos)
            qp = qp.replace(pos=pos)

        return qp


    def step(self, qp: QP, act: np.ndarray) -> Tuple[QP, Info]:
        """Generic step function.  Overridden with appropriate step at init."""
        step_funs = {'pbd': self._pbd_step, 'legacy_spring': self._spring_step}
        return step_funs[self.config.dynamics_mode](qp, act)

    def info(self, qp: QP) -> Info:
        """Return info about a system state."""
        info_funs = {'pbd': self._pbd_info, 'legacy_spring': self._spring_info}
        return info_funs[self.config.dynamics_mode](qp)

    

    def _pbd_step(self, qp: QP, act: np.ndarray) -> Tuple[QP, Info]:
        """Position based dynamics stepping scheme."""

        # Just like XPBD except performs two physics substeps per collision update.

        def substep(carry, _):
            qp, info = carry

            # first substep without collisions
            qprev = qp

            # apply acceleration updates for actuators, and forces
            zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))
            zero_q = Q(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 4)))
            dp_a = sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = sum([f.apply(qp, act) for f in self.forces], zero)
            dp_j = sum([j.damp(qp) for j in self.joints], zero)
            qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

            # apply kinetic step
            qp = self.integrator.kinetic(qp)

            # apply joint position update
            dq_j = sum([j.apply(qp) for j in self.joints], zero_q)
            qp = self.integrator.update(qp, pos_q=dq_j)

            # apply pbd velocity projection
            qp = self.integrator.velocity_projection(qp, qprev)

            qprev = qp
            # second substep with collisions

            # apply acceleration updates for actuators, and forces
            dp_a = arrop.sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = arrop.sum([f.apply(qp, act) for f in self.forces], zero)
            dp_j = arrop.sum([j.damp(qp) for j in self.joints], zero)
            qp = self.integrator.update(qp, acc_p=dp_a + dp_f + dp_j)

            # apply kinetic step
            qp = self.integrator.kinetic(qp)

            # apply joint position update
            dq_j = arrop.sum([j.apply(qp) for j in self.joints], zero_q)
            qp = self.integrator.update(qp, pos_q=dq_j)

            collide_data = [c.position_apply(qp, qprev) for c in self.colliders]
            dq_c = arrop.sum([c[0] for c in collide_data], zero_q)
            dlambda = [c[1] for c in collide_data]
            contact = [c[2] for c in collide_data]
            qp = self.integrator.update(qp, pos_q=dq_c)

            # apply pbd velocity updates
            qp_right_before = qp
            qp = self.integrator.velocity_projection(qp, qprev)
            # apply collision velocity updates
            dp_c = arrop.sum([
                c.velocity_apply(qp, dlambda[i], qp_right_before, contact[i])
                for i, c in enumerate(self.colliders)
            ], zero)
            qp = self.integrator.update(qp, vel_p=dp_c)

            info = Info(info.contact + dp_c, info.joint, info.actuator + dp_a)
            return (qp, info), ()

        # update collider statistics for culling
        for c in self.colliders:
            c.cull.update(qp)

        zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))
        info = Info(contact=zero, joint=zero, actuator=zero)

        (qp, info), _ = arrop.scan(substep, (qp, info), (), self.config.substeps // 2)
        return qp, info


    def _pbd_info(self, qp: QP) -> Info:
        """Return info about a system state."""
        zero_q = Q(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 4)))
        zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))
    
        # TODO: sort out a better way to get first-step collider data
        dq_c = arrop.sum([c.apply(qp) for c in self.colliders], zero)
        dq_j = arrop.sum([j.apply(qp) for j in self.joints], zero_q)
        info = Info(dq_c, dq_j, zero)
        return info


    def _spring_step(self, qp: QP, act: np.ndarray) -> Tuple[QP, Info]:
        """Spring-based dynamics stepping scheme."""

        # Resolves actuator forces, joints, and forces at acceleration level, and
        # resolves collisions at velocity level with baumgarte stabilization.

        def substep(carry, _):
            qp, info = carry

            # apply kinetic step
            qp = self.integrator.kinetic(qp)

            # apply acceleration level updates for joints, actuators, and forces
            zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))
            dp_j = arrop.sum([j.apply(qp) for j in self.joints], zero)
            dp_a = arrop.sum([a.apply(qp, act) for a in self.actuators], zero)
            dp_f = arrop.sum([f.apply(qp, act) for f in self.forces], zero)
            qp = self.integrator.update(qp, acc_p=dp_j + dp_a + dp_f)

            # apply velocity level updates for collisions
            dp_c = arrop.sum([c.apply(qp) for c in self.colliders], zero)
            qp = self.integrator.update(qp, vel_p=dp_c)

            info = Info(info.contact + dp_c, info.joint + dp_j, info.actuator + dp_a)
            return (qp, info), ()

            # update collider statistics for culling
            for c in self.colliders:
                c.cull.update(qp)

        zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))
        info = Info(contact=zero, joint=zero, actuator=zero)

        (qp, info), _ = np.scan(substep, (qp, info), (), self.config.substeps)
        return qp, info


    def _spring_info(self, qp: QP) -> Info:
        """Return info about a system state."""
        zero = P(np.zeros((self.num_bodies, 3)), np.zeros((self.num_bodies, 3)))

        dp_c = arrop.sum([c.apply(qp) for c in self.colliders], zero)
        dp_j = arrop.sum([j.apply(qp) for j in self.joints], zero)
        info = Info(dp_c, dp_j, zero)
        return info