"""Minimal MuJoCo demo of online Jacobian estimation and circle tracking.

The estimator and controller mirror ``dex_controller_node.py``: a diagonal
RLS covariance estimates a two-dimensional control Jacobian, and a damped
pseudoinverse plus null-space pullback produces joint-position increments.
Only the thumb, index, and middle finger actuators participate.
"""

from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer


MODEL_PATH = Path(__file__).with_name("shadow_hand") / "scene_pen_realer.xml"
RESET_KEYFRAME = 0
CONTROL_DECIMATION = 20

# Circle reference and task-space controller.
CIRCLE_RADIUS = 0.005
CIRCLE_ANGULAR_SPEED = 0.2
POSITION_GAIN = 3.0

# Jacobian estimator and joint-space controller.
TASK_DIM = 2
P_INIT = 0.1
OBS_NOISE = 1e-3
CONFIDENCE_FLOOR = 0.01
CONFIDENCE_FORGETTING_FACTOR = 0.999
MOTION_THRESHOLD = 1e-4
DAMPING = 0.003
PULLBACK_GAIN = 0.05
MAX_JOINT_VEL = 0.8
COMMAND_EMA_WEIGHT = 0.8

# A small random Jacobian lets the full controller move immediately. Without
# the ROS controller's initial excitation waypoints, an all-zero Jacobian
# would produce an all-zero command and could never bootstrap the estimator.
J_INIT_SCALE = 1.2e-3
RANDOM_SEED = 42


def _name(model, object_type, object_id):
    return mujoco.mj_id2name(model, object_type, object_id)


def _finger_actuator_ids(model):
    """Return the fixed thumb/index/middle actuator set."""
    finger_tags = ("_TH", "_FF", "_MF")
    ids = [
        actuator_id
        for actuator_id in range(model.nu)
        if any(
            tag in _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for tag in finger_tags
        )
    ]
    if not ids:
        raise RuntimeError("No thumb, index, or middle finger actuators found")
    return np.asarray(ids, dtype=int)


def _actuator_dof_ids(model, actuator_ids):
    """Map Shadow Hand actuator names to the joint velocities they control."""
    dof_ids = []
    for actuator_id in actuator_ids:
        actuator_name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = actuator_name.replace("_A_", "_").replace("J0", "J1")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise RuntimeError(
                f"Could not map actuator {actuator_name!r} to joint {joint_name!r}"
            )
        dof_ids.append(model.jnt_dofadr[joint_id])
    return np.asarray(dof_ids, dtype=int)


class JacobianCircleController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.actuator_ids = _finger_actuator_ids(model)
        self.dof_ids = _actuator_dof_ids(model, self.actuator_ids)
        self.actuator_count = len(self.actuator_ids)
        self.dt = model.opt.timestep * CONTROL_DECIMATION

        self.pen_tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pen_tip"
        )
        if self.pen_tip_id == -1:
            raise RuntimeError("Body 'pen_tip' was not found")

        self.rng = np.random.RandomState(RANDOM_SEED)
        self.J = np.empty((TASK_DIM, self.actuator_count))
        self.p = np.empty(self.actuator_count)
        self.prev_delta_q_cmd = np.empty(self.actuator_count)
        self.prev_q = np.empty(self.actuator_count)  # measured joint angles
        self.prev_x = np.empty(TASK_DIM)  # measured pen-tip position
        self.init_ctrl = np.empty(model.nu)
        self.circle_center = np.empty(3)
        self.start_time = 0.0
        self.decimation_counter = CONTROL_DECIMATION
        mujoco.mj_forward(self.model, self.data)
        self.reset()

    def reset(self):
        """Reset estimator state around the simulation's current grip."""
        self.J[:] = self.rng.randn(TASK_DIM, self.actuator_count) * J_INIT_SCALE
        self.p[:] = P_INIT
        self.prev_delta_q_cmd[:] = 0.0
        self.prev_q[:] = self.data.qpos[self.dof_ids]
        self.prev_x[:] = self.data.xpos[self.pen_tip_id][:TASK_DIM]
        self.init_ctrl[:] = self.data.ctrl
        self.start_time = self.data.time
        self.decimation_counter = CONTROL_DECIMATION

        # Start at the bottom of the circle, exactly at the current pen tip.
        initial_offset = np.array([0.0, -CIRCLE_RADIUS, 0.0])
        self.circle_center[:] = self.data.xpos[self.pen_tip_id] - initial_offset

    def circle_reference(self, time):
        """Return circle position and velocity at simulation time ``time``."""
        phase = CIRCLE_ANGULAR_SPEED * (time - self.start_time) - np.pi / 2.0
        offset = CIRCLE_RADIUS * np.array(
            [np.cos(phase), np.sin(phase), 0.0]
        )
        velocity = CIRCLE_RADIUS * CIRCLE_ANGULAR_SPEED * np.array(
            [-np.sin(phase), np.cos(phase), 0.0]
        )
        return self.circle_center + offset, velocity

    def _update_jacobian(self, current_velocity, dq):
        """Apply the diagonal-covariance RLS update used by the ROS node."""
        active = np.abs(dq) > MOTION_THRESHOLD
        self.p[active] = np.minimum(
            self.p[active] / CONFIDENCE_FORGETTING_FACTOR, P_INIT
        )

        denominator = self.p @ (dq * dq) + OBS_NOISE
        prediction_error = current_velocity - self.J @ dq
        numerator = prediction_error[:, None] * (self.p * dq)[None, :]
        self.J += numerator / denominator
        self.p[:] = np.maximum(
            self.p * (1.0 - self.p * dq * dq / denominator),
            CONFIDENCE_FLOOR,
        )

    def control_cb(self, model, data):
        # only run the controller every CONTROL_DECIMATION steps
        if self.decimation_counter < CONTROL_DECIMATION:
            self.decimation_counter += 1
            return
        self.decimation_counter = 1

        dq_cmd = self.prev_delta_q_cmd / self.dt
        # compute velocity based on the difference from the previous control step
        current_q = data.qpos[self.dof_ids]
        dq = (current_q - self.prev_q) / self.dt
        self.prev_q[:] = current_q
        current_x = data.xpos[self.pen_tip_id][:TASK_DIM]
        dx = (current_x - self.prev_x) / self.dt
        self.prev_x[:] = current_x
        if np.any(np.abs(dq) > MOTION_THRESHOLD):
            # self._update_jacobian(dx, dq_cmd)
            self._update_jacobian(dx, dq)

        target_position, target_velocity = self.circle_reference(data.time)
        position_error = target_position[:TASK_DIM] - current_x
        # TODO: add D and I terms once it works on some level
        commanded_velocity = POSITION_GAIN * position_error + target_velocity[
            :TASK_DIM
        ]

        # Damped pseudoinverse and null-space pullback, matching the ROS node.
        J_pinv = self.J.T @ np.linalg.inv(
            self.J @ self.J.T + DAMPING * np.eye(TASK_DIM)
        )
        null_projector = np.eye(self.actuator_count) - J_pinv @ self.J
        tracking_dq = J_pinv @ commanded_velocity
        pullback = self.init_ctrl[self.actuator_ids] - data.ctrl[self.actuator_ids]
        delta_q = (
            tracking_dq + PULLBACK_GAIN * null_projector @ pullback
        ) * self.dt

        delta_q = np.clip(delta_q, -MAX_JOINT_VEL * self.dt, MAX_JOINT_VEL * self.dt)
        delta_q = (
            COMMAND_EMA_WEIGHT * delta_q
            + (1.0 - COMMAND_EMA_WEIGHT) * self.prev_delta_q_cmd
        )

        data.ctrl[self.actuator_ids] += delta_q
        data.ctrl[:] = np.clip(
            data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        self.prev_delta_q_cmd[:] = delta_q


def _draw_marker(scene, geom_id, position, color, radius):
    mujoco.mjv_initGeom(
        scene.geoms[geom_id],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        position,
        np.eye(3).ravel(),
        color,
    )


def run_viewer(model, data, controller):
    """Run the simulation and draw the circle plus its current target."""
    path_point_count = 48

    with viewer.launch_passive(model, data) as sim_viewer:
        sim_viewer.cam.distance = 0.4
        sim_viewer.cam.azimuth = -150
        sim_viewer.cam.elevation = -50
        sim_viewer.cam.lookat = [0.2, -0.2, 0]

        scene = sim_viewer.user_scn
        geom_start = scene.ngeom
        scene.ngeom += path_point_count + 1
        previous_time = data.time

        while sim_viewer.is_running():
            if data.time < previous_time:
                controller.reset()
            previous_time = data.time

            for index, phase in enumerate(
                np.linspace(0.0, 2.0 * np.pi, path_point_count, endpoint=False)
            ):
                position = controller.circle_center + CIRCLE_RADIUS * np.array(
                    [np.cos(phase), np.sin(phase), 0.0]
                )
                _draw_marker(
                    scene,
                    geom_start + index,
                    position,
                    np.array([0.0, 0.3, 1.0, 0.7]),
                    0.00025,
                )

            target_position, _ = controller.circle_reference(data.time)
            _draw_marker(
                scene,
                geom_start + path_point_count,
                target_position,
                np.array([1.0, 0.0, 0.0, 1.0]),
                0.0005,
            )

            mujoco.mj_step(model, data)
            sim_viewer.sync()


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, RESET_KEYFRAME)

    controller = JacobianCircleController(model, data)
    actuator_names = [
        _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in controller.actuator_ids
    ]
    print("Controlled actuators:", ", ".join(actuator_names))

    mujoco.set_mjcb_control(controller.control_cb)
    try:
        run_viewer(model, data, controller)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
