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

# Parameters for Circle trajectory
CIRCLE_RADIUS = 0.005
CIRCLE_ANGULAR_SPEED = 0.2

# Parameters for Jacobian estimator 
TASK_DIM = 3  # control xyz of pen tip
P_CONST = 0.1  # use constant value for P
OBS_NOISE = 1e-2

# Parameters for inverse Jacobian based controller
CONTROL_DECIMATION = 10  # The controller is run every CONTROL_DECIMATION simulation steps.
DAMPING = 0.005
PULLBACK_GAIN = 0.5
POSITION_GAIN = 10.

# initially excite the joints so an all-zero jacobian can learn from the joint and pen-tip motion before circle tracking starts.
EXCITATION_STEP_DURATION = 2.
EXCITATION_JNT_VELOCITY_SCALE = 2e-2
EXCITATION_NOISE_MEMORY = 0.95  # To make the random excitation somewhat smooth. 1 is random walk, 0 is white noise
RANDOM_SEED = 42

def _name(model, object_type, object_id):
    return mujoco.mj_id2name(model, object_type, object_id)


class JacobianCircleController:
    def __init__(
        self,
        model,
        data,
        actuator_ids,
        dof_ids,
        excitation_groups,
        *,
        observation_noise=OBS_NOISE,
        damping=DAMPING,
        pullback_gain=PULLBACK_GAIN,
        position_gain=POSITION_GAIN,
        excitation_velocity_scale=EXCITATION_JNT_VELOCITY_SCALE,
    ):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            actuator_ids: indices of actuators to control
            dof_ids: indices of the degrees of freedom corresponding to the actuators (differs from the actuator IDs for some models)
            excitation_groups: which groups of actuators to randomly shake together during initial excitation phase
            observation_noise: regularization used by the Jacobian estimator
            damping: regularization used by the damped pseudoinverse
            pullback_gain: strength of the null-space pull toward the initial grip
            position_gain: proportional gain for pen-tip position tracking
            excitation_velocity_scale: magnitude of the bootstrap joint excitation
        """
        self.model = model
        self.data = data
        self.actuator_ids = actuator_ids
        self.dof_ids = dof_ids
        self.excitation_groups = excitation_groups
        self.dt = model.opt.timestep * CONTROL_DECIMATION
        self.observation_noise = observation_noise
        self.damping = damping
        self.pullback_gain = pullback_gain
        self.position_gain = position_gain
        self.excitation_velocity_scale = excitation_velocity_scale

        actuator_names = [
            _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            for actuator_id in self.actuator_ids
        ]
        print("Started Jacobian controller which controls:", ", ".join(actuator_names))


        self.pen_tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pen_tip"
        )
        if self.pen_tip_id == -1:
            raise RuntimeError("Body 'pen_tip' was not found")

        self.rng = np.random.RandomState(RANDOM_SEED)
        self.J = np.empty((TASK_DIM, self.actuator_ids.size))
        self.prev_x = np.empty(TASK_DIM)  # measured pen-tip position
        self.init_ctrl = np.empty(model.nu)
        self.excitation_ctrl = np.empty(model.nu)
        self.circle_center = np.empty(3)
        self.start_time = 0.0
        self.decimation_counter = CONTROL_DECIMATION
        mujoco.mj_forward(self.model, self.data)
        self.reset()

    def reset(self):
        """Reset estimator state around the simulation's current grip."""
        self.J[:] = 0.
        self.prev_x[:] = self.data.xpos[self.pen_tip_id][:TASK_DIM]
        self.init_ctrl[:] = self.data.ctrl
        self.excitation_ctrl[:] = 0.0
        self.prev_excitation_stage = -1
        self.start_time = self.data.time
        self.decimation_counter = CONTROL_DECIMATION

        # the initial pen position should be the top of the circle that it draws
        self.circle_center[:] = self.data.xpos[self.pen_tip_id] + np.array([-CIRCLE_RADIUS, 0.0, 0.0])

    def circle_reference(self, time):
        """Return circle position and velocity at simulation time ``time``."""
        tracking_time = max(time - self.start_time - len(self.excitation_groups) * EXCITATION_STEP_DURATION, 0.0)
        phase = CIRCLE_ANGULAR_SPEED * tracking_time
        offset = CIRCLE_RADIUS * np.array(
            [np.cos(phase), np.sin(phase), 0.0]
        )
        velocity = CIRCLE_RADIUS * CIRCLE_ANGULAR_SPEED * np.array(
            [-np.sin(phase), np.cos(phase), 0.0]
        )
        return self.circle_center + offset, velocity

    def _update_jacobian(self, current_velocity, dq):
        """Apply the diagonal-covariance RLS update used by the ROS node."""
        denominator = P_CONST * (dq * dq) + self.observation_noise
        prediction_error = current_velocity - self.J @ dq
        numerator = prediction_error[:, None] * (P_CONST * dq)[None, :]
        self.J += numerator / denominator
        # print("Jacobian update:", self.J)

    def control_cb(self, model, data):
        # only run the controller every CONTROL_DECIMATION steps
        if self.decimation_counter < CONTROL_DECIMATION:
            self.decimation_counter += 1
            return
        self.decimation_counter = 1

        dq = data.qvel[self.dof_ids]
        current_x = data.xpos[self.pen_tip_id][:TASK_DIM]
        dx = (current_x - self.prev_x) / self.dt
        self.prev_x[:] = current_x
        
        # self._update_jacobian(dx, dq_cmd)
        self._update_jacobian(dx, dq)

        target_position, target_velocity = self.circle_reference(data.time)
        position_error = target_position[:TASK_DIM] - current_x
        commanded_velocity = (
            self.position_gain * position_error + target_velocity[:TASK_DIM]
        )

        # Damped pseudoinverse and null-space pullback, matching the ROS node.
        J_pinv = self.J.T @ np.linalg.inv(
            self.J @ self.J.T + self.damping * np.eye(TASK_DIM)
        )
        null_projector = np.eye(self.actuator_ids.size) - J_pinv @ self.J
        tracking_dq = J_pinv @ commanded_velocity
        pullback = self.init_ctrl[self.actuator_ids] - data.ctrl[self.actuator_ids]
        delta_q = (
            tracking_dq + self.pullback_gain * null_projector @ pullback
        ) * self.dt

        data.ctrl[self.actuator_ids] += delta_q
        data.ctrl[:] = np.clip(
            data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        excitation_stage = int((data.time - self.start_time) / EXCITATION_STEP_DURATION)
        if excitation_stage < len(self.excitation_groups):
            # Smooth random excitation: correlated noise avoids abrupt target jumps.
            group = self.excitation_groups[excitation_stage]
            if excitation_stage != self.prev_excitation_stage:
                print(f"Excitation stage {excitation_stage} of {len(self.excitation_groups)}: shaking joints {group}")
                self.prev_excitation_stage = excitation_stage
            self.excitation_ctrl *= EXCITATION_NOISE_MEMORY
            self.excitation_ctrl[group] += (
                self.rng.randn(group.stop - group.start)
                * self.excitation_velocity_scale
                * np.sqrt(1.0 - EXCITATION_NOISE_MEMORY**2)
            )
            data.ctrl[:] = self.init_ctrl + self.excitation_ctrl
        else:
            if self.prev_excitation_stage != len(self.excitation_groups):
                print("Excitation finished, starting trajectory following")
                self.prev_excitation_stage += 1




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
        sim_viewer.cam.lookat = [0.35, 0., -0.1]

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
    # load the model and reset the simulation
    shadow_hand_model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("shadow_hand") / "scene_pen.xml"))
    shadow_hand_data = mujoco.MjData(shadow_hand_model)
    mujoco.mj_resetDataKeyframe(shadow_hand_model, shadow_hand_data, 0)

    # define the joint groups that shold be excited together during the bootstrap phase
    shadow_hand_excitation_groups = (
        slice(0, 2),  # wrist
        slice(2, 7),  # thumb
        slice(7, 10),  # index
        slice(10, 12),  # middle
    )
    shadow_hand_actuator_ids = np.arange(13)

    def shadow_hand_actuator2dof_ids(model, actuator_ids):
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
    shadow_hand_dof_ids = shadow_hand_actuator2dof_ids(shadow_hand_model, shadow_hand_actuator_ids)

    controller = JacobianCircleController(
        shadow_hand_model, shadow_hand_data, shadow_hand_actuator_ids, shadow_hand_dof_ids, shadow_hand_excitation_groups
    )

    mujoco.set_mjcb_control(controller.control_cb)
    try:
        run_viewer(shadow_hand_model, shadow_hand_data, controller)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
