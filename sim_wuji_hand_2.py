from pathlib import Path

import mujoco
import numpy as np

from sim import JacobianCircleController, run_viewer

WUJI_OBSERVATION_NOISE = 1e-2
WUJI_DAMPING = 5e-3
WUJI_PULLBACK_GAIN = 0.8
WUJI_POSITION_GAIN = 10.0
WUJI_EXCITATION_VELOCITY_SCALE = 1.5e-2


def create_wuji_hand_sim():
    """Create the reset Wuji Hand model, data, and tuned circle controller."""
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).with_name("wuji_hand_2") / "scene_pen.xml")
    )
    # The shared controller expects one callback per simulation step, as in
    # sim.py. Wuji's XML defaults to RK4, which invokes it at intermediate
    # stages as well and corrupts the controller's decimation and estimates.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # only the thumb, index, and middle fingers participate.
    actuator_ids = np.arange(12)
    dof_ids = model.jnt_dofadr[model.actuator_trnid[actuator_ids, 0]]
    excitation_groups = (
        slice(0, 4), # thumb
        slice(4, 8), # index
        slice(8, 12), # middle
    )

    controller = JacobianCircleController(
        model,
        data,
        actuator_ids,
        dof_ids,
        excitation_groups,
        observation_noise=WUJI_OBSERVATION_NOISE,
        damping=WUJI_DAMPING,
        pullback_gain=WUJI_PULLBACK_GAIN,
        position_gain=WUJI_POSITION_GAIN,
        excitation_velocity_scale=WUJI_EXCITATION_VELOCITY_SCALE,
    )
    return model, data, controller


def main():
    model, data, controller = create_wuji_hand_sim()

    mujoco.set_mjcb_control(controller.control_cb)
    try:
        run_viewer(model, data, controller)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
