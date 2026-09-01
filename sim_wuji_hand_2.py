from pathlib import Path

import mujoco
import numpy as np

from sim import JacobianCircleController, run_viewer


def main():
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
        model, data, actuator_ids, dof_ids, excitation_groups
    )

    mujoco.set_mjcb_control(controller.control_cb)
    try:
        run_viewer(model, data, controller)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
