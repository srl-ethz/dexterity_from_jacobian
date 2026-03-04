import mujoco
from mujoco import viewer
import numpy as np
try:
    from qpsolvers import solve_qp
except ImportError:
    solve_qp = None
if solve_qp is None:
    print("qpsolvers not found: falling back to least-squares MPC. Install qpsolvers+osqp for QP MPC.")

"""
run the simulation using the estimated Jacobian based controller
"""

model_path = "shadow_hand/scene_cube.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# initial commanded hand pose that angles hand downwards and lightly closes the fingers
init_ctrl = [0.08, -0.3,
                0.2, 1.2, 0.2, 0.4, 0,
                -0.1, 0.6, 2,
                0.0, 0.4, 2,
                -0.1, 0.4, 2,
                0, -0.3, 0.5, 2,]
init_ctrl = np.array(init_ctrl)
data.ctrl[:] = init_ctrl

# params for sphere position task with all actuators
#actuators_enabled = np.arange(model.nu)  # use all actuators
# eps = 0.002  # how much to weigh the "going back to init pose" term

# params for sphere position task with no wrist
actuators_enabled = np.arange(2,model.nu)  # disable the first two actuators (they control the wrist)
eps = 0.001

actuator_num = len(actuators_enabled)

# linear MPC parameters
mpc_horizon = 24  # longer horizon; practical with the QP backend
mpc_ctrl_reg = 3e-4
mpc_smooth_reg = 3e-3
mpc_init_pose_reg = eps
mpc_max_joint_vel = 50.0  # [rad/s]
mpc_fingertip_weight = 0.2  # secondary objective weight (relative to object task)
mpc_fingertip_kp = 50.0  # proportional gain for fingertip target velocity
mpc_fingertip_max_speed = 0.4  # [m/s] cap to keep fingertip objective stable
grasp_open_out_amp = 0.08  # [m] outward opening amplitude (reduced)
grasp_contact_in_amp = 0.01  # [m] inward pressure for the two contact fingers

# Choose which task-space objective to track:
# "sphere" -> 3D position trajectory, "cube" -> 6D pose (position + orientation).
task_mode = "cube"
# Per-component task tracking weights used inside MPC.
# Cube ordering is [x, y, z, rot_x, rot_y, rot_z].
task_component_weights = np.ones(6) if task_mode == "cube" else np.ones(3)

# get the indices to access the robot's state
# get the names of the actuators
actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in actuators_enabled]
# get the indices of the corresponding joints
# convert the actuator names to the joint names (specific to the shadow hand)
actuated_joint_names = [actuator_name.replace("_A_", "_") for actuator_name in actuator_names]
actuated_joint_names = [jnt_name.replace("0", "1") for jnt_name in actuated_joint_names]
actuated_joint_ids = []
for joint_name in actuated_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id != -1, f"Joint {joint_name} not found"
    actuated_joint_ids.append(joint_id)
# start addr in 'qvel' for joint's data
actuated_dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in actuated_joint_ids]
# start addr in 'qpos' for joint's data
actuated_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in actuated_joint_ids]
print(f"{actuator_names=}\n{actuated_joint_names=}\n{actuated_joint_ids=}\n{actuated_dof_ids=}\n{actuated_qpos_ids=}")

# find out which actuators belong to each finger
finger_name_filters = ["_TH", "_FF", "_MF", "_RF", "_LF"]
actuator2finger = np.zeros(actuator_num) - 1  # -1 means not assigned to any finger
for i in range(actuator_num):
    for finger_id, finger_name_filter in enumerate(finger_name_filters):
        if finger_name_filter in actuator_names[i]:
            actuator2finger[i] = finger_id
            break
    if (actuator2finger[i] == -1):
        print(f"Actuator {actuator_names[i]} not assigned to any finger")
print(f"{actuator2finger=}")

# get the indices to access the object's state
object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
assert object_id != -1, "Object not found"
object_dof_ids = np.arange(model.body_dofadr[object_id], model.body_dofadr[object_id] + model.body_dofnum[object_id])
object_qpos_ids = np.arange(model.body_jntadr[object_id], model.body_jntadr[object_id] + 7)
print(f"{object_id=}\n{object_dof_ids=}\n{object_qpos_ids=}")

# get the indices to access the ghost object (just to show the target pose)
ghost_object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_ghost")
assert ghost_object_id != -1, "Ghost object not found"

mujoco.mj_forward(model, data)
body_init_pose = data.qpos[object_qpos_ids].copy()
print(f"{body_init_pose=}")
body_target_pos = np.zeros(3)

# Fingertip bodies used for grasp shaping (front/index/middle/ring/little + thumb).
fingertip_body_names = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"]
fingertip_body_ids = np.array(
    [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in fingertip_body_names], dtype=int
)
assert np.all(fingertip_body_ids >= 0), "At least one fingertip body was not found"
# Exactly these two fingertips are kept in contact-target mode.
always_contact_fingertip_names = ["rh_ffdistal", "rh_mfdistal"]
always_contact_tip_ids = {
    fingertip_body_names.index(name) for name in always_contact_fingertip_names if name in fingertip_body_names
}
assert len(always_contact_tip_ids) == 2, "Configure exactly two always-contact fingertips"

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return v / n

# Store the fingertip directions in the object's local frame at startup.
# These directions are later rotated with the object to define moving target points around the cube.
R_object_init = data.xmat[object_id].reshape(3, 3).copy()
cube_center_init = data.xpos[object_id].copy()
fingertip_dirs_local = []
for tip_id in fingertip_body_ids:
    tip_world = data.xpos[tip_id].copy()
    dir_world = _normalize(tip_world - cube_center_init)
    fingertip_dirs_local.append(R_object_init.T @ dir_world)
fingertip_dirs_local = np.array(fingertip_dirs_local)
fingertip_target_radius = 0.045

# the estimated control Jacobian; row count depends on task-space dimension
task_dim = 6 if task_mode == "cube" else 3
J = np.zeros((task_dim, actuator_num))
# covariance of the estimated J
p = np.ones(actuator_num) * 1e-1


def check_finger_contact():
    """
    check if each finger is in contact with the object
    """
    # if the body name contains any of these strings, it belongs to a finger
    finger_name_filters = ['rh_th', 'rh_ff', 'rh_mf', 'rh_rf', 'rh_lf']
    finger_contact_detected = np.zeros(5)
    for contact in data.contact:
        collision_body_ids = [model.geom_bodyid[geom] for geom in contact.geom]
        if object_id in collision_body_ids:
            # this contact is with the object; find out if it is in contact with a finger
            other_body_id = collision_body_ids[0] if collision_body_ids[1] == object_id else collision_body_ids[1]
            other_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
            for i, finger_name_filter in enumerate(finger_name_filters):
                if finger_name_filter in other_body_name:
                    finger_contact_detected[i] = 1
                    break
    # print(f"{finger_contact_detected=}")
    return finger_contact_detected


def compute_task_space_command():
    """
    compute the task space command that will bring the system closer to task space goal
    it is supposed to be the desired velocity in the task space
    """
    # the target should slowly draw a circle
    phase = data.time * 3
    body_target_pos[:] = body_init_pose[:3] + 0.02 * np.array([np.sin(phase), np.cos(1.4*phase), np.cos(1.3*phase)*0.2])
    body_target_pos[2] -= 0.02
    body_target_vel = np.array([0.02 * np.cos(phase), -0.02 * 1.4 * np.sin(1.4*phase), -0.02 * 1.3 * np.sin(1.3*phase)*0.2])
    # move the mocap object to the target position for visualization
    data.mocap_pos[:] = body_target_pos
    body_pos = data.xpos[object_id]
    
    task_space_vel = (body_target_pos - body_pos) * 8 + body_target_vel
    return task_space_vel

def compute_task_space_vel():
    """
    return the velocity of task
    """
    return data.qvel[object_dof_ids[:3]]

def compute_task_space_command_cube():
    """
    compute command to rotate the cube towards target orientation
    """
    # rotate target continuously around world z-axis
    angular_speed = -1  # [rad/s], increase for faster continuous spinning
    phase = data.time * angular_speed
    quat_target = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat_target, np.array([0.0, 0.0, 1.0]), phase)

    quat_current = data.xquat[object_id]
    pos_current = data.xpos[object_id]
    pos_target = body_init_pose[:3] + np.array([0.0, 0.0, 0.04])
    rot_diff = np.zeros(3)
    mujoco.mju_subQuat(rot_diff, quat_target, quat_current)
    pos_diff = pos_target - pos_current
    data.mocap_quat[:] = quat_target
    data.mocap_pos[:] = pos_target
    
    return np.concatenate((pos_diff * 8, rot_diff * 1))


def compute_task_space_vel_cube():
    """
    compute rotational velocity of the cube
    """
    return data.qvel[object_dof_ids]


def compute_fingertip_grasp_task():
    """
    Build fingertip Jacobian and desired fingertip velocities for grasp shaping around the cube.
    """
    R_object = data.xmat[object_id].reshape(3, 3)
    cube_center = data.xpos[object_id]

    J_tip = np.zeros((3 * len(fingertip_body_ids), actuator_num))
    v_tip_des = np.zeros(3 * len(fingertip_body_ids))

    for i, tip_id in enumerate(fingertip_body_ids):
        # Keep exactly two fingertips pulled inward (contact), push all others outward.
        breath_i = -grasp_contact_in_amp if i in always_contact_tip_ids else grasp_open_out_amp
        target_radius_i = max(0.01, fingertip_target_radius + breath_i)
        target_world = cube_center + R_object @ (fingertip_dirs_local[i] * target_radius_i)
        tip_world = data.xpos[tip_id]
        tip_vel_des = mpc_fingertip_kp * (target_world - tip_world)
        tip_vel_des = np.clip(tip_vel_des, -mpc_fingertip_max_speed, mpc_fingertip_max_speed)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, int(tip_id))
        J_tip_i = jacp[:, actuated_dof_ids]

        row_slice = slice(3 * i, 3 * i + 3)
        J_tip[row_slice, :] = J_tip_i
        v_tip_des[row_slice] = tip_vel_des

    return J_tip, v_tip_des


def compute_mpc_delta_q(J_eff, task_space_vel_desired, J_tip, v_tip_des, ctrl_enabled, ctrl_lower, ctrl_upper, dt):
    """
    finite-horizon linear MPC in velocity form.
    Uses qpsolvers (OSQP) when installed; otherwise falls back to least-squares.
    """
    n = actuator_num
    N = mpc_horizon

    # decision variable is stacked joint velocity commands: [dq_0, ..., dq_{N-1}]
    # Apply task-component weights so we can penalize selected axes only.
    W_task = np.diag(np.sqrt(task_component_weights))
    A_track_obj = np.kron(np.eye(N), W_task @ J_eff)
    b_track_obj = np.tile(W_task @ task_space_vel_desired, N)

    A_track_tip = np.sqrt(mpc_fingertip_weight) * np.kron(np.eye(N), J_tip)
    b_track_tip = np.sqrt(mpc_fingertip_weight) * np.tile(v_tip_des, N)

    A_ctrl = np.sqrt(mpc_ctrl_reg) * np.eye(n * N)
    b_ctrl = np.zeros(n * N)

    D = np.eye(N) - np.eye(N, k=-1)
    A_smooth = np.sqrt(mpc_smooth_reg) * np.kron(D, np.eye(n))
    b_smooth = np.zeros(n * N)

    # predicted ctrl_k = ctrl_0 + dt * sum_{j=0..k} dq_j
    L = np.tril(np.ones((N, N)))
    A_init = np.sqrt(mpc_init_pose_reg) * dt * np.kron(L, np.eye(n))
    b_init = np.sqrt(mpc_init_pose_reg) * np.tile(init_ctrl[actuators_enabled] - ctrl_enabled, N)

    A_ls = np.vstack((A_track_obj, A_track_tip, A_ctrl, A_smooth, A_init))
    b_ls = np.concatenate((b_track_obj, b_track_tip, b_ctrl, b_smooth, b_init))

    # Build linear inequality constraints:
    # 1) box constraints on dq at each horizon step
    lb = -np.ones(n * N) * mpc_max_joint_vel
    ub = np.ones(n * N) * mpc_max_joint_vel
    # 2) predicted commanded positions must stay within actuator limits
    L = np.tril(np.ones((N, N)))
    S = dt * np.kron(L, np.eye(n))
    ctrl_upper_stacked = np.tile(ctrl_upper - ctrl_enabled, N)
    ctrl_lower_stacked = np.tile(ctrl_lower - ctrl_enabled, N)
    G = np.vstack((S, -S))
    h = np.concatenate((ctrl_upper_stacked, -ctrl_lower_stacked))

    if solve_qp is not None:
        # min 0.5 x^T P x + q^T x, with Gx <= h and lb <= x <= ub
        P = 2.0 * (A_ls.T @ A_ls)
        q = -2.0 * (A_ls.T @ b_ls)
        # tiny diagonal regularization improves numerical conditioning for long horizons
        P += np.eye(n * N) * 1e-9
        dq_seq = solve_qp(P, q, G=G, h=h, lb=lb, ub=ub, solver="osqp")
        if dq_seq is None:
            # Fallback if the QP solver fails to converge
            dq_seq, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
    else:
        dq_seq, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)

    dq_0 = dq_seq[:n]
    return dq_0 * dt

def control_cb(model, data):
    """
    callback function called on every step, and is used to set the control command
    """
    # don't do anything for the first moments (until ball falls)
    finger_contacts = check_finger_contact()
    # compute for which actuators affect the object currently (the finger that the actuator belongs to is in contact with the object)
    actuator_affecting_object_ids = []
    for i in range(actuator_num):
        # Actuators not mapped to a finger are excluded from contact-based control.
        if actuator2finger[i] < 0:
            continue
        if finger_contacts[int(actuator2finger[i])]:
            actuator_affecting_object_ids.append(i)

    # first update the estimation of the Jacobian, just for the actuators whose fingers are in contact with the object
    global J, p
    q = data.qpos[actuated_qpos_ids]
    dq = data.qvel[actuated_dof_ids]
    # Select measured task velocity according to the active task mode.
    if task_mode == "cube":
        u = compute_task_space_vel_cube()
    else:
        u = compute_task_space_vel()
    r = 1e-3  # observation noise variance

    # just update the part of the Jacobian that affects the object
    J_slice = J[:, actuator_affecting_object_ids]
    p_slice = p[actuator_affecting_object_ids]
    q_slice = q[actuator_affecting_object_ids]
    dq_slice = dq[actuator_affecting_object_ids]

    numerator = (u - J_slice @ dq_slice).reshape((-1, 1)) @ (p_slice * dq_slice).reshape((1, -1))
    denominator = p_slice.T @ (dq_slice * dq_slice) + r
    J_slice[:] += numerator / denominator
    p_slice[:] *= 1 - p_slice * dq_slice * dq_slice / denominator

    J[:, actuator_affecting_object_ids] = J_slice
    p[actuator_affecting_object_ids] = p_slice

    # Select desired task-space command according to the active task mode.
    if task_mode == "cube":
        task_space_vel_desired = compute_task_space_command_cube()
    else:
        task_space_vel_desired = compute_task_space_command()
    # build full Jacobian over enabled actuators (columns for non-contact fingers stay zero)
    J_eff = np.zeros((task_dim, actuator_num))
    J_eff[:, actuator_affecting_object_ids] = J_slice
    J_tip, v_tip_des = compute_fingertip_grasp_task()

    # compute updated commanded joint position with a finite-horizon linear MPC
    dt = model.opt.timestep
    ctrl_enabled = data.ctrl[actuators_enabled]
    ctrl_lower = model.actuator_ctrlrange[actuators_enabled, 0]
    ctrl_upper = model.actuator_ctrlrange[actuators_enabled, 1]
    delta_q = compute_mpc_delta_q(
        J_eff,
        task_space_vel_desired,
        J_tip,
        v_tip_des,
        ctrl_enabled,
        ctrl_lower,
        ctrl_upper,
        dt,
    )
    # don't move too fast
    delta_q = np.clip(delta_q, -mpc_max_joint_vel * dt, mpc_max_joint_vel * dt)
    data.ctrl[actuators_enabled] += delta_q
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

mujoco.set_mjcb_control(control_cb)


viewer.launch(model, data)
