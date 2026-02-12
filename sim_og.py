import mujoco
from mujoco import viewer
import numpy as np

"""
run the simulation using the estimated Jacobian based controller
"""

model_path = "shadow_hand/scene_pen.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset the state to keyframe 0

# copied from the XML
init_ctrl = [0, 0,
             0.24, 1., 0, 0.5, 0.3,
             0, 0.4, 2.2,
             0, 1, 2.,
             0, 1., 3.14,
             0, 0, 1., 3.14]
init_ctrl = np.array(init_ctrl)
data.ctrl[:] = init_ctrl

# params for sphere position task with all actuators
actuators_enabled = np.arange(model.nu)  # use all actuators
eps = 0.002  # how much to weigh the "going back to init pose" term

# params for sphere position task with no wrist
actuators_enabled = np.arange(2, model.nu)  # disable the first two actuators (they control the wrist)
eps = 0.003

actuator_num = len(actuators_enabled)

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

pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
assert pen_id != -1, "Pen not found"

# get the indices to access the ghost object (just to show the target pose)
# ghost_object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_ghost")
# assert ghost_object_id != -1, "Ghost object not found"

mujoco.mj_forward(model, data)
body_init_pose = data.qpos[object_qpos_ids].copy()
print(f"{body_init_pose=}")
body_target_pos = np.zeros(3)

# the estimated control Jacobian
J = np.zeros((3, actuator_num))
# covariance of the estimated J
p = np.ones(actuator_num) * 1e-1

def path(t):
    """
    when given a parameter t, returns the point on the path at t and the time derivative (i.e. velocity at that point)
    """
    a = 0.005
    offset = np.array([0.09, -0.35, -0.068])
    x = a * (np.cos(t) / (1 + np.sin(t)**2))
    y = a * (np.sin(t) * np.cos(t) / (1 + np.sin(t)**2))
    dx = a*(np.sin(t)**2 - 3)*np.sin(t)/(np.sin(t)**2 + 1)**2
    dy = a*(1 - 3*np.sin(t)**2)/(np.sin(t)**2 + 1)**2

    x = a * np.cos(t)
    y = a * np.sin(t)
    dx = -a * np.sin(t)
    dy = a * np.cos(t)

    return np.array([x, y, 0]) + offset, np.array([dx, dy, 0])

def check_finger_contact():
    """
    check if each finger is in contact with the object
    """
    # if the body name contains any of these strings, it belongs to a finger
    finger_name_filters = ['rh_th', 'rh_ff', 'rh_mf', 'rh_rf', 'rh_lf']
    finger_contact_detected = np.zeros(5)
    for contact in data.contact:
        collision_body_ids = [model.geom_bodyid[geom] for geom in contact.geom]
        if pen_id in collision_body_ids:
            # this contact is with the object; find out if it is in contact with a finger
            other_body_id = collision_body_ids[0] if collision_body_ids[1] == pen_id else collision_body_ids[1]
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
    # move the mocap object to the target position for visualization
    data.mocap_pos[:] = body_target_pos
    body_pos = data.xpos[object_id]
    
    task_space_vel = (body_target_pos - body_pos) * 8
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
    quat_current = data.xquat[object_id]
    quat_target = data.xquat[ghost_object_id]
    pos_current = data.xpos[object_id]
    pos_target = data.xpos[ghost_object_id]
    rot_diff = np.zeros(3)
    mujoco.mju_subQuat(rot_diff, quat_target, quat_current)
    pos_diff = pos_target - pos_current
    data.mocap_quat[:] = quat_target
    data.mocap_pos[:] = body_init_pose[:3] + np.array([0., 0., -0.04])
    if np.linalg.norm(rot_diff) < 0.1:
        print("Target orientation reached")
        # generate new target orientation
        quat_target = np.random.rand(4)
        quat_target /= np.linalg.norm(quat_target)
        data.mocap_quat[:] = quat_target
    
    return np.concatenate((pos_diff * 8, rot_diff * 1))


def compute_task_space_vel_cube():
    """
    compute rotational velocity of the cube
    """
    return data.qvel[object_dof_ids]

def compute_task_space_command_pen():
    t = data.time
    target_pos, target_vel = path(t)
    body_pos = data.xpos[object_id]
    task_space_vel = (target_pos - body_pos) * 8 + target_vel
    return task_space_vel

def compute_task_space_vel_pen():
    return data.sensordata

def control_cb(model, data):
    """
    callback function called on every step, and is used to set the control command
    """
    # don't do anything for the first moments (until ball falls)
    finger_contacts = check_finger_contact()
    # compute for which actuators affect the object currently (the finger that the actuator belongs to is in contact with the object)
    actuator_affecting_object_ids = []
    for i in range(actuator_num):
        if finger_contacts[int(actuator2finger[i])]:
            actuator_affecting_object_ids.append(i)
    # matrix that reduces the dimension 
    actuator_affecting_object_selectionmatrix = np.zeros((len(actuator_affecting_object_ids), actuator_num))
    for i, actuator_id in enumerate(actuator_affecting_object_ids):
        actuator_affecting_object_selectionmatrix[i, actuator_id] = 1

    # first update the estimation of the Jacobian, just for the actuators whose fingers are in contact with the object
    global J, p
    q = data.qpos[actuated_qpos_ids]
    dq = data.qvel[actuated_dof_ids]
    u = compute_task_space_vel_pen()
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

    task_space_vel_desired = compute_task_space_command_pen()
    # compute the updated commanded joint position which tries to achieve the desired task space vel while bringing it back to initial pose
    dt = model.opt.timestep
    ctrl_0 = init_ctrl[actuators_enabled] - data.ctrl[actuators_enabled]
    # Tikhonov regularization with a shifted center
    delta_q = np.linalg.inv(actuator_affecting_object_selectionmatrix.T@J_slice.T@J_slice@actuator_affecting_object_selectionmatrix + eps*np.eye(actuator_num)) @\
              (actuator_affecting_object_selectionmatrix.T@J_slice.T @ task_space_vel_desired + eps * ctrl_0) * dt
    # don't move too fast
    max_joint_vel = 10
    delta_q = np.clip(delta_q, -max_joint_vel*dt, max_joint_vel*dt)
    data.ctrl[actuators_enabled] += delta_q
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

mujoco.set_mjcb_control(control_cb)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # how much in the future and past to draw t
    t_trail_timerange = 1.
    # how many points to use to draw the future path
    t_draw_future_num_points = 10
    # add the required number of geoms to draw the future path
    scene = viewer.user_scn
    scene.ngeom += t_draw_future_num_points

    while viewer.is_running():
        #### DRAW PAST AND FUTURE PATH ####
        t = data.time
        for i in range(t_draw_future_num_points):
            t_ = t + (t_trail_timerange/2 - t_trail_timerange * i / t_draw_future_num_points)
            pos, _ = path(t_)
            rgba = np.array([1., 1., 1., 1. - i / t_draw_future_num_points])
            if i == t_draw_future_num_points//2:
                rgba[:] = [1, 0, 0, 1]  # make the current point red
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1-i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.001, 0, 0]),  # size
                pos,
                np.eye(3).flatten(),  # rotation
                rgba,
            )
        
        #### simulate ####
        mujoco.mj_step(model, data)
        viewer.sync()