import mujoco
from mujoco import viewer
import numpy as np

"""
run the simulation using the estimated Jacobian based controller
"""

#region <Mujoco Setup>
#=MUJOCO SETUP===================================================================================================================
model_path = "shadow_hand/scene_pen_realer.xml" 
model = mujoco.MjModel.from_xml_path(model_path)    # static state of the system (static model parameters like geometry, joints, acutators, hierarchy, etc.)
data = mujoco.MjData(model)                         # dynamic state of the system (joint positions, velocities, forces, contacts, etc.)
mujoco.mj_resetDataKeyframe(model, data, 0)         # Reset the state to keyframe 0 (keyframe is named snapshot of state stored in xml, keyframe zero is the first one)

print(f"Model nq: {model.nq}")  # Should be 31
print(f"Keyframe qpos length: {len(model.key_qpos[0])}")  # Should also be 31

# # Keyframe 0 (copied from the XML)
# init_ctrl = [0, 0,                                  # wrist actuators
#              0.24, 1.0, 0, 0.5, 0.3,                # thumb actuators
#              0., 0.4, 2.2,                           # forefinger actuators
#              0, 1.0, 2.0,                           # middle finger actuators   
#              0, 1.0, 3.14,                          # ring finger actuators
#              0, 0, 1.0, 3.14]                       # little finger actuators

init_ctrl = [0, 0, 
         0.80619, 0.80652, 0.002094, 0, 0.590452, 
         -0.073311, 0.4, 2.2, 
         -0.024437, 0.737076, 1.91662, 
         0, 1, 3.14, 
         0, 0, 1, 3.14]

init_ctrl = np.array(init_ctrl)
data.ctrl[:] = init_ctrl
#====================================================================================================================
#endregion

#region <Shadow Hand Setup> 
#=SHADOW HAND SETUP===================================================================================================================
# get the indices to access the robot's state (params for writing task with no wrist)

actuators_enabled = np.arange(0, model.nu)      # disable the first two actuators (wrist), else actuators_enabled = np.arange(model.nu) to enable all actuators 
actuator_num = len(actuators_enabled)
eps = 0.005                                     # how much we weigh the going back to init pose term without the wrist actuators, else eps = 0.002

actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in actuators_enabled]

# convert the actuator names to the names of the associated joints (specific to the shadow hand)
actuated_joint_names = [actuator_name.replace("_A_", "_") for actuator_name in actuator_names]
actuated_joint_names = [jnt_name.replace("0", "1") for jnt_name in actuated_joint_names]            # actuator names with "0" actuate tendons that go through joints with "1" in their names

# get the indices of the corresponding joints
actuated_joint_ids = []
for joint_name in actuated_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id != -1, f"Joint {joint_name} not found"
    actuated_joint_ids.append(joint_id)

print(f"{actuator_names=}\n{actuated_joint_names=}\n{actuated_joint_ids=}")

# creates list of start addresses in 'qvel' for joint's data
actuated_dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in actuated_joint_ids]
# creates list of start addresses in 'qpos' for joint's data
actuated_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in actuated_joint_ids]

print(f"{actuated_dof_ids=}\n{actuated_qpos_ids=}")

# find out which actuators belong to each finger
finger_name_filters = ["_TH", "_FF", "_MF", "_RF", "_LF"]
actuator2finger = np.zeros(actuator_num) - 1                                # -1 means not assigned to any finger
for i in range(actuator_num):
    for finger_id, finger_name_filter in enumerate(finger_name_filters):
        if finger_name_filter in actuator_names[i]:
            actuator2finger[i] = finger_id
            break
    if (actuator2finger[i] == -1):
        print(f"Actuator {actuator_names[i]} not assigned to any finger")

print(f"{actuator2finger=}")
#====================================================================================================================
#endregion

#region <Object Setup>
#=OBJECT SETUP===================================================================================================================
# get the indices to access the object's state, in this case, the object is the pen tip

object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
assert object_id != -1, "Object not found"

# build the index ranges

object_dof_ids = np.arange(model.body_dofadr[object_id], model.body_dofadr[object_id] + model.body_dofnum[object_id])
object_qpos_ids = np.arange(model.body_jntadr[object_id], model.body_jntadr[object_id] + 7)                             # free object has 7 qpos (3 for position, 4 for orientation (quaternions))

print(f"{object_id=}\n{object_dof_ids=}\n{object_qpos_ids=}")

pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
assert pen_id != -1, "Pen not found"

# recompute data and then use it to set the initial pose of the object
mujoco.mj_forward(model, data)
body_init_pose = data.qpos[object_qpos_ids].copy()

print(f"{body_init_pose=}")

# initialize the target position of the object
body_target_pos = np.zeros(3)
#====================================================================================================================
#endregion

#region <Jacobian and Global Variables Setup>
#=JACOBIAN SETUP===================================================================================================================
J = np.zeros((3, actuator_num))         # the estimated control Jacobian -> maps joint velocities to task space velocities

p = np.ones(actuator_num) * 1e-1        # covariance of the estimated J -> how uncertain we are about each entry in J

c_filtered = 1.0
finger_counter = np.zeros(5)           # counts how many consecutive steps each finger has been not in contact with the object, used for contact detection with some filtering to avoid flickering when the contact is lost for a few steps due to noise or other reasons
counter = 0
path_center = np.zeros(3)
object_radius = 0.001
"""
the covariance is used for weighing the update step (used for each column of J, which corresponds to an actuator, so it's a vector of length actuator_num)
technically this is a vector of variances for each actuator, but since we assume the noise is uncorrelated between actuators, the covariance matrix is diagonal and can be represented as a vector of variances.

as just mentioned the entries of the covariance matrix not on the main diagonal would define the inter actuator dependencies that the controller learns, 
for example, if two actuators always move together to achieve a certain task space velocity, then the covariance of those two actuators would be similar, 
and the update of one actuator would also update the other actuator in a similar way
"""
#====================================================================================================================
#endregion

#region <Contact Detection>
#=Contact Detection===================================================================================================================
def check_finger_contact():
    """
    check if each finger is in contact with the object
    """
    finger_name_filters = ['rh_th', 'rh_ff', 'rh_mf', 'rh_rf', 'rh_lf']                                         # if the body name contains any of these strings, it belongs to a finger
    finger_name_filters_additional = ['middle', 'distal']                                                   # if the body name contains any of these strings, it belongs to the parts of the finger
    finger_contact_detected = np.zeros(5)
    for contact in data.contact:
        collision_body_ids = [model.geom_bodyid[geom] for geom in contact.geom]                                 # collision detection is between geoms
        if pen_id in collision_body_ids:
            other_body_id = collision_body_ids[0] if collision_body_ids[1] == pen_id else collision_body_ids[1]
            other_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
            for i, finger_name_filter in enumerate(finger_name_filters):
                if finger_name_filter in other_body_name:
                    for additional_filter in finger_name_filters_additional:
                        if additional_filter in other_body_name:
                            finger_contact_detected[i] = 1
                            break

    global counter
    counter += 1
    if counter % 500 == 0:
        print(f"{finger_contact_detected=}")
        current_body_pos = data.xpos[object_id]
        print(f"{current_body_pos=}")
        if counter % 500 == 0:
            counter = 0

    return finger_contact_detected
#endregion

#region <Pen Task>
#=PEN TASK===================================================================================================================
def path(t):
    """
    when given a parameter t, returns the point on the path at t and the time derivative (i.e. velocity at that point)
    """
    r = 0.005                                   # radius
    # offset = np.array([0.09, -0.35, -0.068])    # offset to move the center of the path to a desired location (relative to the initial position of the object)
    time_scale_factor = 0.5                    
    slower_path = time_scale_factor * t

    global object_radius

    # circular path
    x = r * np.cos(slower_path)
    y = r * np.sin(slower_path)
    dx = -r * time_scale_factor * np.sin(slower_path)
    dy = r * time_scale_factor * np.cos(slower_path)

    t = data.time
    global path_center
    if t < 0.5:
        # pen_tip_init_pos = data.xpos[object_id] + np.array([0, 0, -0.0005]) 
        # start_offset = np.array([0.005, 0, 0])
        # path_center = pen_tip_init_pos + start_offset
        pen_tip_init_pos = data.xpos[object_id] + np.array([0, 0, -object_radius]) 
        start_offset = np.array([0, r, 0])
        path_center = pen_tip_init_pos + start_offset

    return np.array([x, y, 0]) + path_center, np.array([dx, dy, 0])

def compute_task_space_command_pen():
    t = data.time
    target_pos, target_vel = path(t)
    body_pos = data.xpos[object_id]
    task_space_vel = (target_pos - body_pos) * 9 + target_vel           # scale the position error with gain 8 to get the desired velocity, and add the target velocity
    return task_space_vel

def compute_task_space_vel_pen():
    return data.sensordata          # the sensor data is, as defined in the xml, the velocity of the pen tip
#====================================================================================================================
#endregion

#region <Control CB>
#=CONTROL CB===================================================================================================================
def control_cb(model, data):
    """
    callback function called on every step, and is used to set the control command
    """
    # check which fingers are in contact with the object
    finger_contacts = check_finger_contact()
    finger_contacts_filtered = finger_contacts.copy()
    global finger_counter
    for k in range(5):
        if finger_contacts[k] == 1: 
            finger_counter[k] = 0
        elif finger_contacts[k] == 0:
            finger_counter[k] += 1

            if finger_counter[k] > 50:
                finger_contacts_filtered[k] = 0
            else:
                finger_contacts_filtered[k] = 1
    
    # finger_contacts_filtered = finger_contacts


    contacts = sum(finger_contacts_filtered)
    c = contacts / 3.0
    global c_filtered
    c_filtered = (0.95) * c_filtered + (0.05) * c

    # compute which actuators currently affect the object (the finger that the actuator belongs to is in contact with the object)
    actuator_affecting_object_ids = []
    actuator_affecting_object_ids.append(0)
    actuator_affecting_object_ids.append(1)
    for i in range(actuator_num):
        # go through each actuator, check if the associated finger is in contact with the object
        if finger_contacts_filtered[int(actuator2finger[i])]:
            actuator_affecting_object_ids.append(i)

    # create reduced dimensionality matrix by excluding currently non-relevant actuators
    actuator_affecting_object_selectionmatrix = np.zeros((len(actuator_affecting_object_ids), actuator_num))
    for i, actuator_id in enumerate(actuator_affecting_object_ids):
        actuator_affecting_object_selectionmatrix[i, actuator_id] = 1

    # update the estimation of the relevant Jacobian J_slice and its covariance p_slice

    q = data.qpos[actuated_qpos_ids]
    dq = data.qvel[actuated_dof_ids]
    u = compute_task_space_vel_pen()
    r = 1e-3                                            # observation noise variance
    J_slice = J[:, actuator_affecting_object_ids]       # just update the part of the Jacobian that affects the object, this type of list accessing only includes the columns of J that correspond to the currently relevant actuators
    p_slice = p[actuator_affecting_object_ids]
    q_slice = q[actuator_affecting_object_ids]
    dq_slice = dq[actuator_affecting_object_ids]

    numerator = (u - J_slice @ dq_slice).reshape((-1, 1)) @ (p_slice * dq_slice).reshape((1, -1))
    denominator = p_slice.T @ (dq_slice * dq_slice) + r

    J_slice[:] += numerator / denominator
    p_slice[:] *= 1 - p_slice * dq_slice * dq_slice / denominator

    # update the full Jacobian and covariance matrix with the updated slices, and leave the unupdated parts as they are
    J[:, actuator_affecting_object_ids] = J_slice
    p[actuator_affecting_object_ids] = p_slice

    """
    Update Rule Explanation: 

        The update rule is derived from a probabilistic model where we assume that the error in task-space velocity prediction is Gaussian with variance r, and we also have a prior on the Jacobian with covariance p. 
        The update is a form of Bayesian update where we weigh the new information (the error in prediction) against our prior uncertainty (p) to get a new estimate of the Jacobian (J_slice) and its covariance (p_slice).
    
        -(u)                                    the observed task-space velocity, 
        -(J_slice @ dq_slice)                   the predicted task-space velocity based on our current Jacobian estimate and the joint velocities
        -(p_slice * dq_slice)                   weighted joint velocities (weighs the contribution of each actuator's velocity to the update based on our current uncertainty about that actuator's effect on the task-space velocity (the Jacobian entries corresponding to that actuator))
        -(p_slice.T @ (dq_slice * dq_slice))    gives us a measure of how much we expect the joint velocities to affect the task-space velocity based on our current uncertainty
        
        The numerator computes the product of the prediction error (u - J_slice @ dq_slice) and the weighted joint velocities (p_slice * dq_slice), which gives us a direction to update the Jacobian. 
        The denominator normalizes this update by the total uncertainty, which includes both the uncertainty in our Jacobian estimate (p_slice) and the observation noise (r).

        -> We want to update our Jacobian estimate in a way that reduces the prediction error, while also considering how certain we are about our current estimate (p_slice) and how noisy our observations are (r).

        We update the jacobian estimate by adding the new measurement weighted by the current uncertainty and how informative the measurement is (numerator/denominator). 
        If the prediction error is large and we are uncertain about the Jacobian (large p_slice), we will have a larger update. If the observation noise r is large, we will have a smaller update since we trust the new measurement less.
        
        We update the covariance p_slice to reflect whether we learned something new about the jacobian, in which case the uncertainty shrinks. 
        The more an actuator contributes to the task-space velocity (the larger p_slice * dq_slice * dq_slice is), the more we reduce our uncertainty about that actuator's effect on the task-space velocity.
        
        General Notes:
            -reshape(-1, 1) makes a column vector with as many rows as required
    """

    # compute the updated commanded joint positions which try to achieve the desired task space velocity while bringing it back to initial pose

    task_space_vel_desired = compute_task_space_command_pen()
    task_space_vel_desired_adjusted = c_filtered * task_space_vel_desired       # scale the desired task space velocity with the contact confidence, so that when the confidence is low, the controller tries to go back to the initial pose and explore around it to find a better configuration, and when the confidence is high, it tries to achieve the desired task space velocity

    dt = model.opt.timestep

    # calculate the pullback term to the initial pose, to avoid drifting too far from the initial pose and helps with exploration in the beginning when the Jacobian estimate is bad (by encouraging the controller to try different configurations around the initial pose)
    ctrl_0 = init_ctrl[actuators_enabled] - data.ctrl[actuators_enabled]

    eps_adjusted = eps * (1 + 0.5 * (1 - c_filtered))

    # Tikhonov regularization with a shifted center -> delta_q is a position style command (strictly it represents whatever the actuators ctrl represents in xml)
    # We use Tikhonov to ensure stable and invertible solution, and to introduce a bias towards the initial control command to prevent drift
    # it takes this form because this is originally a minimization of the cost function: ||J_slice @ delta_q - task_space_vel_desired||^2 + eps * ||delta_q - ctrl_0||^2, where the first term tries to achieve the desired task space velocity and the second term tries to keep the control command close to the initial command, and eps is the regularization parameter that weighs these two objectives
    delta_q = np.linalg.inv(actuator_affecting_object_selectionmatrix.T@J_slice.T@J_slice@actuator_affecting_object_selectionmatrix + eps*np.eye(actuator_num)) @\
              (actuator_affecting_object_selectionmatrix.T@J_slice.T @ task_space_vel_desired_adjusted + eps_adjusted * ctrl_0) * dt
    
    # delta_q = 0 * delta_q

    if np.max(np.abs(delta_q)) > 0.1:
        print(f"{delta_q=}")
    # limit speed (scale velocity with dt to get position limit)
    max_joint_vel = 10
    delta_q = np.clip(delta_q, -max_joint_vel*dt, max_joint_vel*dt)

    # update the control command for the next step, while enforcing the actuator control limits defined in the model
    data.ctrl[actuators_enabled] += delta_q
    data.ctrl[:] = np.clip(data.ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
#====================================================================================================================
#endregion

#region <Mujoco Viewer Setup>
#=MUJOCO SIM===================================================================================================================
mujoco.set_mjcb_control(control_cb)

with mujoco.viewer.launch_passive(model, data) as viewer:
# with mujoco.viewer.launch(model, data) as viewer:


    viewer.cam.distance = 1
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20

    t_path_visualization_timerange = 4.0                     # how much in the future and past to draw path
    t_path_draw_future_num_points = 10               # how many points to use to draw the future path
    
    trail_len = 500                             # max number of points in pen-tip trail (circular buffer)
    trail_stride = 5                            # record every N steps to control trail density
    trail_positions = [None] * trail_len        # circular buffer for trail positions
    trail_head = 0                              # current write position in circular buffer
    trail_count = 0                             # number of valid positions in buffer
    trail_step_counter = 0


    
    # add the required number of geoms to draw the future path + permanent trail
    scene = viewer.user_scn
    future_geom_start = scene.ngeom
    scene.ngeom += t_path_draw_future_num_points + trail_len

    while viewer.is_running():
        t = data.time
        trail_step_counter += 1

        # saved_pen_qpos = data.qpos[-7:].copy()
        # print(f"{saved_pen_qpos=}")
        
        # Record pen-tip position for permanent trail (circular buffer)
        if trail_step_counter % trail_stride == 0:
            trail_positions[trail_head] = data.xpos[object_id].copy()
            trail_head = (trail_head + 1) % trail_len
            trail_count = min(trail_count + 1, trail_len)
        
        # Draw path visualization (red and blue)
        for i in range(t_path_draw_future_num_points):
            t_ = t + (t_path_visualization_timerange/2 - t_path_visualization_timerange * i / t_path_draw_future_num_points)
            pos, _ = path(t_)
            rgba = np.array([0.0, 0.0, 1.0, 1.0 - i / t_path_draw_future_num_points])
            # make current point red and fully opaque
            if i == t_path_draw_future_num_points//2:
                rgba[:] = [1, 0, 0, 1]  
        
            mujoco.mjv_initGeom(scene.geoms[future_geom_start + i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0005, 0, 0]),
                pos,
                np.eye(3).flatten(),
                rgba,
            )
        
        # Draw permanent pen-tip trail (black)
        for i in range(trail_count):
            # Calculate index in circular buffer (oldest to newest)
            idx = (trail_head - trail_count + i) % trail_len
            pos = trail_positions[idx]
            if pos is None:
                continue
            rgba = np.array([0.0, 0.0, 0.0, 1.0])  # solid black
            
            mujoco.mjv_initGeom(scene.geoms[future_geom_start + t_path_draw_future_num_points + i],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.0007, 0, 0]),
                pos,
                np.eye(3).flatten(),
                rgba,
            )
        
        # update the simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        
#====================================================================================================================
#endregion