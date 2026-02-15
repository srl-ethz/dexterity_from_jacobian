import mujoco
from mujoco import viewer
import numpy as np
from enum import Enum

"""
run the simulation using the estimated Jacobian based controller
"""

# Global variables for adjusting

# Task space command
position_gain = 9.0                     # gain for the position error term in the task space velocity command
eps = 0.005                             # how much we weigh the going back to init pose term in the task space velocity command, (tradeoff between preventing drifting when no contact and worse path following when contact)

# Path tracking
r = 0.005                               # path radius
time_scale_factor = 0.5                 # speed of path          
following_time_limit = 0.5              # time that path center follows moving pen tip (what worked best so far is 0.05, 0.5 (main one) 0.8, 1)
grid_period = 10.0                      # period of the grid path, i.e. how long it takes in SCALED TIME to do one full loop of the grid path, i.e. if we scale time by 0.5 and the period is 10.0 that means it takes 20 seconds !!!! HOWEVER THIS IS IN SIM TIME WHICH IS SLOWER THAN REAL TIME DUE TO OVERHEAD SO IN REALITY ITS LONGER !!!!

# Path grid setup
segment_length = 0.005                     # length of segments in the grid for writing letters,
deactivate_keyboard_input = False          # if true, we ignore keyboard input and just follow the shape defined by path_shape
object_init_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # initial pose of the object

# Scene setup
testing = False                         # if true we give zero command - mode used to tune grip manually

# Path Shapes
class PathShape(Enum):
    REST = 0
    CIRCLE = 1
    FIGURE8 = 2
    SQUARE = 3
    TRIANGLE = 4
    BOOTLEG_LETTER_A = 5                # works but ugly
path_shape = PathShape.REST

# Global variables for initialization
path_center = np.zeros(3)               # initial path center
c_filtered = 1.0                        # initial value for c_filtered
object_radius = 0.001                   # pentip radius copied from xml
counter = 0                             # counts up to 500 for reduced printing output
counter2 = 0                            # counts up to 500 for reduced printing output for keyboard input vertices
finger_counter = np.zeros(5)            # counts how many consecutive steps each finger has been not in contact with the object

class SimulationController:
    def __init__(self):
        self.inputs = []
        self.start_path = False
        self.active_letter = None
        self.t_start = 0.0
        self.path_flag = False
        self.letter_anchor = np.zeros(3)
        self.letter_finished = False
        
    def keyboard_callback(self, keycode):
        """Called whenever a key is pressed during simulation"""
        if keycode == 256:  # Enter key
            # print(f"Start pathtracking with inputs: {self.inputs}")
            self.start_path = True
        elif keycode >= 48 and keycode <= 57:  # Number keys 0-9
            self.inputs.append(keycode - 48)
        elif keycode == 65 or keycode == 97:  # A or a
            self.inputs.append('A')
            print(f"Added 'A' to inputs, current inputs: {self.inputs}")
        elif keycode == 66 or keycode == 98:  # B or b
            self.inputs.append('B')
            print(f"Added 'B' to inputs, current inputs: {self.inputs}")
        elif keycode == 67 or keycode == 99:  # C or c
            self.inputs.append('C')
            print(f"Added 'C' to inputs, current inputs: {self.inputs}")
        elif keycode == 68 or keycode == 100:  # D or d
            self.inputs.append('D')
            print(f"Added 'D' to inputs, current inputs: {self.inputs}")
        elif keycode == 69 or keycode == 101:  # E or e
            self.inputs.append('E')
            print(f"Added 'E' to inputs, current inputs: {self.inputs}")
        elif keycode == 70 or keycode == 102:  # F or f
            self.inputs.append('F')
            print(f"Added 'F' to inputs, current inputs: {self.inputs}")
        elif keycode == 71 or keycode == 103:  # G or g
            self.inputs.append('G')
            print(f"Added 'G' to inputs, current inputs: {self.inputs}")
        elif keycode == 72 or keycode == 104:  # H or h
            self.inputs.append('H')
            print(f"Added 'H' to inputs, current inputs: {self.inputs}")
        elif keycode == 73 or keycode == 105:  # I or i
            self.inputs.append('I')
            print(f"Added 'I' to inputs, current inputs: {self.inputs}")
        elif keycode == 74 or keycode == 106:  # J or j
            self.inputs.append('J')
            print(f"Added 'J' to inputs, current inputs: {self.inputs}")
        elif keycode == 75 or keycode == 107:  # K or k
            self.inputs.append('K')
            print(f"Added 'K' to inputs, current inputs: {self.inputs}")
        elif keycode == 76 or keycode == 108:  # L or l
            self.inputs.append('L')
            print(f"Added 'L' to inputs, current inputs: {self.inputs}")
        elif keycode == 77 or keycode == 109:  # M or m
            self.inputs.append('M')
            print(f"Added 'M' to inputs, current inputs: {self.inputs}")
        elif keycode == 78 or keycode == 110:  # N or n
            self.inputs.append('N')
            print(f"Added 'N' to inputs, current inputs: {self.inputs}")
        elif keycode == 79 or keycode == 111:  # O or o
            self.inputs.append('O')
            print(f"Added 'O' to inputs, current inputs: {self.inputs}")
        elif keycode == 80 or keycode == 112:  # P or p
            self.inputs.append('P')
            print(f"Added 'P' to inputs, current inputs: {self.inputs}")
        elif keycode == 81 or keycode == 113:  # Q or q
            self.inputs.append('Q')
            print(f"Added 'Q' to inputs, current inputs: {self.inputs}")
        elif keycode == 82 or keycode == 114:  # R or r
            self.inputs.append('R')
            print(f"Added 'R' to inputs, current inputs: {self.inputs}")
        elif keycode == 83 or keycode == 115:  # S or s
            self.inputs.append('S')
            print(f"Added 'S' to inputs, current inputs: {self.inputs}")
        elif keycode == 84 or keycode == 116:  # T or t
            self.inputs.append('T')
            print(f"Added 'T' to inputs, current inputs: {self.inputs}")
        elif keycode == 85 or keycode == 117:  # U or u
            self.inputs.append('U')
            print(f"Added 'U' to inputs, current inputs: {self.inputs}")
        elif keycode == 86 or keycode == 118:  # V or v
            self.inputs.append('V')
            print(f"Added 'V' to inputs, current inputs: {self.inputs}")
        elif keycode == 87 or keycode == 119:  # W or w
            self.inputs.append('W')
            print(f"Added 'W' to inputs, current inputs: {self.inputs}")
        elif keycode == 88 or keycode == 120:  # X or x
            self.inputs.append('X')
            print(f"Added 'X' to inputs, current inputs: {self.inputs}")
        elif keycode == 89 or keycode == 121:  # Y or y
            self.inputs.append('Y')
            print(f"Added 'Y' to inputs, current inputs: {self.inputs}")
        elif keycode == 90 or keycode == 122:  # Z or z
            self.inputs.append('Z')
            print(f"Added 'Z' to inputs, current inputs: {self.inputs}")
        
controller = SimulationController()

#region <Mujoco Setup>
#=MUJOCO SETUP===================================================================================================================
model_path = "shadow_hand/scene_pen_realer.xml" 
model = mujoco.MjModel.from_xml_path(model_path)    # static state of the system (static model parameters like geometry, joints, acutators, hierarchy, etc.)
data = mujoco.MjData(model)                         # dynamic state of the system (joint positions, velocities, forces, contacts, etc.)
mujoco.mj_resetDataKeyframe(model, data, 6)         # Reset the state to keyframe 0 (keyframe is named snapshot of state stored in xml, keyframe zero is the first one)

init_ctrl = data.ctrl.copy()                        # save the initial control command from keyframe  
#====================================================================================================================
#endregion

#region <Mujoco Helper Functions>
#=MUJOCO HELPER FUNCTIONS===================================================================================================================
def reset():
    """
    reset everything upon keyframe change
    """
    global J, p, c_filtered, path_center, counter, finger_counter, object_init_pose
    J[:] = 0
    p[:] = 1e-1
    c_filtered = 1.0
    path_center[:] = 0.0
    counter = 0
    finger_counter[:] = 0
    controller.active_letter = None
    controller.inputs = []
    controller.t_start = 0.0
    controller.path_flag = False
    object_init_pose = data.qpos[object_qpos_ids].copy()
    controller.letter_anchor = data.xpos[object_id].copy()

    init_ctrl[:] = data.ctrl.copy()    # update the initial control command to the new keyframe's control command

#====================================================================================================================
#endregion

#region <Shadow Hand Setup> 
#=SHADOW HAND SETUP===================================================================================================================
actuators_enabled = np.arange(0, model.nu)      # enable all actuators 
actuator_num = len(actuators_enabled)

actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in actuators_enabled]

# convert the actuator names to the names of the associated joints (specific to the shadow hand)
actuated_joint_names = [actuator_name.replace("_A_", "_") for actuator_name in actuator_names]
actuated_joint_names = [jnt_name.replace("0", "1") for jnt_name in actuated_joint_names]        # actuator names with "0" actuate tendons that go through joints with "1" in their names

# get the indices of the corresponding actuated joints
actuated_joint_ids = []
for joint_name in actuated_joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id != -1, f"Joint {joint_name} not found"
    actuated_joint_ids.append(joint_id)

# print(f"{actuator_names=}\n{actuated_joint_names=}\n{actuated_joint_ids=}")

# creates list of start addresses in 'qvel' for joint's data
actuated_dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in actuated_joint_ids]
# creates list of start addresses in 'qpos' for joint's data
actuated_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in actuated_joint_ids]

# print(f"{actuated_dof_ids=}\n{actuated_qpos_ids=}")

# find out which actuators belong to each finger
finger_name_filters = ["_TH", "_FF", "_MF", "_RF", "_LF"]
actuator2finger = np.zeros(actuator_num) - 1                                # -1 means not assigned to any finger
for i in range(actuator_num):
    for finger_id, finger_name_filter in enumerate(finger_name_filters):
        if finger_name_filter in actuator_names[i]:
            actuator2finger[i] = finger_id
            break
    # if (actuator2finger[i] == -1):
    #     print(f"Actuator {actuator_names[i]} not assigned to any finger")
# print(f"{actuator2finger=}")
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

# print(f"{object_id=}\n{object_dof_ids=}\n{object_qpos_ids=}")

pen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen")
assert pen_id != -1, "Pen not found"

# recompute data and then use it to set the initial pose of the object
mujoco.mj_forward(model, data)     # <- might not be necessary here
object_init_pose = data.qpos[object_qpos_ids].copy()

# print(f"{object_init_pose=}")
#====================================================================================================================
#endregion

#region <Jacobian and Covariance Setup>
#=JACOBIAN AND COVARIANCE SETUP===================================================================================================================
J = np.zeros((3, actuator_num))         # the estimated control Jacobian -> maps joint velocities to task space velocities

p = np.ones(actuator_num) * 1e-1        # covariance of the estimated J -> how uncertain we are about each entry in J

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
    finger_name_filters = ['rh_th', 'rh_ff', 'rh_mf', 'rh_rf', 'rh_lf']                 # if the body name contains any of these strings, it belongs to a finger
    finger_name_filters_additional = ['middle', 'distal']                               # if the body name contains any of these strings, it belongs to the relevant parts of the finger
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

    # global counter
    # counter += 1
    # if counter % 500 == 0:
    #     print(f"{finger_contact_detected=}")
    #     counter = 0
        # current_body_pos = data.xpos[object_id]
        # print(f"{current_body_pos=}")

    return finger_contact_detected
#====================================================================================================================
#endregion

#region <Pen Task>
#=PEN TASK===================================================================================================================
def piecewise_path(vertices, adjusted_time, time_scale_factor):
    """
    Evaluate position and velocity on a closed piecewise linear path.
    vertices: list of (x, y) tuples defining the closed polygon
    Returns: x, y, dx_dt, dy_dt
    """
    global grid_period
    
    n = len(vertices)       # number of corners
    seg_lengths = []        # segment lengths
    
    # calculate segment lengths between all vertices starting with vertex 0 and 1 and ending with vertex n-1 and 0 to close the loop
    for i in range(n):
        d = np.sqrt((vertices[(i+1)%n][0] - vertices[i][0])**2 +
                    (vertices[(i+1)%n][1] - vertices[i][1])**2)
        seg_lengths.append(d) 
    total_len = sum(seg_lengths)    # total shape perimeter

    # map adjusted_time (period 10.0) to distance along perimeter, 
    # i.e. find the equivalent distance along the perimeter for the given adjusted_time parameter
    dist = (adjusted_time % (grid_period)) / (grid_period) * total_len
    if dist < 0:
        dist += total_len

    # find which segment and interpolate
    acc = 0             # accumulated length
    for i in range(n):
        # if the following is true we are on the ith segment, so we can calculate the position and velocity by interpolating between vertex i and vertex i+1
        if acc + seg_lengths[i] > dist + 1e-12:         
            frac = (dist - acc) / seg_lengths[i]            # fraction of segment length we have covered
            # position along segment i according to the fraction we have covered
            x = vertices[i][0] + frac * (vertices[(i+1)%n][0] - vertices[i][0])
            y = vertices[i][1] + frac * (vertices[(i+1)%n][1] - vertices[i][1])
            # velocity: d(pos)/dt = direction * (total_len / 2pi) * time_scale_factor
            dir_x = (vertices[(i+1)%n][0] - vertices[i][0]) / seg_lengths[i]
            dir_y = (vertices[(i+1)%n][1] - vertices[i][1]) / seg_lengths[i]
            speed = total_len / (grid_period) * time_scale_factor         # dont forget to scale the speed with the time_scale_factor
            return x, y, dir_x * speed, dir_y * speed
        # if acc + seg_lengths[i] is not greater than dist, move to the next segment and update the accumulated length
        acc += seg_lengths[i]

    # fallback to first vertex (safety feature)
    return vertices[0][0], vertices[0][1], 0.0, 0.0

def grid_definition(letter=None):
    """
    Define a 3x3 grid of vertices for writing letters
    And then a dictionary of letter paths that define the order in which we visit the vertices for each letter
    """

    global segment_length
    global counter2
    global object_init_pose

    # positions of the vertices (the flipped order is due to how the hand is positioned in space)
    center_center = (x0, y0) = (0.0, 0.0)
    center_left = (x0 - segment_length, y0)
    center_right = (x0 + segment_length, y0)
    bottom_center = (x0, y0 + segment_length)
    bottom_left = (x0 - segment_length, y0 + segment_length)
    bottom_right = (x0 + segment_length, y0 + segment_length)
    top_center = (x0, y0 - segment_length)
    top_left = (x0 - segment_length, y0 - segment_length)
    top_right = (x0 + segment_length, y0 - segment_length)

    # Default layout and fallback path if letter not LETTER_PATHS
    vertices = [
        (top_left), (top_center), (top_right),
        (center_left), (center_center), (center_right),
        (bottom_left), (bottom_center), (bottom_right)
    ]

    LETTER_PATHS = {
        'A': [center_center, bottom_center, top_center, top_left, bottom_left, center_left, center_center],
        'B': [center_center, bottom_center, bottom_left, top_left, top_center, center_center, center_left, center_center],
        'C': [center_center, bottom_center, bottom_left, bottom_center, top_center, top_left, top_center, center_center],
        'D': [center_center, bottom_center, center_left, top_center, center_center],
        'E': [center_center, center_left, center_center, bottom_center, bottom_left, bottom_center, top_center, top_left, top_center, center_center],
        'F': [center_center, center_left, center_center, bottom_center, top_center, top_left, top_center, center_center],
        'G': [center_center, bottom_center, bottom_left, center_left, bottom_left, bottom_center, top_center, top_left, top_center, center_center],
        'H': [center_center, bottom_center, top_center, center_center, center_left, bottom_left, top_left, center_left, center_center],
        'I': [center_center, top_center, bottom_center, center_center],
    }

    vertices = list(LETTER_PATHS.get(letter, vertices))     # if letter is not in LETTER_PATHS, use the full grid as default

    # counter2 += 1
    # if counter2 % 500 == 0:
    #     print(f"Defined vertices for letter {letter}: {vertices}")
    #     print(f"Initial object pose for grid definition: {object_init_pose}")
    #     counter2 = 0

    return vertices

def path(t):
    """
    when given a parameter t, returns the point on the path at t and the time derivative (i.e. velocity at that point)
    """

    global r   
    global time_scale_factor
    global object_radius
    global path_center
    global following_time_limit
    global path_shape
    global deactivate_keyboard_input

    time = data.time
    adjusted_time = time_scale_factor * (t - following_time_limit)

    # define x, y, dx, dy for safety
    x, y, dx, dy = 0.0, 0.0, 0.0, 0.0

    if deactivate_keyboard_input == True:
        controller.inputs = []
        controller.active_letter = None

    # only modify controller state when called from the control loop,
    # not from visualization calls which use different t values just to draw the path
    is_control_call = (abs(t - data.time) < 1e-3)   

    if is_control_call:
        if controller.inputs or controller.active_letter is not None:     # if we have inputs to process or we are currently processing a letter, we want to follow the grid path, otherwise we just follow the shape defined by path_shape
            if controller.active_letter is None:

                controller.letter_anchor = data.xpos[object_id].copy()     # we set the anchor to the current pen position, so that the path is defined around the current pentip position
                controller.t_start = time
                controller.active_letter = controller.inputs.pop(0)
                controller.path_flag = False
                print(f"Starting to follow letter {controller.active_letter} remaining inputs: {controller.inputs}")

            elif controller.active_letter is not None:
                if (time - controller.t_start) > (following_time_limit + grid_period / time_scale_factor):   # after one full loop of the circle, we can move on to the next letter, this is just to give some time to settle on the new path before we start following it
                    
                    controller.letter_finished = True
                    controller.active_letter = None
                    print(f"Finished following letter {controller.active_letter}")

                    if controller.inputs:
                        controller.letter_anchor = data.xpos[object_id].copy()     # update the anchor to the current pen position, so that the grid moves with the pen, this is important for when we start following the path, so that the path is already under the pen tip and we dont have to wait for it to move there
                        controller.t_start = time
                        controller.active_letter = controller.inputs.pop(0)
                        controller.path_flag = False
                        print(f"Starting to follow letter {controller.active_letter} remaining inputs: {controller.inputs}")
                    
                    else:
                        controller.letter_finished = True
                        controller.path_flag = True
                        print(f"No more letters to follow, switching to shape following")
        else: 
            controller.path_flag = True
            print("Following shape defined by path_shape since no letter inputs detected")

    # Evaluate the letter path (read-only, safe to call from visualization too)
    if controller.active_letter is not None and not controller.path_flag:
        adjusted_time = time_scale_factor * (time - controller.t_start)
        vertices = grid_definition(controller.active_letter)
        x, y, dx, dy = piecewise_path(vertices, adjusted_time, time_scale_factor)
        return np.array([x, y, -object_radius]) + controller.letter_anchor, np.array([dx, dy, 0])

    if controller.path_flag:
        
        if path_shape == PathShape.REST:
            x, y, dx, dy = 0.0, 0.0, 0.0, 0.0

        elif path_shape == PathShape.CIRCLE:
            #this is some additional finetuning for the circle, but not absolutely necessary by any means, we could just have adjusted_time = time_scale_factor * (t - following_time_limit)
            if time < following_time_limit:
                adjusted_time = -np.pi / 2                                           # path point sits at pen tip during settling
            else:
                adjusted_time = time_scale_factor * (t - following_time_limit) - np.pi / 2   # start at pen tip, then move along circle

            # # # circular path
            x = r * np.cos(adjusted_time)
            y = r * np.sin(adjusted_time)
            dx = -r * time_scale_factor * np.sin(adjusted_time)
            dy = r * time_scale_factor * np.cos(adjusted_time)

        elif path_shape == PathShape.FIGURE8:

            # figure 8 path
            r = 0.01
            x = r * (np.cos(adjusted_time) / (1 + np.sin(adjusted_time)**2))
            y = r * (np.sin(adjusted_time) * np.cos(adjusted_time) / (1 + np.sin(adjusted_time)**2))
            dx = r*(np.sin(adjusted_time)**2 - 3)*np.sin(adjusted_time)/(np.sin(adjusted_time)**2 + 1)**2
            dy = r*(1 - 3*np.sin(adjusted_time)**2)/(np.sin(adjusted_time)**2 + 1)**2

        elif path_shape == PathShape.SQUARE:
            # square with side 2r, starting from bottom-center going right
            verts = [(0, -r), (r, -r), (r, r), (-r, r), (-r, -r)]
            x, y, dx, dy = piecewise_path(verts, adjusted_time, time_scale_factor)

        elif path_shape == PathShape.TRIANGLE:
            # equilateral triangle inscribed in circle of radius r, starting from bottom
            verts = [(0, -r), (r*np.sqrt(3)/2, r/2), (-r*np.sqrt(3)/2, r/2)]
            x, y, dx, dy = piecewise_path(verts, adjusted_time, time_scale_factor)

        elif path_shape == PathShape.BOOTLEG_LETTER_A:
            # letter A: left leg up, back to middle, crossbar right, right leg up, right leg down, return
            verts = [(-r, -r), (-r, r), (-r, 0), (r, 0), (r, r), (r, -r), (-r, -r)]
            x, y, dx, dy = piecewise_path(verts, adjusted_time, time_scale_factor)
    
    if time < following_time_limit:
        pen_tip_init_pos = data.xpos[object_id] + np.array([0, 0, -object_radius])
        path_center = pen_tip_init_pos - np.array([x, y, 0])                # place path_center so that current (x, y) lands on pen tip

    return np.array([x, y, 0]) + path_center, np.array([dx, dy, 0])

def compute_task_space_command_pen():

    global position_gain
    t = data.time
    target_pos, target_vel = path(t)
    body_pos = data.xpos[object_id]
    task_space_vel = (target_pos - body_pos) * position_gain + target_vel           # scale the position error with position gain to get the desired velocity, and add the target velocity
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
    
    global finger_counter
    global c_filtered

    obs_noise = 1e-3                                            # observation noise variance
    finger_contact_threshold_time = 50                          # number of consecutive steps a finger has to be not in contact before we consider it as not affecting the object
    C_filter_weight = 0.95                                      # weight for the low pass filter on the contact state
    eps_scaling = 0.5
    time = data.time
    dt = model.opt.timestep

    # check which fingers are in contact with the object
    finger_contacts = check_finger_contact()
    finger_contacts_filtered = finger_contacts.copy()
    for k in range(5):
        if finger_contacts[k] == 1: 
            finger_counter[k] = 0
        elif finger_contacts[k] == 0:
            finger_counter[k] += 1
            if finger_counter[k] > finger_contact_threshold_time:
                finger_contacts_filtered[k] = 0
            else:
                finger_contacts_filtered[k] = 1
    contacts = sum(finger_contacts_filtered)
    c = contacts / 3.0
    c_filtered = C_filter_weight * c_filtered + (1 - C_filter_weight) * c

    # compute which actuators currently affect the object (the finger that the actuator belongs to is in contact with the object)
    actuator_affecting_object_ids = []
    actuator_affecting_object_ids.append(0)     # the wrist always affects the object
    actuator_affecting_object_ids.append(1)
    for i in range(actuator_num):
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
    
    J_slice = J[:, actuator_affecting_object_ids]       # just update the part of the Jacobian that affects the object, this type of list accessing only includes the columns of J that correspond to the currently relevant actuators
    p_slice = p[actuator_affecting_object_ids]
    
    q_slice = q[actuator_affecting_object_ids]
    dq_slice = dq[actuator_affecting_object_ids]

    numerator = (u - J_slice @ dq_slice).reshape((-1, 1)) @ (p_slice * dq_slice).reshape((1, -1))
    denominator = p_slice.T @ (dq_slice * dq_slice) + obs_noise

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

    # calculate the task space velocity command, 
    # initially either set it to zero for the pen to settle into its initial pose without following the path right  or random noise to already get initial values for the jacobian
    # while avoiding the pen trying to follow the path and the path movign together with it, which increases the speed
    if time < following_time_limit:
        # task_space_vel_desired_adjusted = np.zeros(3) 
        task_space_vel_desired_adjusted = np.random.randn(3) * 5 * 1e-2     # add a bit of noise to the command to encourage exploration and learning of the Jacobian even when the pen is not moving much, which is important for learning the correct Jacobian entries for the actuators that affect the pen but are not yet in contact with the pen at the beginning
    else:
        task_space_vel_desired = compute_task_space_command_pen()
        task_space_vel_desired_adjusted = c_filtered * task_space_vel_desired       # scale the desired task space velocity with the contact confidence, so that when the confidence is low, the controller tries iess hard to achieve the desired task space velocity

    # calculate the pullback term to the initial pose, to avoid drifting too far from the initial pose 
    ctrl_0 = init_ctrl[actuators_enabled] - data.ctrl[actuators_enabled]
    eps_adjusted = eps * (1 + eps_scaling * (1 - c_filtered))                       # scale the regularization term with the contact confidence, so that when the confidence is low, we weigh more the going back to initial pose term

    # Tikhonov regularization
    """
    # Tikhonov regularization with a shifted center -> delta_q is a position style command (strictly it represents whatever the actuators ctrl represents in xml)
    # We use Tikhonov to ensure stable and invertible solution, and to introduce a bias towards the initial control command to prevent drift
    # it takes this form because this is originally a minimization of the cost function: ||J_slice @ delta_q - task_space_vel_desired||^2 + eps * ||delta_q - ctrl_0||^2, 
    # where the first term tries to achieve the desired task space velocity and the second term tries to keep the control command close to the initial command, and eps is the regularization parameter that weighs these two objectives
    """        
    delta_q = np.linalg.inv(actuator_affecting_object_selectionmatrix.T@J_slice.T@J_slice@actuator_affecting_object_selectionmatrix + eps*np.eye(actuator_num)) @\
              (actuator_affecting_object_selectionmatrix.T@J_slice.T @ task_space_vel_desired_adjusted + eps_adjusted * ctrl_0) * dt
    
    # if we are testing or defining grasping pose, we command zero velocity
    if testing:
        delta_q = 0 * delta_q

    # limit speed (scale velocity with dt to get position limit) <- never actually anywhere near this limit 
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

with mujoco.viewer.launch_passive(model, data, key_callback=controller.keyboard_callback) as viewer:

    viewer.cam.distance = 1
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20

    t_path_visualization_timerange = 4.0                # how much in the future and past to draw path
    t_path_draw_future_num_points = 10                  # how many points to use to draw the future path
    
    trail_len = 2000                                    # max number of points in pen-tip trail (circular buffer)
    trail_stride = 5                                    # record every N steps to control trail density
    trail_positions = [None] * trail_len                # circular buffer for trail positions
    trail_head = 0                                      # current write position in circular buffer
    trail_count = 0                                     # number of valid positions in buffer
    trail_step_counter = 0                              # counts steps to determine when to record trail position

    trail_copy_offset = np.array([0.0, 0.05, 0.0])      # offset for the shifted trail copy

    # add the required number of geoms to draw the future path + permanent trail + shifted copy
    scene = viewer.user_scn
    future_geom_start = scene.ngeom
    scene.ngeom += t_path_draw_future_num_points + trail_len + trail_len

    # keep track of time to identify resets
    prev_time = data.time

    # hide old trail geoms (both original and shifted copy) that are still in the scene
    def clear_trail_geoms(trail_length):
        for i in range(trail_length):
            for offset in [0, trail_len]:  # original + shifted copy
                mujoco.mjv_initGeom(scene.geoms[future_geom_start + t_path_draw_future_num_points + offset + i],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.array([0.0, 0.0, 0.0, 0.0]),
                )

    def draw(position, color, size, geom_index):
        mujoco.mjv_initGeom(scene.geoms[geom_index],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([size, 0, 0]),
            position,
            np.eye(3).flatten(),
            color,
        )

    while viewer.is_running():

        t = data.time
        trail_step_counter += 1

        # clear trail when a letter finishes
        if controller.letter_finished:
            controller.letter_finished = False
            trail_positions = [None] * trail_len
            trail_head = 0
            trail_count = 0
            trail_step_counter = 0
            clear_trail_geoms(trail_len)
                    
        # reset everything if we detect a reset in the simulation (time goes backwards)
        if t < prev_time: 
            reset()

            # clear trail
            trail_positions = [None] * trail_len
            trail_head = 0
            trail_count = 0
            trail_step_counter = 0
            clear_trail_geoms(trail_len)
            # print("Reset detected!")

        # continiously update time to detect resets
        prev_time = t
        
        # Draw path visualization (red and blue)
        for i in range(t_path_draw_future_num_points):
            t_ = t + (t_path_visualization_timerange/2 - t_path_visualization_timerange * i / t_path_draw_future_num_points)
            pos, _ = path(t_)
            rgba = np.array([0.0, 0.0, 1.0, 1.0 - i / t_path_draw_future_num_points])
            # make current point red and fully opaque
            if i == t_path_draw_future_num_points//2:
                rgba[:] = [1, 0, 0, 1]  

            draw(pos, rgba, 0.0005, future_geom_start + i)

        # Record pen-tip position for permanent trail (circular buffer)
        if t > 5* following_time_limit:
            if trail_step_counter % trail_stride == 0:
                trail_positions[trail_head] = data.xpos[object_id].copy()
                trail_head = (trail_head + 1) % trail_len
                trail_count = min(trail_count + 1, trail_len)
        
            # Draw permanent and shifted pen-tip trails (white)
            for i in range(trail_count):
                # Calculate index in circular buffer (oldest to newest)
                idx = (trail_head - trail_count + i) % trail_len
                pos = trail_positions[idx]
                if pos is None:
                    continue
                rgba = np.array([1.0, 1.0, 1.0, 1.0])  # solid white

                draw(pos, rgba, 0.0002, future_geom_start + t_path_draw_future_num_points + i)  # original trail
                draw(pos + trail_copy_offset, rgba, 0.0002, future_geom_start + t_path_draw_future_num_points + trail_len + i)  # shifted copy
                
        # update the simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        
#====================================================================================================================
#endregion