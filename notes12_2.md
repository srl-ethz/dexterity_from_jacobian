notes 13.2

ok grip and mujoco is decent now, now i need to start with algo 

it needs to follow the path quicker
and better finger gating, namely it needs to not just remove the finger from the pen, i.e. maybe lock it so that the finger stays in contact with the 
pen above all else also include the wrist to see if that helps increadibly small area of writig, which is due to the fingers not being able to do big motions

especially the flat pencil grip, which would be the most stable i think, isnt currently feasible due to no range of motino, that might change with wrist activity


potential idea: use high level controller to improve grip, i.e. maximize grip quality metric, while not affecting pen tip position, and use the wrist actuators to move pen tip to follow the path 


another thing i need to to is to lower the ben in the grip, that should also increase the area it can use to draw, plus that will be necessary to use this in reality where we cant draw on air






what currently works:

keyframe 6
r = 0.005
time_scale_factor = 0.5
following_time_limit = 0.5
-0.012
friction 1.5
lower paper 

with r = 0.005, time_scale_factor = 0.5 and t < 0.05> both with 1.5 friction and paper at low position and pen position at z axis -0.012:
(i think these settings gave me nice and flat circles one or twice)
    qpos from keyframe 0 and ctrl from keyframe 5/6 (startup settings)      -> nice circles at a big angle          
    keyframe 5                                                              -> nice circles at a small angle
    keyframe 6                                                              -> nice circles at a small angle
with r = 0.004, time_scale_factor = 0.5 and t < 0.05> both with 1.5 friction and paper at low position and pen position at z axis -0.012:
    qpos from keyframe 0 and ctrl from keyframe 5/6 (startup settings)      -> nice circle at a steep angle          
    keyframe 5                                                              -> circle at a steep angle
    keyframe 6                                                              -> circle at a steep angle

with r = 0.004, time_scale_factor = 0.5 and t < 0.05> both with 1.5 friction and paper at low position and pen position at z axis -0.005:
    qpos from keyframe 0 and ctrl from keyframe 5/6 (startup settings)      -> nice circle at a steep angle          
    keyframe 5                                                              -> circle at a steep angle
    keyframe 6                                                              -> ugly circle at a low angle

with r = 0.005, time_scale_factor = 0.5 and t < 0.05> both with 1.5 friction and paper at low position and pen position at z axis -0.005:
    qpos from keyframe 0 and ctrl from keyframe 5/6 (startup settings)      -> nice circle at a steep angle          
    keyframe 5                                                              -> nothing useful
    keyframe 6                                                              -> nothing useful


with r = 0.004, time_scale_factor = 0.5 and t < 0.05> both with 1.5 friction and paper at low position and pen position at z axis -0.009:
    qpos from keyframe 0 and ctrl from keyframe 5/6 (startup settings)      ->           
    keyframe 5                                                              -> 
    keyframe 6                                                              -> ugly circle at a low angle

(the other keyframes work with all these settings two but their tracings are less nice, mainly some tweaking back and forth and flatter ellipsoids)

with these settings:
pen_tip_init_pos = data.xpos[object_id] + np.array([0, 0, -object_radius]) 
start_offset = np.array([0, r, 0])
path_center = pen_tip_init_pos + start_offset

