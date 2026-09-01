# dexterity_from_jacobian
        
## How to apply controller to new robot hand models

### making `scene_pen.xml`
Steps to make a scene in which the hand holds a pen, defined by a keyframe. The keyframe can be activated in the MuJoCo GUI with the `Key` slider in the upper right.

1. prepare your high-quality MJCF robot hand model which includes position actuators
1. make a tilted version of the model where the hand pose is set to a "writing" pose
    - e.g. `wuji_hand_2/right_tilted.xml`
    - unfortunately the `<attach>` feature cannot be used to load the model for our case, since it had some trouble with setting `qpos` keyframe fields for the attached model
1. create a new MJCF file which includes the tilted robot hand model and the pen body
    - e.g. `wuji_hand_2/scene_pen.xml`
    - uses the `<include>` feature to add the tilted model to the scene, which doesn't have the issue with keyframes
    - you can copy the `<body name="pen" ...` from that file for the pen model
1. set good-looking "writing" hand pose manually by adjusting the **Control** sliders in the MuJoCo GUI (ignore the pen for now)
    - once you're satisfied, "Copy state" from the GUI and fill in the `ctrl` fields for your keyframe
1. move the pen into the hand- the pen can be moved kinematically by pausing, double clicking the pen, and dragging it while pressing ctrl (right drag to rotate, left drag to move)
1. adjust the pen's grip in the hand- be patient!
1. finally, "Copy Pose" again and paste both the `ctrl` and `qpod` fields to your keyframe (don't copy the `time` field, the time is used to detect resets in the code)
