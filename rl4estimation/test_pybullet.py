import pybullet as p
import time
import pybullet_data
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
from gymnasium import spaces

# config
joints_66dof = [
    "jL5S1_rotx" , "jRightHip_rotx" , "jLeftHip_rotx" , "jLeftHip_roty" , "jLeftHip_rotz" , "jLeftKnee_rotx" , "jLeftKnee_roty" ,
    "jLeftKnee_rotz" , "jLeftAnkle_rotx" , "jLeftAnkle_roty" , "jLeftAnkle_rotz" , "jLeftBallFoot_rotx" , "jLeftBallFoot_roty" ,
    "jLeftBallFoot_rotz" , "jRightHip_roty" , "jRightHip_rotz" , "jRightKnee_rotx" , "jRightKnee_roty" , "jRightKnee_rotz" ,
    "jRightAnkle_rotx" , "jRightAnkle_roty" , "jRightAnkle_rotz" , "jRightBallFoot_rotx" , "jRightBallFoot_roty" , "jRightBallFoot_rotz" ,
    "jL5S1_roty" , "jL5S1_rotz" , "jL4L3_rotx" , "jL4L3_roty" , "jL4L3_rotz" , "jL1T12_rotx" , "jL1T12_roty" , "jL1T12_rotz" ,
    "jT9T8_rotx" , "jT9T8_roty" , "jT9T8_rotz" , "jLeftC7Shoulder_rotx" , "jT1C7_rotx" , "jRightC7Shoulder_rotx" , "jRightC7Shoulder_roty" ,
    "jRightC7Shoulder_rotz" , "jRightShoulder_rotx" , "jRightShoulder_roty" , "jRightShoulder_rotz" , "jRightElbow_rotx" , "jRightElbow_roty" ,
    "jRightElbow_rotz" , "jRightWrist_rotx" , "jRightWrist_roty" , "jRightWrist_rotz" , "jT1C7_roty" , "jT1C7_rotz" , "jC1Head_rotx" ,
    "jC1Head_roty" , "jC1Head_rotz" , "jLeftC7Shoulder_roty" , "jLeftC7Shoulder_rotz" , "jLeftShoulder_rotx" , "jLeftShoulder_roty" ,
    "jLeftShoulder_rotz" , "jLeftElbow_rotx" , "jLeftElbow_roty" , "jLeftElbow_rotz" , "jLeftWrist_rotx" , "jLeftWrist_roty" ,
    "jLeftWrist_rotz"
]

# start the physics engine
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# improve initial camera view and remove GUI clatter
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# --- INITIAL CAMERA PARAMETERS (Adjust as needed) ---
# These will be used to maintain a consistent viewing angle relative to the world,
# while the target follows the human.
g_camera_distance = 2.5
g_camera_yaw = 60      # Angle around Z-axis (world frame)
g_camera_pitch = -25   # Angle above/below XY plane
g_camera_target_z_offset = 0.6 # Look slightly above the base_pos (e.g., towards torso)

p.resetDebugVisualizerCamera(cameraDistance=g_camera_distance, 
                             cameraYaw=g_camera_yaw,
                             cameraPitch=g_camera_pitch,
                             cameraTargetPosition=[0, 0, g_camera_target_z_offset]
                         )

# load a custom human URDF model
urdf_path = "../urdfs/humanSubject01_66dof.urdf"
try:
    humanId = p.loadURDF(
        urdf_path, 
        [0, 0, 1], 
        p.getQuaternionFromEuler([0, 0, 0]), 
        useFixedBase=False
    )
except p.error as e:
    print(f"Error loading URDF '{urdf_path}': {e}")
    print(f"Please check the path and ensure the URDF file is valid.")
    p.disconnect()
    exit(1)

colors = {
    "trunk": [0.1, 0.8, 0.5, 1.0],    # Bright Emerald Green
    "arms":  [0.0, 0.6, 0.95, 1.0],   # Vivid Azure Blue
    "legs":  [0.4, 0.9, 0.7, 1.0],    # Lighter Teal / Seafoam Green
    "head":  [1.0, 0.3, 0.4, 1.0]     # Bright Coral Pink
}

link_name_keywords_by_region = {
    "trunk": ["L5S1", "L4L3", "L1T12", "T9T8", "T1C7", "Pelvis", "Torso", "Spine"], # "Pelvis" is often the base link name
    "arms":  ["Shoulder", "Elbow", "Wrist", "Arm", "Hand", "C7Shoulder"], # Covers Left and Right
    "legs":  ["Hip", "Knee", "Ankle", "BallFoot", "Leg"], # Covers Left and Right
    "head":  ["Head", "Neck", "C1Head"]
}

print("Applying custom colors to regions...")
# --- Change color for the BASE link ---
base_link_name = p.getBodyInfo(humanId, physicsClientId=physicsClient)[0].decode('utf-8').lower()
print(f"Base link name: '{base_link_name}'")
base_colored = False
for region, keywords in link_name_keywords_by_region.items():
    for keyword in keywords:
        if keyword.lower() in base_link_name:
            p.changeVisualShape(humanId, -1, rgbaColor=colors[region])
            print(f"  Colored base link ('{base_link_name}') as {region} with color {colors[region]}")
            base_colored = True
            break
    if base_colored:
        break
if not base_colored:
    print(f"  Base link ('{base_link_name}') not matched to any region, retaining original color.")


# --- Change color for ALL OTHER links ---
num_joints = p.getNumJoints(humanId)
for i in range(num_joints):
    joint_info = p.getJointInfo(humanId, i)
    link_name_urdf = joint_info[12].decode('utf-8').lower() # Name of the link that is the CHILD of this joint
    
    # If link_name is empty, skip (can happen for fixed joints not defining a new link visually sometimes)
    if not link_name_urdf:
        continue

    # print(f"Checking link: '{link_name_urdf}' (child of joint '{joint_info[1].decode('utf-8')}')")
    link_colored = False
    for region, keywords in link_name_keywords_by_region.items():
        for keyword in keywords:
            if keyword.lower() in link_name_urdf:
                p.changeVisualShape(objectUniqueId=humanId,
                                    linkIndex=i, # The linkIndex is the same as the jointIndex for non-fixed joints
                                    rgbaColor=colors[region])
                # print(f"  Colored link '{link_name_urdf}' (index {i}) as {region} with color {colors[region]}")
                link_colored = True
                break # Found a region for this link
        if link_colored:
            break # Move to the next link
    # if not link_colored:
    #     print(f"  Link '{link_name_urdf}' (index {i}) not matched, retaining original color.")
# -------------------------------------------------

# identify the actuated joint indices
num_joints = p.getNumJoints(humanId)
actuated_joint_indices = {}
print(f"Human '{p.getBodyInfo(humanId)[1].decode('utf-8')}' has {num_joints} joints:")
for i in range(num_joints):
    joint_info = p.getJointInfo(humanId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]
    print(f"  Joint {i}: {joint_name} (type: {joint_type})")
    if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
        actuated_joint_indices[joint_name] = i

# Verify that all joints in joints_66dof are found in the URDF
for j_name_config in joints_66dof:
    if j_name_config not in actuated_joint_indices:
        print(f"  WARNING: Joint '{j_name_config}' from 'joints_66dof' list WAS NOT FOUND among the actuatable joints in the URDF. Data for this joint cannot be applied.")

print(f"\nActuated joint indices: {actuated_joint_indices}, total number: {len(actuated_joint_indices)}")
num_actuated_joints = len(actuated_joint_indices)

# load custom data
task_name = "forward_walking_clapping_hands"
data_dir = f"../local_data/xsens/cheng/{task_name}/IK_data/processed"
pb = np.load(f"{data_dir}/pb_{task_name}.npy")
rb = np.load(f"{data_dir}/rb_{task_name}.npy")
s = np.load(f"{data_dir}/s_{task_name}.npy")
sdot = np.load(f"{data_dir}/sdot_{task_name}.npy")
assert pb.shape[0] == rb.shape[0] == s.shape[0] == sdot.shape[0], "Data shape mismatch!"
data_points = pb.shape[0]

        
# 3. Prepare your Kinematic Data Sequence (Example Data)
# Each item: {"base_pos": [x,y,z], "base_ori_quat": [qx,qy,qz,qw], "joint_angles": [j1,j2,...]}
# The order in "joint_angles" should correspond to the order in "actuated_joint_indices"
kinematic_data_sequence = []
for t in range(data_points): 
    base_pos = pb[t, :] # Extract base position from pb data
    base_as_R = rb[t, :].reshape((3, 3))
    base_as_q = R.from_matrix(base_as_R).as_quat() # Convert rotation matrix to quaternion
    joint_angles = s[t, :]
    joint_velocities = sdot[t, :]

    kinematic_data_sequence.append({
        "base_pos": base_pos,
        "base_ori_quat": base_as_q,
        "base_ori_matrix": base_as_R,
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities
    })

# --- Trajectory Visualization Variables ---
base_trajectory_points = []
trajectory_color = [1.0, 0.2, 0.2]  # A nice red
trajectory_line_width = 2.5

# Orientation Frame settings
ORIENTATION_FRAME_LENGTH = 0.1  # Length of the axes
ORIENTATION_FRAME_LINE_WIDTH = 1.5
ORIENTATION_X_COLOR = [1, 0, 0]  # Red for X
ORIENTATION_Y_COLOR = [0, 1, 0]  # Green for Y
ORIENTATION_Z_COLOR = [0, 0, 1]  # Blue for Z
DRAW_FRAME_EVERY_N_STEPS = 15  # Draw a frame every Nth step to avoid clutter

# 4. Simulation/Playback Loop
playback_fps = 60.0 # Desired playback speed
for frame_idx, frame_data in enumerate(kinematic_data_sequence):
    current_base_pos = frame_data["base_pos"]
    current_base_ori = frame_data["base_ori_quat"]
    current_base_ori_matrix = frame_data["base_ori_matrix"]
    current_joint_angles = frame_data["joint_angles"]
    current_joint_velocities = frame_data["joint_velocities"]

    # --- Set Base State ---
    p.resetBasePositionAndOrientation(humanId, current_base_pos, current_base_ori)

    # --- Set Joint States ---
    if len(current_joint_angles) != num_actuated_joints:
        print(f"Warning: Data has {len(current_joint_angles)} joint angles, but model has {num_actuated_joints} actuated joints. Skipping joint update.")
    else:
        for joint_name, joint_index in actuated_joint_indices.items():
            target_joint_index = joints_66dof.index(joint_name)
            target_angle = current_joint_angles[target_joint_index]
            target_velocity = current_joint_velocities[target_joint_index]
            # You can also set targetVelocity and forces if needed, but for pure kinematic replay,
            # just setting the position is enough.
            p.resetJointState(bodyUniqueId=humanId,
                              jointIndex=joint_index,
                              targetValue=target_angle,
                              targetVelocity=target_velocity) # Set velocity to 0 for pure kinematic state

    # --- TRAJECTORY VISUALIZATION of the Base ---
    # `current_base_pos` is the position of the base for this frame
    if len(base_trajectory_points) > 0:
        last_pos = base_trajectory_points[-1]
        p.addUserDebugLine(
            lineFromXYZ=last_pos,
            lineToXYZ=current_base_pos,
            lineColorRGB=trajectory_color,
            lineWidth=trajectory_line_width,
            lifeTime=0  # 0 means the line is persistent
        )
    base_trajectory_points.append(list(current_base_pos)) # Store a copy (list() ensures it's a copy)

    # --- ORIENTATION FRAME VISUALIZATION ---
    if frame_idx % DRAW_FRAME_EVERY_N_STEPS == 0:
        # The columns of the rotation matrix are the new X, Y, Z axes in world coordinates
        x_axis_world = current_base_ori_matrix[:, 0]
        y_axis_world = current_base_ori_matrix[:, 1]
        z_axis_world = current_base_ori_matrix[:, 2]

        # Start point for all axes is the current base position
        origin = current_base_pos

        # End points for each axis
        x_end = origin + x_axis_world * ORIENTATION_FRAME_LENGTH
        y_end = origin + y_axis_world * ORIENTATION_FRAME_LENGTH
        z_end = origin + z_axis_world * ORIENTATION_FRAME_LENGTH

        # Draw the X, Y, Z axes
        p.addUserDebugLine(origin, x_end, ORIENTATION_X_COLOR, ORIENTATION_FRAME_LINE_WIDTH, lifeTime=0)
        p.addUserDebugLine(origin, y_end, ORIENTATION_Y_COLOR, ORIENTATION_FRAME_LINE_WIDTH, lifeTime=0)
        p.addUserDebugLine(origin, z_end, ORIENTATION_Z_COLOR, ORIENTATION_FRAME_LINE_WIDTH, lifeTime=0)
    # --- END ORIENTATION FRAME VISUALIZATION ---

    # --- CAMERA FOLLOW LOGIC ---
    # The camera will target a point slightly above the human's base position.
    # The distance, yaw, and pitch relative to this target point will remain fixed.
    camera_target_actual_position = [
        current_base_pos[0],
        current_base_pos[1],
        current_base_pos[2] + g_camera_target_z_offset # Target a bit higher than the base
    ]

    p.resetDebugVisualizerCamera(
        cameraDistance=g_camera_distance,
        cameraYaw=g_camera_yaw,           # Use the globally defined yaw
        cameraPitch=g_camera_pitch,         # Use the globally defined pitch
        cameraTargetPosition=camera_target_actual_position
    )
    # --- END CAMERA FOLLOW LOGIC ---

    # Step the simulation to update visuals and internal state
    p.stepSimulation()
    time.sleep(1.0 / playback_fps)

num_segments_plotted = len(base_trajectory_points) -1 if len(base_trajectory_points) > 0 else 0
print(f"Playback finished. Plotted {num_segments_plotted} trajectory segments.")

# Keep the simulation window open for viewing
print("Simulation window will remain open. Close it manually or press Ctrl+C in the terminal to exit.")
try:
    while p.isConnected():
        p.getKeyboardEvents() # Process keyboard events to allow closing, keep GUI responsive
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Exiting due to Ctrl+C.")
finally:
    if p.isConnected():
        p.disconnect()
