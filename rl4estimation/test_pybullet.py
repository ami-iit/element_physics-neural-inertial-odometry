import pybullet as p
import time
import pybullet_data
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

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


# load a custom human URDF model
humanId = p.loadURDF(
    "../urdfs/humanSubject01_66dof.urdf", 
    [0, 0, 1], 
    p.getQuaternionFromEuler([0, 0, 0]), 
    useFixedBase=False
)


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

print(f"\nActuated joint indices: {actuated_joint_indices}, total number: {len(actuated_joint_indices)}")
num_actuated_joints = len(actuated_joint_indices)

# load custom data
data_dir = "../xsens_data/cheng/forward_walking_normal/IK_data/processed"
pb = np.load(f"{data_dir}/pb_forward_walking_normal.npy")
rb = np.load(f"{data_dir}/rb_forward_walking_normal.npy")
s = np.load(f"{data_dir}/s_forward_walking_normal.npy")
sdot = np.load(f"{data_dir}/sdot_forward_walking_normal.npy")
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
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities
    })

# 4. Simulation/Playback Loop
playback_fps = 60.0 # Desired playback speed
for frame_data in kinematic_data_sequence:
    current_base_pos = frame_data["base_pos"]
    current_base_ori = frame_data["base_ori_quat"]
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

    # Step the simulation to update visuals and internal state
    p.stepSimulation()
    time.sleep(1.0 / playback_fps)

print("Playback finished.")
time.sleep(5) # Keep window open for a bit
p.disconnect()

