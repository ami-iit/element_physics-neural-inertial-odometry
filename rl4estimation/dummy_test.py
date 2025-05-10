import numpy as np
from scipy.spatial.transform import Rotation as R
from human_base_env import OfflineHumanBaseTrackingEnv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#--------------Load and Process data-------------------
def load_and_process_data(imu_path, state_path, task, imu_features, state_features):
    """Loads data from .npy files and structures it for the environment."""
    # load base data
    state_dict = {}
    for item in state_features:
        state_dict[item] = np.load(f"{state_path}/{item}_{task}.npy")
    imu_dict = {}
    for item in imu_features:
        imu_dict[item] = np.load(f"{imu_path}/{item}_{task}.npy")


#--------------Policy Network-------------------
class SimplePolicy(nn.Module):
    r"""Define a naive policy network for debuging purposes."""
    def __init__(self, obs_dim, action_dim):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, action_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action_mean = self.fc_mean(x) # Outputting the mean of the action (delta kinematics)
        return action_mean

#--------------Main Function-------------------
if __name__ == "__main__":
    # 0. Set up paths configurations
    task_name = "forward_walking_normal"
    STATE_FEATURES = ["pb", "rb", "vb", "s", "sdot"]
    IMU_FEATURES = ["lacc", "lw"]
    imu_path = f"../local_data/xsens/cheng/{task_name}/link_data/processed"
    state_path = f"../local_data/xsens/cheng/{task_name}/IK_data/processed"
    
    # 1. TODO: Load and process data
    print(f"Loading and processing data...")
    try:
        processed_data = load_and_process_data(
            imu_path, 
            state_path, 
            task_name,
            IMU_FEATURES,
            STATE_FEATURES
        )
        if not processed_data or not processed_data[0]:
            print(f"ERROR: Data loading failed or returned empty. Check paths and loading logic")
            exit()
        print(f"Data loaded successfully. Number of frames: {processed_data[0].shape[0]}")
        print(f"Frames in first test episode: {processed_data[0].shape[0]}")
    except Exception as e:
        print(f"ERROR: Exception occurred while loading data: {e}")
        exit()

    # 2. TODO: Initialize environment
    print(f"Initializing environment...")
    try:
        env = OfflineHumanBaseTrackingEnv()
        print(f"Environment initialized successfully.")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"ERROR: Exception occurred while initializing environment: {e}")
        exit()
    
    # 3.Initialize policy network
    print(f"Initializing policy network...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    try:
        policy = SimplePolicy(obs_dim, action_dim)
        policy.eval()
        print(f"Policy network initialized with random weights.")
    except Exception as e:
        print(f"ERROR: Exception occurred while initializing policy network: {e}")
        exit()
    
    # 4. Testing loop
    num_test_episodes = 10
    max_steps_per_episode = 500

    for i_episode in range(num_test_episodes):
        print(f"\n--- Starting test episode {i_episode + 1}")
        try:
            obs, info = env.reset()
            print(f"Initial observation shape: {obs.shape}")
        except Exception as e:
            print(f"ERROR: Exception occurred while resetting environment: {e}")
            continue # skip to next episode or break

        terminated = False
        truncated = False
        total_reward = 0
        estimated_base_trajectory = []
        ground_truth_base_trajectory = []

        # store initial GT for comparison
        if "ground_truth_base" in info and info["ground_truth_base"] is not None:
            ground_truth_base_trajectory.append(info["ground_truth_base"].copy())
        elif hasattr(env, "current_base_gt") and env.current_base_gt is not None:
            ground_truth_base_trajectory.append(env.current_base_gt.copy())

        # the initial estimated state in the env is set from GT at reset
        # record that as the first point of the estimated trajectory
        if hasattr(env, "previous_estimated_base_state"):
            estimated_base_trajectory.append(env.previous_estimated_base_state.copy())
   
        for step_num in range(max_steps_per_episode):
            # convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) # Add batch dimension
            with torch.no_grad():
                # get action from policy network
                action_tensor = policy(obs_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    print(f"Error during env.step() at step {step_num+1}: {e}")
                    break # stop this episode
                obs = next_obs
                total_reward += reward

                # store trajs
                if "ground_truth_base" in info and info["ground_truth_base"] is not None:
                    ground_truth_base_trajectory.append(info["ground_truth_base"].copy())
                
                if hasattr(env, "previous_estimated_base_state"):
                    estimated_base_trajectory.append(env.previous_estimated_base_state.copy())

                if terminated or truncated:
                    print(f"Episode terminated or truncated at step {step_num+1}.")
                    break
        print(f"Episode {i_episode + 1} finished. Total reward: {total_reward:.4f}")
        if estimated_base_trajectory:
            print(f"Final estimated base position: {estimated_base_trajectory[-1][:3]}")
        
        if ground_truth_base_trajectory:
            print(f"Final ground truth base position: {ground_truth_base_trajectory[-1][:3]}")

    env.close()
    print("Environment closed.")

    # Optional: Plotting (if you have matplotlib and trajectories are not empty)
    if ground_truth_base_trajectory and estimated_base_trajectory:
        gt_pos = np.array([frame[:3] for frame in ground_truth_base_trajectory])
        est_pos = np.array([frame[:3] for frame in estimated_base_trajectory])

        # Ensure same length for plotting if one is shorter due to reset logic
        min_len = min(len(gt_pos), len(est_pos))
        gt_pos = gt_pos[:min_len]
        est_pos = est_pos[:min_len]


        if min_len > 1 : # Need at least 2 points to plot
            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
            labels = ['X', 'Y', 'Z']
            for i in range(3):
                axs[i].plot(gt_pos[:, i], label=f'Ground Truth {labels[i]}')
                axs[i].plot(est_pos[:, i], label=f'Estimated {labels[i]}', linestyle='--')
                axs[i].set_ylabel(f'{labels[i]} Position')
                axs[i].legend()
            axs[-1].set_xlabel('Timestep')
            plt.suptitle('Base Position Tracking (Untrained Policy)')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("Not enough trajectory points to plot.")