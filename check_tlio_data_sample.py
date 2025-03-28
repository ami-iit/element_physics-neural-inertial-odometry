import os
from os import path as osp
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# config data seqs
data_dir = "./local_data/tlio_golden"
with open(osp.join(data_dir, "all_ids.txt")) as f:
    data_list = np.array([
        s.strip() for s in f.readlines() if len(s.strip()) > 0
    ])
print(f"num of data sequences: {len(data_list)}")

# take one example seq to check
example_idx = 10
seq_idx = data_list[example_idx]
with open(osp.join(data_dir, seq_idx, 
                   "imu0_resampled_description.json"), 'r') as f:
    d = json.load(f)
print(d)
col_names = d['columns_name(width)']
n_cols = len(col_names)
seq_duration_hrs = 1e-6 * (d['t_end_us'] - d['t_start_us'])
n_rows = d['num_rows']
freq = d['approximate_frequency_hz']

file_name = osp.join(data_dir, seq_idx, "imu0_resampled.npy")
seq = np.load(file_name)
print(seq.shape)

seq_t = seq[:, :1]
seq_quat = seq[:, 7:11] # xyzw
r = Rot.from_quat(seq_quat)
seq_R = r.as_matrix()
seq_p = seq[:, 11:14]
print(f"Groud truth pose info: \n"
      f"Orientation shape: {seq_R.shape} \n"
      f"Position shape: {seq_p.shape}")

# visualize the 3D trajectory with retrieved pose
# ideally we could use an agent like a human or a robot
# for now we simply visualize the trajectory in 3D
def plot_pose(ax, position, rotation_matrix, scale=0.1):
    """Plot a coordinate frame at a given position with rotation"""
    origin = np.array(position)
    x_axis = origin + scale * rotation_matrix[:, 0]
    y_axis = origin + scale * rotation_matrix[:, 1]
    z_axis = origin + scale * rotation_matrix[:, 2]

    ax.quiver(*origin, *(x_axis - origin), color='r', label="X-axis")
    ax.quiver(*origin, *(y_axis - origin), color='g', label="Y-axis")
    ax.quiver(*origin, *(z_axis - origin), color='b', label="Z-axis")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(seq_p[:, 0], seq_p[:, 1], seq_p[:, 2], 'k-', label="Trajectory")
plt.show()
# Plot poses
""" for pos, rot in zip(seq_p, seq_R):
    plot_pose(ax, pos, rot)

ax.set_xlim([-1, 3])
ax.set_ylim([-1, 3])
ax.set_zlim([-1, 3])
ax.set_title("3D Trajectory with Poses")
ax.legend()
plt.show() """