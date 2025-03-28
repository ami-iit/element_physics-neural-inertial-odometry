from os import path as osp
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualizer as vis
from progressbar import progressbar as pbar

# config data seqs
use_tlio_data = False
use_tlio_ifeel = True
data_dir_baf = "./local_data/ifeel_baf"
data_dir_tlio = "./local_data/tlio_golden"

if use_tlio_data:
    with open(osp.join(data_dir_tlio, "all_ids.txt")) as f:
        data_list = np.array([
            s.strip() for s in f.readlines() if len(s.strip()) > 0
        ])
    print(f"num of data sequences: {len(data_list)}")

    # take one example seq to check
    example_idx = 200
    seq_idx = data_list[example_idx]
    with open(osp.join(data_dir_tlio, seq_idx, 
                    "imu0_resampled_description.json"), 'r') as f:
        d = json.load(f)
    print(d)
    col_names = d['columns_name(width)']
    n_cols = len(col_names)
    seq_duration_hrs = 1e-6 * (d['t_end_us'] - d['t_start_us'])
    n_rows = d['num_rows']
    freq = d['approximate_frequency_hz']

    file_name = osp.join(data_dir_tlio, seq_idx, "imu0_resampled.npy")
    seq = np.load(file_name)
    print(seq.shape)

    seq_t = seq[:, :1]
    seq_quat = seq[:, 7:11] # xyzw
    r = Rot.from_quat(seq_quat)
    seq_R = r.as_matrix()
    seq_p = seq[:, 11:14]
elif use_tlio_ifeel:
    # load tlio data
    with open(osp.join(data_dir_tlio, "all_ids.txt")) as f:
        data_list = np.array([
            s.strip() for s in f.readlines() if len(s.strip()) > 0
        ])
    example_idx = 200
    seq_idx = data_list[example_idx]
    with open(osp.join(data_dir_tlio, seq_idx, 
                    "imu0_resampled_description.json"), 'r') as f:
        d = json.load(f)
    file_name = osp.join(data_dir_tlio, seq_idx, "imu0_resampled.npy")
    seq = np.load(file_name)

    seq_quat = seq[:, 7:11] # xyzw
    r = Rot.from_quat(seq_quat)
    seq_R = r.as_matrix()
    seq_p = seq[:, 11:14]
    H_tlio = np.zeros((4, 4))
    H_tlio[:3, :3] = seq_R[0]
    H_tlio[:3, 3] = seq_p[0]

    # load ifeel baf data
    file_name = osp.join(data_dir_baf, "base_data.npy")
    bases = np.load(file_name, allow_pickle=True)
    seq_euler = bases["forward_walking"]["base_orientation"].reshape((-1, 3))
    seq_R_baf = Rot.from_euler('xyz', seq_euler).as_matrix()
    seq_p_baf = bases["forward_walking"]["base_position"].reshape((-1, 3))
    H_baf = np.zeros((4, 4))
    H_baf[:3, :3] = seq_R_baf[0]
    H_baf[:3, 3] = seq_p_baf[0]

    # compute the transformation matrix from baf to tlio
    H_trans = H_tlio.T @ H_baf

else:
    # read the ifeel baf data
    file_name = osp.join(data_dir_baf, "base_data.npy")
    bases = np.load(file_name, allow_pickle=True)
    seq_euler = bases["forward_walking"]["base_orientation"].reshape((-1, 3))
    seq_R = Rot.from_euler('xyz', seq_euler).as_matrix()
    seq_p = bases["forward_walking"]["base_position"].reshape((-1, 3))


print(f"Groud truth pose info: \n"
      f"Orientation shape: {seq_R.shape} \n"
      f"Position shape: {seq_p.shape}")

# visualize the 3D trajectory with retrieved pose
Hb1 = np.matrix([
            [1.0, 0., 0., 0.],
            [0., 1.0, 0., 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]
            ])
Hb2 = np.matrix([
            [1.0, 0., 0., 0.],
            [0., 1.0, 0., 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]
            ])
urdf_path = "./urdfs/humanSubject01_66dof.urdf"
print(f"preparing the visualizer...")
visualizer = vis.HumanURDFVisualizer(path=urdf_path, model_names=["human66dof_p1", "human66dof_p2"])
visualizer.load_model(colors=[(0.2, 0.2, 0.2, 0.9), (1.0, 0.2, 0.2, 0.9)])
print(f"visualizer initialized")

transR = np.matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(seq_p[:, 0], seq_p[:, 1], seq_p[:, 2], 'k-', label="Trajectory")
plt.show()

for i in pbar(range(seq_p.shape[0])):
    jpos = np.zeros((66, ))
    Hb1[:3, :3] = seq_R[i, :].reshape((3, 3))
    Hb1[:3, 3] = seq_p[i, :].reshape((3, 1))
    Hb1 = Hb1 @ H_trans
    visualizer.update([jpos, jpos], [Hb1, Hb1], False, None)
    visualizer.run()


