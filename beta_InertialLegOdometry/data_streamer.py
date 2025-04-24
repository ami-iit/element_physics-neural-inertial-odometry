from os import path as osp
import os
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.io.matlab import mat_struct
from scipy.io.matlab import MatReadError
import h5py
from pathlib import Path
sys.path.append(osp.join(os.path.dirname(__file__), ".."))
from visualizer.visualizer import HumanURDFVisualizer
import config as cfg
import utils

class DataStreamer:
    def __init__(self):
        # assume imu and gt pose have synchronized timestamps
        self.ts = None
        self.dataset_size = None
        self.node = "node3"
        self.imu_attris = ["angVel", "linAcc", "orientation"]
        self.base_attris = [
            'base_linear_velocity',
            'base_position',
            'base_angular_velocity',
            'base_orientation'
        ]
        self.joint_attris = ['positions', 'velocities']

        # imu data
        self.acc_raw = None
        self.gyro_raw = None
        self.acc_calib = None
        self.gyro_calib = None
        # groud truth pose data
        self.seq_p = None
        self.seq_R = None
        self.seq_v = None
        # calibration data
        self.acc_bias = None
        self.gyro_bias = None
    
    def _check_keys(self, dict):
        r"""
        Chect if entries in dictionary are mat-objects. 
        If yes, _todict is called to change them to nested dictionaries.
        """
        for key in dict:
            if isinstance(dict[key], mat_struct):
                dict[key] = self._todict(dict[key])
        return dict
    
    def _todict(self, mat_obj):
        """
        A recursive function which constructs nested dictionaries from mat-objects
        """
        dict = {}
        for strg in mat_obj._fieldnames:
            elem = mat_obj.__dict__[strg]
            if isinstance(elem, mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict
            
    def _open(self, root_dir, data_file):
        r"""Open a mat file, covering both v5 and v7.3 formats."""
        if not data_file.endswith('.mat'):
            raise ValueError(f"File {data_file} is not a mat file.")
        
        try:
            with open(osp.join(root_dir, data_file), 'rb') as f:
                header = f.read(128)
            if header[:4] == b'MATL':
                mat_data = h5py.File(osp.join(root_dir, data_file), 'r')
                print(f"Open {data_file} as HDF5-based MATLAB v7.3 format file.")
            return mat_data
        except (OSError, MatReadError):
                raise ValueError(f"{data_file} not a valid .mat file or unsupported format.")

    def load(self, root_dir, task, load_all=False, filename=None):
        """
        Load imu data and pose data from the root_dir.
        imu_data:
            - timestamps: ts
            - raw acc measurements: acc_raw
            - raw gyro measurements: gyro_raw
        pose_data:
            - positions: seq_p
            - orientations: seq_R
            - velocities: seq_v
        """
        self.traj = {}
        if load_all:
            for task in os.listdir(root_dir):
                mat_data = self._open(f"{root_dir}/{task}", filename)
                keys = sorted(mat_data.keys())
                full_data = self._check_keys(mat_data[keys[1]])
                self.traj[task] = full_data
        else:
            mat_data = self._open(f"{root_dir}/{task}", filename)
            keys = sorted(mat_data.keys())
            single_data = self._check_keys(mat_data[keys[1]])
            print(f"single mat data keys: {list(single_data.keys())}")
            self.traj[task] = single_data
        print(f"Full dicts: {list(self.traj.keys())}")

    
    def prepare_imu_seq(self, task):
        """
        Extract the imu data from node 3 (attached on pelvis):
            - ts: timestamps (sec)
            - acc_raw: raw acc measurements
            - gyro_raw: raw gyro measurements
        NOTE: the timestamps stored for acc and gyro are slightly different (ms level error),
        simply take one of them (e.g., acc) as the common timestamps.
        """
        # read timestamps from accelemetor
        self.imu_ts = np.array(self.traj[task][self.node]["linAcc"]["timestamps"])
      
        # read gyro, acc and orientation from imu
        self.imus = {
            attri: np.array(self.traj[task][self.node][attri]["data"]) for attri in self.imu_attris
        }
        self.imus["ts"] = self.imu_ts

        print(f"imu (acc) timestamps: {self.imu_ts}")
        for item in self.imu_attris:
            assert self.imu_ts.shape[0] == self.imus[item].shape[0]
        print(f"imu sequence length: {self.imu_ts.shape[0]}")
    

    def prepare_pose_seq(self, task):
        """
        Extrat the base pose data as ground truth trajectory:
            - ts: timestamps (sec)
            - pb: base position
            - rb: base orientation (rpy euler angles)
            - vb: linear base velocity
            - wb: angular base velocity
        NOTE: the timestamps for each pose attribute are different. Take the one of base position as common timestamps.
        """
        self.pose_ts = np.array(self.traj[task]["human_state"]["base_position"]["timestamps"])
        self.poses = {
            attri: np.array(self.traj[task]["human_state"][attri]["data"]) for attri in self.base_attris
        }
        self.poses["ts"] = self.pose_ts

        print(f"pose (base position) timestamps: {self.pose_ts}")
        for item in self.base_attris:
            assert self.pose_ts.shape[0] == self.poses[item].shape[0]
        print(f"pose sequence length: {self.pose_ts.shape[0]}")

    def prepare_joint_seq(self, task):
        """
        Extract the joint positions and velocities.
        """
        self.joint_ts = np.array(self.traj[task]["joints_state"]["positions"]["timestamps"])
        self.jointskin = {
            attri: np.array(self.traj[task]["joints_state"][attri]["data"]) for attri in self.joint_attris
        }
        self.jointskin["ts"] = self.joint_ts

        print(f"joints kinematics timestamps: {self.joint_ts}")
        for item in self.joint_attris:
            assert self.joint_ts.shape[0] == self.jointskin[item].shape[0]
        print(f"jointskin sequence length: {self.joint_ts.shape[0]}")
    
    def get_step_traj(self, idx):
        return
    

    def prepare_meas_and_cov(self):
        return

if __name__ == '__main__':
    print(f"Test imu data streaming...")
    mat_dir = "../local_data/ifeel_baf/raw"
    task = "forward_walking"
    filename = f"Gianluca_{task}.mat"
    
    #np.set_printoptions(threshold=5000, linewidth=200)
    dp = DataStreamer()
    dp.load(root_dir=mat_dir, task=task, filename=filename)
    dp.prepare_imu_seq(task)
    dp.prepare_pose_seq(task)
    dp.prepare_joint_seq(task)

    # visualize the traj
    if False:
        urdf_path = "../urdfs/humanSubject01_66dof.urdf"
        vis = HumanURDFVisualizer(path=urdf_path, model_names=["gt1", "gt2"])
        vis.load_model(colors=[(0.2 , 0.2, 0.2, 0.6), (1.0 , 0.2, 0.2, 0.3)])
        Hb_gt1 = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                        [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
        Hb_gt2 = np.matrix([[1.0, 0., 0., 0.], [0., 1.0, 0., 0.],
                            [0., 0., 1.0, 0.], [0., 0., 0., 1.0]])
        
        dlen = dp.pose_ts.shape[0]
        for i in range(dlen):
            pb_step = dp.poses["base_position"][i].reshape((3, 1))
            rb_step = Rotation.from_euler('xyz', dp.poses["base_orientation"][i]).as_matrix()
            Hb_gt1[:3, :3] = rb_step
            Hb_gt1[:3, 3] = pb_step

            s_step = dp.jointskin["positions"][i].reshape(-1)
            sdot_step = dp.jointskin["velocities"][i].reshape(-1)
            s_new, sdot_new = utils.extend_joint_state_preds(
                s_step, sdot_step, cfg.joints_31dof, cfg.joints_66dof
            )

            vis.update(
                [s_new, s_new],
                [Hb_gt1, Hb_gt1],
                False, None
            )
            vis.run()




