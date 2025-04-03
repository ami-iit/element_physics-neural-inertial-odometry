from os import path as osp
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.io.matlab import mat_struct
from scipy.io.matlab import MatReadError
import h5py
from pathlib import Path

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

    def load(self, root_dir, load_all=True, filename=None):
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
            for file in os.listdir(root_dir):
                mat_data = self._open(root_dir, file)
                keys = sorted(mat_data.keys())
                full_data = self._check_keys(mat_data[keys[1]])
                self.traj[Path(file).stem[9:]] = full_data
        else:
            mat_data = self._open(root_dir, filename)
            keys = sorted(mat_data.keys())
            single_data = self._check_keys(mat_data[keys[1]])
            self.traj[Path(file).stem[9:]] = single_data
        print(f"Full dicts: {self.traj.keys()}")

    
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
        print(f"pose sequence length: {self.pose_ts.shape[0]}")
    
    def get_step_traj(self, idx):
        return
    

    def prepare_meas_and_cov(self):
        return
    

if __name__ == '__main__':
    print(f"Test imu data streaming...")
    mat_dir = "../local_data/ifeel_baf/raw"
    task = "forward_walking"
    
    dp = DataStreamer()
    dp.load(mat_dir)
    dp.prepare_imu_seq(task)
    dp.prepare_pose_seq(task)

