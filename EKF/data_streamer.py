from os import path as osp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


class DataStreamer:
    def __init__(self):
        self.ts = None
        self.dataset_size = None
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

    def load(self, root_dir, imu_data, pose_data):
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
        return
    
    def get_imu_input(self):
        return
    

    def get_imu_calib(self):
        return
    

    def get_pose_traj(self):
        return
    
    def get_meas_and_cov(self):
        return
    

