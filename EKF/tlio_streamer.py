r"""Simply stream the TLIO data for testing."""
from os import path as osp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.logging import logging
import utils.math_utils as maths

class TLIOStreamer:
    def __init__(self):
        # imu data from imu_samples_0.csv (raw)
        self.ts_all = None
        self.gyro_all = None
        self.acc_all = None
        self.ds_size = None

        # vio data from imu0_resampled.npy (calibrated)
        self.vio_ts = None
        self.vio_p = None
        self.vio_v = None
        self.vio_euler = None
        self.vio_R = None
        self.vio_quat = None
        self.vio_ba = None
        self.vio_bg = None

    def load_imu_all(self, args):
        """Load IMU timestamps, acc and gyro from imu_samples_0.csv file."""
        logging.info(
            f"Loading imu data from " 
            + osp.join(args.io.root_dir, args.io.dataset_number, "imu_samples_0.csv")
        )
        imu_data = pd.read_csv(osp.join(
            args.io.root_dir, 
            args.io.dataset_number, 
            "imu_samples_0.csv"
        ))
        imu_ts = np.copy(imu_data.iloc[:, 0]) * 1e-3 # original timestamps recorded as millisecond
        imu_gyro = np.copy(imu_data.iloc[:, 2:5])
        imu_acc = np.copy(imu_data.iloc[:, 5:8])
        if args.io.start_from_ts is not None:
            idx_start = np.where(imu_ts >= args.io.start_from_ts)[0][0]
        else:
            idx_start = 50

        self.imu_ts = imu_ts[idx_start:]
        self.imu_acc = imu_acc[idx_start:]
        self.imu_gyro = imu_gyro[idx_start:]

        self.ds_size = self.imu_ts.shape[0] # number of data points in the loaded sequence
        self.init_ts = self.imu_ts[0] # the first timestamp recorded in the loaded sequence

    def load_vio_all(self, args):
        """Load VIO timestamps, position, orientation and velocity from imu0_resampled.npy file."""
        logging.info(
            f"Loading vio data from " 
            + osp.join(args.io.root_dir, args.io.dataset_number, "imu0_resampled.npy")
        )
        vio_data = np.load(osp.join(args.io.root_dir, args.io.dataset_number, "imu0_resampled.npy"))
        self.vio_ts = vio_data[:, 0] * 1e-6 # original timestamps recorded as microsecond
        self.vio_quat = vio_data[:, -10:-6] # original orientation as quaternion
        self.vio_p = vio_data[:, -6:-3]
        self.vio_v = vio_data[:, -3:]

        self.vio_R = Rotation.from_quat(self.vio_quat).as_matrix() # orientation as rotation matrix
        self.vio_euler = Rotation.from_quat(self.vio_quat).as_euler("xyz", degrees=True) # orientation as euler angles

    def get_datapoint(self, idx):
        """Return a single datapoint form the imu sequence."""
        ts = self.imu_ts[idx] * 1e-6 # sec
        acc = self.imu_acc[idx, :].reshape((3, 1))
        gyro = self.imu_gyro[idx, :].reshape((3, 1))
        return ts, acc, gyro
    
    def get_meas_from_vio(self, ts_oldest_state, ts_end):
        """
        Return a simulated measurement (position displacement) from the VIO data.
        Args:
            - ts_oldest_state; the timestamp of the oldest state in the filter.
            - ts_end: the timestamp at which we want to compute a simulated displacement.
        Return:
            - meas: a simulated displacement measurement expressed in a gravity-aligned frame.
            - meas_cov: a covariance matrix representing the uncertainty in meas.
        """
        # find two closest timestamps around ts_oldest_state
        # print(f"left idx: {np.array(np.where(self.vio_ts < ts_oldest_state))[0]}")
        # print(f"right idx: {np.array(np.where(self.vio_ts > ts_oldest_state))[0]}")
        # print(f"vio ts: {self.vio_ts}")
        ts_oldest_state_sec = ts_oldest_state * 1e-6
        ts_end_sec = ts_end * 1e-6
        idx_left = np.array(np.where(self.vio_ts < ts_oldest_state_sec))[0, -1]
        idx_right = np.array(np.where(self.vio_ts > ts_oldest_state_sec))[0, 0]
        interp_vio_ts = self.vio_ts[idx_left: idx_right+1]
        # extract the corresponding euler angles from VIO data
        interp_vio_euler = self.vio_euler[idx_left: idx_right+1, :]

        # get the interpolated orientatin at the timestamp ts_oldest_state (namely the first element in the imu buffer)
        vio_eulers_unwrapped = maths.unwrap_rpy(interp_vio_euler)
        vio_euler_unwrapped = interp1d(interp_vio_ts, vio_eulers_unwrapped, axis=0)(ts_oldest_state_sec)
        #print(f"vio euler uw shape: {vio_euler_unwrapped.shape}")
        vio_euler = np.deg2rad(maths.wrap_rpy(vio_euler_unwrapped))

        # compute simulated displacement
        ts_interp = np.array([ts_oldest_state_sec, ts_end_sec]) # time period until timestamp ts_end
        vio_interp_p = interp1d(self.vio_ts, self.vio_p, axis=0)(ts_interp)
        vio_meas_displacement = vio_interp_p[1] - vio_interp_p[0]

        # compute the correpsonding covariance (manually defined small values)
        vio_meas_cov = np.diag(np.array([1e-2, 1e-2, 1e-2]))

        # rotate measurement displacement to a gravity-aligned frame
        Ri_z = Rotation.from_euler("z", vio_euler[2]).as_matrix()
        vio_meas = Ri_z.dot(vio_meas_displacement.reshape((3, 1)))

        return vio_meas, vio_meas_cov



