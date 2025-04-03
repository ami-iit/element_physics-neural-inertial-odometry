r"""
References:
    - https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_buffer.py
    - https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_calib.py
"""
import os.path as osp
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from utils.logging import logging

#----------------IMU CALIBRATOR CLASS----------------#
class IMUCalibrator:
    def __init__(self):
        r"""
        Store and manage the calibration parameters for an IMU. It includes transformation amtrices, 
        scaling fators, and biases for acclerometer and gyroscope calibration.
        """
        self.accScaleInv = np.eye(3) # inverse scale matrix for accelerometer
        self.gyroScaleInv = np.eye(3) 
        self.accBias = np.zeros((3, 1)) # accelerometer bias
        self.gyroBias = np.zeros((3, 1))
        self.gyroGravitySensitivity = np.zeros((3, 3)) # gyro g-sensitivity matrix
    
    @classmethod
    def get_calib_from_offline_dataset(cls, args):
        r"""
        Initialize an instance of the class using calibration dara stored in a JSON file.
        Args:
            - cls: the class itself, allowing the method to create and return an instance
            - args: containing paths and other settings
        """
        instance = cls()
        # log the path to the calibration file
        logging.info(
            "loading offline calibration data from"
            + osp.join(args.io.root_dir, args.io.dataset_number, "calibration.json")
        )
        with open(osp.join(args.io.root_dir, args.io.dataset_number, "calibration.json"), 'r') as f:
            calib_json = json.load(f)
        
        instance.accBias = np.array(calib_json["Accelerometer"]["Bias"]["Offset"])[:, None]
        instance.gyroBias = np.array(calib_json["Gyroscope"]["Bias"]["Offset"])[:, None]
        instance.accScaleInv = np.linalg.inv(np.array(
            calib_json["Accelerometer"]["Model"]["RectificationMatrix"]
        ))
        instance.gyroScaleInv = np.linalg.inv(np.array(
            calib_json["Gyroscope"]["Model"]["RectificationMatrix"]
        ))
        return instance

    def _validate_imu_data(self, acc, gyro):
        assert acc.shape == gyro.shape # acc and gyro must have the same shape (3, N)
        assert acc.shape[0] == 3
        assert acc.ndim == 2

    def calibrate_raw_imu(self, acc, gyro):
        r"""Apply calibration to raw accelerometer and gyroscope data."""
        self._validate_imu_data(acc, gyro)

        acc_calib = self.accScaleInv @ acc - self.accBias
        gyro_calib = self.gyroScaleInv @ gyro - self.gyroGravitySensitivity @ acc - self.gyroBias

        return acc_calib, gyro_calib
    
    def scale_raw_imu(self, acc, gyro):
        r"""Not applying bias correction."""
        self._validate_imu_data(acc, gyro)

        acc_calib = self.accScaleInv @ acc
        gyro_calib = self.gyroScaleInv @ gyro - self.gyroGravitySensitivity @ acc
        return acc_calib, gyro_calib


#----------------IMU BUFFER CLASS----------------#
class IMUBuffer:
    def __init__(self):
        """
        Store and interpolate IMU data, specifically acc and gyro over time t_us.
        Support adding new data with interpolation between timestamps.
        """
        self.nn_t_us = np.array([], dtype=int)
        self.nn_acc = np.array([])
        self.nn_gyro = np.array([])
    
    def cal_interpolated_imu_data(
            self, last_t_us, t_us, last_gyro, gyro, last_acc, acc, requested_interpolated_t
        ):
        """
        Take two IMU readings at last_t_us and t_us,
        and interpolate them at specified timestamp requested_interpolated_t.
        """
        # ensure the timestamps are integers
        assert isinstance(last_t_us, int) and isinstance(t_us, int)

        # if last_t_us is negative, no previous data exists, interpolation not possible
        # directly assign the current acc and gyro values
        if last_t_us < 0:
            acc_interpolated, gyro_interpolated = acc.T, gyro.T
        else:
            if np.any(requested_interpolated_t < last_t_us) or np.any(requested_interpolated_t > t_us):
                raise ValueError(f"Requested interpolation times must be between {last_t_us} and {t_us}")
            # interpolate acc
            acc_interpolated = interp1d(
                np.array([last_t_us, t_us], dtype=np.uint64).T,
                np.concatenate([last_acc.T, acc.T]),
                axis=0
            )(requested_interpolated_t)
            # interpolate gyro
            gyro_interpolated = interp1d(
                np.array([last_t_us, t_us], dtype=np.uint64).T,
                np.concatenate([last_gyro.T, gyro.T]),
                axis=0
            )(requested_interpolated_t)
        # store the interpolated data
        self._add_interpolated_data(requested_interpolated_t, acc_interpolated, gyro_interpolated)

    def _add_interpolated_data(self, t_us, acc, gyro):
        """Add interpolated imu data to the buffer."""
        assert isinstance(t_us, int)
        if self.net_t_us and t_us <= self.net_t_us[-1]:  # Ensure increasing order
            raise ValueError(f"Timestamp {t_us} is not greater than the last timestamp {self.net_t_us[-1]}")
    
        self.nn_t_us = np.append(self.nn_t_us, t_us)
        self.nn_acc = np.append(self.nn_acc, acc)
        self.nn_gyro = np.append(self.nn_gyro, gyro)

    def get_last_k_nn_inputs(self, k):
        """Get last k steps of network input data."""
        last_k_acc = self.nn_acc[-k:, :]
        last_k_gyro = self.nn_gyro[-k:, :]
        last_k_timestamps = self.nn_t_us[-k:, :]
        return last_k_acc, last_k_gyro, last_k_timestamps
    
    def get_whole_nn_inputs(self, t0_us, t1_us):
        """Get the whole sequence of network input data from t_begin to t_end."""
        assert isinstance(t0_us, int) and isinstance(t1_us, int)
        id0 = np.where(self.nn_t_us == t0_us)[0][0]
        id1 = np.where(self.nn_t_us == t1_us)[0][0]
        _acc = self.nn_acc[id0:id1+1, :]
        _gyro = self.nn_gyro[id0:id1+1, :]
        _t_us = self.nn_t_us[id0:id1+1, :]
        return _acc, _gyro, _t_us
    
    def erase_data_before_t0(self, t0_us):
        """Erase the network input data before the timestamp t_begin."""
        assert isinstance(t0_us, int)
        id0 = np.where(self.nn_t_us == t0_us)[0][0]
        self.nn_acc = self.nn_acc[id0:, :]
        self.nn_gyro = self.nn_gyro[id0:, :]
        self.nn_t_us = self.nn_t_us[id0:, :]

    def get_nn_input_data_size(self):
        return self.nn_t_us.shape[0]
    
    def print_info(self, query_t_us):
        if not self.nn_t_us:
            print(f"Network input buffer is empty!")
            return
        print(f"t_begin in the buffer: {self.nn_t_us[0]}")
        print(f"t_end in the buffer: {self.nn_t_us[-1]}")
        print(f"Queried timestamp: {query_t_us}")
        print(f"The whole timestamps sequence: {self.nn_t_us}")



