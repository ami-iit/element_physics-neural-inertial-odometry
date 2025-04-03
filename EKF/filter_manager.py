r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker.py
"""
import json
from typing import Optional
import numpy as np
from numba import jit

from EKF.filter import MSCEKF
from EKF.imu_buffer_calib import IMUBuffer, IMUCalibrator
import EKF.as_torch_script as asTorchScript
from utils import general_utils as gut


class FilterManager:
    r"""
    FilterManager class is responsible for:
        - Receiving the raw imu measurements
        - Filling the inertial buffer with calibrated IMU measurements
        - Running the network to get the displacements and uncertainties
        - Driving the filter with IMU readings and network outputs
    """
    def __init__(
            self, 
            model_path, # path to the nn model
            model_param_path, # path to the json file with nn params
            update_freq, # freq of filter udpates
            filter_tuning_cfg, # config file for the filter
            imu_calib: Optional[IMUCalibrator]=None, # optional calibration object for imu
            force_cpu=False, # force model execution on CPU if True
    ):
        #----------------------CONFIG----------------------#
        # ensure the file format is correct
        if not model_param_path.lower().endswith(".json"):
            raise ValueError(f"Error: The file '{model_param_path}' is not a JSON file.")
        
        # loading network configuration
        config_from_nn = gut.dotdict({})
        try:
            with open(model_param_path) as f_json:
                data_json = json.load(f_json)
            # extract required params
            for key in ["imu_freq", "past_time", "window_time", "arch"]:
                if key not in data_json:
                    raise KeyError(f"Missing key '{key}' in JSON file: '{model_param_path}'")
                config_from_nn[key] = data_json[key]
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file '{model_param_path}': {e}")

        # frequencies and size conversion:
        if not (config_from_nn.past_time * config_from_nn.imu_freq).is_integer():
            raise ValueError(f"Past time cannot be converted to integer number of IMU data points.")
        if not (config_from_nn.window_time * config_from_nn.imu_freq).is_integer():
            raise ValueError(f"Window time cannot be converted to integer number of IMU data points.")
        # set to class attributes
        self.imu_freq = config_from_nn.imu_freq
        self.past_data_size = int(config_from_nn.past_time * config_from_nn.imu_freq)
        self.disp_window_size = int(config_from_nn.window_time * config_from_nn.imu_freq)
        self.nn_input_size = self.disp_window_size + self.past_data_size
        
        #----------------------INITIALIZATION----------------------#
        # prepare the filter as state estimator
        self.icalib = imu_calib
        self.filter_tuning_cfg = filter_tuning_cfg
        self.filter = MSCEKF(filter_tuning_cfg)

        # prepare the network as measurement model
        self.meas_net = asTorchScript(model_path, force_cpu)
        self.imu_buffer = IMUBuffer()

    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_nn(self, t0_us, t1_us, t_oldest_state_us):
        net_t0_us = t0_us
        net_t1_us = t1_us
