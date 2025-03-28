r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker.py
"""
import json
from typing import Optional
import numpy as np
from numba import jit
from filter import MSCEKF
from imu_buffer_calib import IMUBuffer, IMUCalibrator
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
        
