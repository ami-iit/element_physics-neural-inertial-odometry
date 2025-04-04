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
from utils.logging import logging
import utils.math_utils as maths

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
            debug_filter=True # filter debugging mode, bypass the network
    ):
        #----------------------CONFIG----------------------#
        self.debug_filter = debug_filter
        if not self.debug_filter:
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
        else:
            logging.info(f"Debugging mode: use vio data for measurements update.")
            self.imu_freq_net = 200.0 # Hz
            self.past_time = 0.0 # sec
            self.window_time = 1.0 # sec
            self.arch = None # network architecture

            if not (self.past_time * self.imu_freq_net).is_integer():
                raise ValueError(f"past time cannot be represented by integer number of IMU data.")
            if not (self.window_time * self.imu_freq_net).is_integer():
                raise ValueError(f"window time cannot be represented by integer number of IMU data.")
            self.past_data_size = int(self.past_time * self.imu_freq_net)
            self.disp_window_size = int(self.window_time * self.imu_freq_net)
            self.net_input_size = self.disp_window_size + self.past_data_size

            if not (self.imu_freq_net / update_freq).is_integer():
                raise ValueError(f"update freq must be divisble by imu_freq_net.")
            if not (self.window_time * update_freq).is_integer():
                raise ValueError(f"window time cannot be represented by integer number of IMU data.")

            self.update_freq = update_freq
            self.clone_every_n_netimu_sample = int(self.imu_freq_net / update_freq)
            assert (self.imu_freq_net % update_freq == 0)
            self.update_distance_num_clone = int(self.window_time * update_freq)

            self.dt_interp_us = int(1.0 / self.imu_freq_net * 1e6)
            self.dt_update_us = int(1.0 / self.update_freq * 1e6)
        
        #----------------------INITIALIZATION----------------------#
        # prepare the filter as state estimator
        self.icalib = imu_calib
        self.filter_tuning_cfg = filter_tuning_cfg
        self.filter = MSCEKF(filter_tuning_cfg)

        # prepare the network as measurement model if not debugging the filter itself
        if not debug_filter:
            self.meas_net = asTorchScript(model_path, force_cpu)
            self.imu_buffer = IMUBuffer()

        # prepare the callbacks
        self.callback_first_update = None # called at first update if set
        self.debug_callback_get_meas = None # to bypass the network for measurement

        # time
        self.last_t_us = -1
        self.t_us_before_next_interpolation = -1
        self.next_interp_t_us = None
        self.next_aug_t_us = None
        self.log_time = -1
        
        # acc and gyro for adding the data to be logged
        self.last_acc_before_next_interp_time = None
        self.last_gyro_before_next_interp_time = None

        self.has_done_first_update = False


    #---------------Prepare imu buffer for network---------------#
    @jit(forceobj=True, parallel=False, cache=False)
    def _get_imu_samples_for_nn(self, t_begin_us, t_oldest_state_us, t_end_us):
        # extract corresponding network input data
        net_tus_begin = t_begin_us
        net_tus_end = t_end_us - self.dt_interp_us
        net_acc, net_gyr, net_tus = self.imu_buffer.get_whole_nn_inputs(net_tus_begin, net_tus_end)

        assert net_gyr.shape[0] == self.net_input_size
        assert net_acc.shape[0] == self.net_input_size
        
        # get data from filter
        R_oldest_state_wfb, _ = self.filter.get_past_state(t_oldest_state_us)  # 3 x 3
        # change the input of the network to be in local frame
        ri_z = maths.compute_euler_from_matrix(R_oldest_state_wfb, "xyz", extrinsic=True)[0, 2]
        Ri_z = np.array([
            [np.cos(ri_z), -(np.sin(ri_z)), 0],
            [np.sin(ri_z), np.cos(ri_z), 0],
            [0, 0, 1]
        ])
        R_oldest_state_wfb = Ri_z.T @ R_oldest_state_wfb

        bg = self.filter.state.s_bg
        # dynamic rotation integration using filter states
        # Rs_net will contains delta rotation since t_begin_us
        Rs_bofbi = np.zeros((net_tus.shape[0], 3, 3))  # N x 3 x 3
        Rs_bofbi[0, :, :] = np.eye(3)
        for j in range(1, net_tus.shape[0]):
            dt_us = net_tus[j] - net_tus[j - 1]
            dR = maths.exponential_SO3_operation((net_gyr[j, :].reshape((3, 1)) - bg) * dt_us * 1e-6)
            Rs_bofbi[j, :, :] = Rs_bofbi[j - 1, :, :].dot(dR)

        # find delta rotation index at time ts_oldest_state
        oldest_state_idx_in_net = np.where(net_tus == t_oldest_state_us)[0][0]

        # rotate all Rs_net so that (R_oldest_state_wfb @ (Rs_bofbi[idx].inv() @ Rs_bofbi[i])
        # so that Rs_net[idx] = R_oldest_state_wfb
        R_bofboldstate = (
            R_oldest_state_wfb @ Rs_bofbi[oldest_state_idx_in_net, :, :].T
        )  # [3 x 3]
        Rs_net_wfb = np.einsum("ip,tpj->tij", R_bofboldstate, Rs_bofbi)
        net_acc_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_acc)  # N x 3
        net_gyr_w = np.einsum("tij,tj->ti", Rs_net_wfb, net_gyr)  # N x 3

        return net_gyr_w, net_acc_w

    #---------------Functions to initialize the filter---------------#
    def _compensate_meas_with_init_calib(self, gyro_raw, acc_raw):
        if self.icalib:
            logging.info(f"Using init biases from offline calib data.")
            init_ba = self.icalib.accBias
            init_bg = self.icalib.gyroBias
            acc_calibed, gyro_calibed = self.icalib.calibrate_raw_imu(acc_raw, gyro_raw)
        else:
            logging.info(f"Using zero biases and raw acc and gyro measurements!")
            init_ba, init_bg = np.zeros((3, 1)), np.zeros((3, 1))
            acc_calibed, gyro_calibed = acc_raw, gyro_raw
        return {
            "init_ba": init_ba,
            "init_bg": init_bg,
            "gyro_calibed": gyro_calibed,
            "acc_calibed": acc_calibed
        }
    
    def _add_interpolated_imu_to_buffer(self, acc_calibed, gyro_calibed, t_us):
        """Update the imu buffer as inputs to the network."""
        self.imu_buffer._add_interpolated_data(
            self.t_us_before_next_interpolation,
            t_us,
            self.last_gyro_before_next_interp_time,
            gyro_calibed,
            self.last_acc_before_next_interp_time,
            acc_calibed,
            self.next_interp_t_us
        )
        # TODO: define the dt_interp_us!
        self.next_interp_t_us += self.dt_interp_us


    def _after_filter_init_member_setup(self, t_us, gyro_calibed, acc_calibed):
        self.next_interp_t_us = t_us # current imu timestamp to next_interp_t_us
        self.next_aug_t_us = t_us
        self.log_time = t_us

        # in debugging mode, we don't need to update the imu buffer
        if not self.debug_filter:
            self._add_interpolated_imu_to_buffer(acc_calibed, gyro_calibed, t_us)
            self.next_aug_t_us = t_us + self.dt_interp_us

        self.last_t_us = t_us
        self.t_us_before_next_interpolation = t_us
        self.last_acc_before_next_interp_time = acc_calibed
        self.last_gyro_before_next_interp_time = gyro_calibed
    

    def init_with_state_at_time(self, t_us, R, v, p, gyro_raw, acc_raw):
        """
        Initialize the filter states with vio data, and the biases with offline calib.
        Args:
            - t_us: imu seq timestamp microsecnd
            - R: interpolated vio orientation at t_us
            - v: interpolated vio velocity at t_us
            - p: interpolated vio position at t_us
            - gyro_raw, acc_raw: raw imu data at t_us
        """
        assert R.shape == (3, 3)
        assert v.shape == (3, 1)
        assert p.shape == (3, 1)

        logging.info(f"Initializing the filter at timestamp: {t_us*1e-6} sec.")
        res = self._compensate_meas_with_init_calib(gyro_raw, acc_raw) # get the calirated imu data and biases
        
        # initialize the covariance and states of the filter
        self.filter.initialize_covs_with_state(t_us, R, v, p, res["init_ba"], res["init_bg"])
        self._after_filter_init_member_setup(t_us, res["gyro_calibed"], res["acc_calibed"])
        return False
    
    def _init_without_state_at_time(self, t_us, gyro_raw, acc_raw):
        assert isinstance(t_us, int)
        res = self._compensate_meas_with_init_calib(gyro_raw, acc_raw)
        self.filter.initialize_covs_with_zero_state(
            t_us, 
            res["acc_calibed"], 
            res["init_ba"], 
            res["init_bg"]
        )
        self._after_filter_init_member_setup(t_us, res["gyro_calibed"], res["acc_calibed"])
        return False

    #---------------Functions to update the filter when already initialized---------------#
    def on_imu_measurement(self, t_us, gyro_raw, acc_raw):
        """Process the incoming imu data."""
        assert isinstance(t_us, int)
        
        # t_us is the timestamp recorded in imu seq
        if t_us - self.last_t_us > 3e3:
            logging.warning(f"Large imu time gap: {t_us - self.last_t_us} microseconds.")
        
        if self.filter.is_initialized:
            return self._on_imu_meas_after_init(t_us, gyro_raw, acc_raw)
        else:
            self._init_without_state_at_time(t_us, gyro_raw, acc_raw)
  
    def _on_imu_meas_after_init(self, t_us, gyro_raw, acc_raw):
        # t_us is the imu timestamps at current step when called
        assert isinstance(t_us, int)
        if self.icalib:
            # calibrate raw imu data with offline calibration
            # used as inputs to network prediction
            acc_calibed, gyro_calibed = self.icalib.calibrate_raw_imu(acc_raw, gyro_raw)

            # calibrate raw imu data with offline calibration scale
            # used for the filter
            acc_raw, gyro_raw = self.icalib.scale_raw_imu(acc_raw, gyro_raw)
        else:
            acc_calibed, gyro_calibed = acc_raw, gyro_raw
        
        # decide if we need to interpolate imu data or do update
        r"""
        After first time initialization, next_interp_t_us is copied from last-step t_us,
        so current-step t_us should be larger than next_interp_t_us,
        hence the do_interpolation_of_imu is True.
        The same of next_aug_t_us, so do_augmentation_and_update also True.
        """
        do_interpolation_of_imu = t_us >= self.next_interp_t_us
        do_augmentation_and_update = t_us >= self.next_aug_t_us

        # if augmenting the state, check that we also compute the interpolated meas
        # if both True, should be no warning
        assert(
            do_augmentation_and_update and do_interpolation_of_imu
        ) or not do_augmentation_and_update, (
            "Augmentation and interpolation does not match!"
        )

        # if do augmentation, give the value of next_aug_t_us to t_augmentation_us
        # which is the value of last-step imu t_us
        t_augmentation_us = self.next_aug_t_us if do_augmentation_and_update else None

        if do_interpolation_of_imu:
            if not self.debug_filter:
                self._add_interpolated_imu_to_buffer(acc_calibed, gyro_calibed, t_us)
        # propagate the filter with last-step and current-step imu timestamp
        self.filter.propagate(acc_raw, gyro_raw, t_us, t_augmentation_us)

        did_update = False
        if do_augmentation_and_update:
            did_update = self._process_update(t_us)
            self.next_aug_t_us += self.dt_interp_us
        self.last_t_us = t_us # always record the last-step imu timestamp t_us

        if t_us < self.t_us_before_next_interpolation:
            self.t_us_before_next_interpolation = t_us
            self.last_acc_before_next_interp_time = acc_calibed
            self.last_gyro_before_next_interp_time = gyro_calibed

        return did_update


    def _process_update(self, t_us):
        # t_us is the current-step imu timestamp
        if (t_us - getattr(self, "log_time", 0))*1e-6 >= 1:
            logging.info(f"update at timestamp: {t_us * 1e-6:.6f} sec | state.N: {self.filter.state.N}")
            self.log_time = t_us
        if self.filter.state.N <= self.update_distance_num_clone:
            return False
        t_oldest_state_us = self.filter.state.si_timestamps_us[
            self.filter.state.N - self.update_distance_num_clone - 1
        ]
        t_begin_us = t_oldest_state_us - self.dt_interp_us * self.past_data_size
        t_end_us = self.filter.state.si_timestamps_us[-1]  # always the last state
        
        # If we do not have enough IMU data yet, just wait for next time
        if not self.debug_filter:
            if t_begin_us < self.imu_buffer.net_t_us[0]:
                return False
        
        # initialize with vio at the first update
        if not self.has_done_first_update and self.callback_first_update:
            self.callback_first_update(self)
        assert t_begin_us <= t_oldest_state_us
        
        if self.debug_callback_get_meas:
            # use vio data for measurements
            # print(f"t_oldest: {t_oldest_state_us}")
            # print(f"t_end: {t_end_us}")
            meas, meas_cov = self.debug_callback_get_meas(t_oldest_state_us, t_end_us)
        else:  
            # using network for measurements
            net_gyr_w, net_acc_w = self._get_imu_samples_for_network(
                t_begin_us, t_oldest_state_us, t_end_us
            )
            meas, meas_cov = self.meas_source.get_displacement_measurement(
                net_gyr_w, net_acc_w
            )
        # filter update
        self.filter.update(meas, meas_cov, t_oldest_state_us, t_end_us)
        self.has_done_first_update = True
        
        # marginalization of all past state with timestamp before or equal ts_oldest_state
        # print(f"t oldest: {t_oldest_state_us}")
        # print(f"past ts: {self.filter.state.si_timestamps_us}")
        oldest_idx = self.filter.state.si_timestamps_us.index(t_oldest_state_us)
        cut_idx = oldest_idx
        #logging.info(f"marginalize state before index {cut_idx}")
        self.filter.marginalize(cut_idx)
        if not self.debug_filter:
            self.imu_buffer.throw_data_before(t_begin_us)
        return True
    