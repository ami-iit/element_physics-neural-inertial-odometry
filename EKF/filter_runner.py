r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker_runner.py
"""
import os
from os import path as osp
import numpy as np
from progressbar import progressbar as pbar
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from EKF.imu_buffer_calib import IMUCalibrator
from EKF.filter_manager import FilterManager
from EKF.data_streamer import DataStreamer
from EKF.tlio_streamer import TLIOStreamer
from utils import general_utils as gut
from utils.logging import logging
from utils.o3d_visualizer import O3dVisualizer

class FilterRunner:
    """
    FilterStreamer class is responsible for:
        - Going through a sequence of trajectory
        - Provide input readings to filter_manager
        - Log the output readings from filter_manager
    """
    def __init__(self, args):
        self.args = args
        # TODO: finalize the DataStreamer class!
        self.dStreamer = TLIOStreamer()
        self.dStreamer.load_imu_all(args)
        self.dStreamer.load_vio_all(args)

        print(f"=========== IMU SEQUENCE INFO==========")
        print(f"imu data size: {self.dStreamer.ds_size}")
        print(f"imu data start timestamp: {self.dStreamer.init_ts*1e-6}, end timestamp: {self.dStreamer.imu_ts[-1]*1e-6}")
        print(f"imu freq: {int(self.dStreamer.ds_size/(self.dStreamer.imu_ts[-1]*1e-6-self.dStreamer.init_ts*1e-6))}")
        print(f"imu acc shape: {self.dStreamer.imu_acc.shape}")
        print(f"imu gyro shape: {self.dStreamer.imu_gyro.shape}")

        print(f"=========== VIO SEQUENCE INFO==========")
        print(f"vio data size: {self.dStreamer.vio_ts.shape[0]}")
        print(f"vio data start timestamp: {self.dStreamer.vio_ts[0]}, end timestamp: {self.dStreamer.vio_ts[-1]}")
        print(f"vio freq: {int(self.dStreamer.vio_ts.shape[0]/(self.dStreamer.vio_ts[-1]-self.dStreamer.vio_ts[0]))}")
        print(f"vio position shape: {self.dStreamer.vio_p.shape}")
        print(f"vio orientation shape: {self.dStreamer.vio_R.shape}")
        print(f"vio velocity shape: {self.dStreamer.vio_v.shape}")
    
        # TODO: prepare the visualizer
        self.visualizer = None
        if args.flags.visualize:
            vio_ghost = np.concatenate([
                self.dStreamer.vio_ts[:, None],
                self.dStreamer.vio_quat,
                self.dStreamer.vio_p
            ], axis=1)
            self.visualizer = O3dVisualizer(vio_ghost)
            logging.info(f"Visualizer is ready!")

        # prepare a dir to store the output files
        self.out_dir = osp.join(args.io.out_dir, args.io.dataset_number)
        if osp.exists(self.out_dir) is False:
            os.mkdir(self.out_dir)
        
        # initialize the log files for filter outputs (pose traj, vel and biases)
        traj_logfile = osp.join(self.out_dir, "stamped_traj_estimate.txt") # to store filter estimated pose seq
        if osp.exists(traj_logfile):
            os.remove(traj_logfile)
            logging.info(f"removing previous trajectory log file {traj_logfile}...")
        else:
            logging.info(f"logging the trajectory file {traj_logfile} for the first time!")
        self.traj_logfile = traj_logfile
        self.seq_traj_logs = []

        vel_logfile = osp.join(self.out_dir, "stamped_vel_estimate.txt") # to store filter estimated velocity seq
        if osp.exists(vel_logfile):
            os.remove(vel_logfile)
            logging.info(f"removing previous velocity log file {vel_logfile}...")
        else:
            logging.info(f"logging the velocity file {vel_logfile} for the first time!")
        self.vel_logfile = vel_logfile
        self.seq_vel_logs = []

        biases_logfile = osp.join(self.out_dir, "stamped_biases_estimate.txt") # to store filter estimated biases seq
        if osp.exists(biases_logfile):
            os.remove(biases_logfile)
            logging.info(f"removing previous trajectory log file {biases_logfile}...")
        else:
            logging.info(f"logging the biases file {biases_logfile} for the first time!")
        self.biases_logfile = biases_logfile
        self.seq_biases_logs = []

        # TODO: figure out why we need this?
        self.log_full_state = args.flags.log_full_state
        if self.log_full_state:
            fullstate_logfile = osp.join(self.out_dir, "full_state.txt")
            if osp.exists(fullstate_logfile):
                os.remove(fullstate_logfile)
                logging.info(f"removing previous trajectory log file {fullstate_logfile}...")
            else:
                logging.info(f"logging the full state file {fullstate_logfile} for the first time!")
            self.fullstate_logfile = fullstate_logfile
            self.full_state_logs = open(self.fullstate_logfile, "w")

        # TODO: get done the imu_calib
        imu_calib = IMUCalibrator.get_calib_from_offline_dataset(args)

        # prepare the filter tuning params
        filter_tuning = gut.dotdict({
            "g_norm": args.filter.g_norm,
            "sigma_na": args.filter.sigma_na, # acc noise std, m/s^2
            "sigma_ng": args.filter.sigma_ng, # gyro noise std, rad/s
            "eta_ba": args.filter.eta_ba, # acc bias evolution param over sampling period (random walk)
            "eta_bg": args.filter.eta_bg, # gyro bias evolution params over sampling period (random walk)
            "init_sigma_attitude": args.filter.init_sigma_attitude, # initial uncertainty in attitude estimate, rad
            "init_sigma_yaw": args.filter.init_sigma_yaw, # initial uncertainty in yaw estimate, rad
            "init_sigma_vel": args.filter.init_sigma_vel, # initial uncertainty in velocity estimate, m/s
            "init_sigma_pos": args.filter.init_sigma_pos, # initial uncertainty in position estimate, m
            "init_sigma_bg": args.filter.init_sigma_bg,
            "init_sigma_ba": args.filter.init_sigma_ba,
            "meas_cov_scale": args.filter.meas_cov_scale, # scaling factor for measurement covariance
            "use_const_cov": args.flags.use_const_cov, # whether to use constant covariance
            "const_cov_val_x": args.consts.const_cov_val_x, 
            "const_cov_val_y": args.consts.const_cov_val_y, 
            "const_cov_val_z": args.consts.const_cov_val_z,
            "add_sim_meas_noise": args.flags.add_sim_meas_noise, # whether to add noise to simulated measurements
            "sim_meas_cov_val": args.consts.sim_meas_cov_val,
            "sim_meas_cov_val_z": args.consts.sim_meas_cov_val_z,
            "mahalanobis_fail_scale": args.consts.mahalanobis_fail_scale # scaling factor for mahalanobis distance threshold
        })

        # TODO: initialize the filter manager object
        # NOTE: allow the filter to read meas from vio instead of network (for debug)
        self.manager = FilterManager(
            model_path=args.io.model_path,
            model_param_path=args.io.model_param_path,
            update_freq=args.filter.update_freq,
            filter_tuning_cfg=filter_tuning,
            imu_calib=imu_calib,
            force_cpu=False,
            debug_filter=True
        )
        # log full state buffer
        self.log_full_state_buffer = None

    def __del__(self):
        """Ensure the file is closed properly when the object is destroyed."""
        if self.log_full_state:
            try:
                self.full_state_logs.close()
            except Exception as e:
                logging.exception(f"Error closing full state log file: {e}")
    
    def add_data_to_be_logged(self, ts, acc, gyro, with_update):
        """Extract and log the filter states at a given timestamp ts."""
        R, v, p, ba, bg = self.manager.filter.get_evolving_state() # return the current filter state
        
        p_temp = p.reshape((3,))
        v_temp = v.reshape((3,))
        ba_temp = ba.reshape((3,))
        bg_temp = bg.reshape((3,))
        # update traj logs
        quat = Rotation.from_matrix(R).as_quat()
        traj_step = np.array([
            ts, p_temp[0], p_temp[1], p_temp[2], 
            quat[0], quat[1], quat[2], quat[3]
        ])
        self.seq_traj_logs.append(traj_step)

        # update velocity logs
        vel_step = np.array([ts, v_temp[0], v_temp[1], v_temp[2]])
        self.seq_vel_logs.append(vel_step)

        # update biases logs
        bias_step = np.array([
            ts, 
            bg_temp[0], bg_temp[1], bg_temp[2], 
            ba_temp[0], ba_temp[1], ba_temp[2]
        ])
        self.seq_biases_logs.append(bias_step)

        # in case also log the past full state
        if self.log_full_state:
            _, Sigma15 = self.manager.filter.get_covariance()
            sigma15_diag = np.diag(Sigma15).reshape((15, 1))
            sigma_yaw_pos = self.manager.filter.get_covariance_yaw_and_pos().reshape((16, 1))
            inno, meas, pred, meas_sigma, inno_sigma = self.manager.filter.get_debug()
            if not with_update:
                inno *= np.nan
                meas *= np.nan
                pred *= np.nan
                meas_sigma *= np.nan
                inno_sigma *= np.nan
            
            ts_temp = ts.reshape((1, 1))
            state_temp = np.concatenate(
                [
                    v, p, ba, bg, 
                    acc, gyro, ts_temp,
                    sigma15_diag, inno, meas, pred,
                    meas_sigma, inno_sigma, sigma_yaw_pos
                ], 
                axis=0
            )
            vec_flat = np.append(R.ravel(), state_temp.ravel(), axis=0)

            # append full state to the buffer array
            if self.log_full_state_buffer is None:
                self.log_full_state_buffer = vec_flat
            else:
                self.log_full_state_buffer = np.vstack((self.log_full_state_buffer, vec_flat))
            
            # write the buffer array to the state logs
            if self.log_full_state_buffer.shape[0] > 100:
                np.savetxt(self.full_state_logs, self.log_full_state_buffer, delimiter=",")
                self.log_full_state_buffer = None

    def save_logs(self, save_as_npy):
        logging.info(f"Saving logs after iterating the whole imu sequence...")
        # save the seq_traj, seq_vel, seq_biases log files
        np.savetxt(
            self.traj_logfile, 
            np.array(self.seq_traj_logs),
            header="ts x y z qx qy qz qw", fmt="%.12f"
        )
        np.savetxt(
            self.vel_logfile,
            np.array(self.seq_vel_logs),
            header="ts vx vy vz", fmt="%.3f"
        )
        np.savetxt(
            self.biases_logfile,
            np.array(self.seq_biases_logs),
            header="ts bg_x bg_y bg_z ba_x ba_y ba_z", fmt="%.3f"
        )
        # if needed, save also the full state log file
        if self.log_full_state:
            np.savetxt(
                self.full_state_logs,
                self.log_full_state_buffer,
                delimiter=","
            )
            self.log_full_state_buffer = None
            self.full_state_logs.close()
            if save_as_npy:
                states = np.loadtxt(self.fullstate_logfile, delimiter=",")
                np.save(self.fullstate_logfile[:-4] + ".npy", states)
                os.remove(self.fullstate_logfile)
        logging.info(f"All filter estimates successfully saved!")

    def run(self):
        """
        This is the main execution loop for runnig an IMU-based tracking system.
            - Process the IMU data
            - Integrate the measurements from VIO gt or network outpus
            - Update the state of a filter 
        """
        # two initialization strategies for first update
        def initialize_with_vio_at_first_update(this):
            logging.info(f"Initialize filter with vio ground truth after {this.last_t_us*1e-6} sec.")
            self.reset_filter_state_from_vio(this)
        
        def initialize_with_zeros_at_first_update():
            logging.info(f"Initialize filter positions and velocities with zeros at first update.")
            self.reset_filter_state_with_zeros()

        # assign one of the callbacks to the manager's callback_first_update method
        if self.args.flags.initialize_with_vio:
            self.manager.callback_first_update = initialize_with_vio_at_first_update
        else:
            self.manager.callback_first_update = initialize_with_zeros_at_first_update
        # bypass the network and extract vio data for the measurement model
        if self.args.flags.use_vio_meas:
            self.manager.debug_callback_get_meas = lambda t0, t1: self.dStreamer.get_meas_from_vio(t0, t1)

        # iterate through the IMU sequence
        n_data = self.dStreamer.ds_size
        factor = 1.0
        for i in pbar(range(int(n_data * factor))):
            # fetch single step imu data
            # TODO: check the unit of imu_ts here!
            imu_ts, acc_raw, gyro_raw = self.dStreamer.get_datapoint(i)
            #print(f"imu_ts: {imu_ts}, shape: {np.array([imu_ts]).shape}")
            imu_t_us = int(imu_ts * 1e6) # convert to microseconds

            # if debugging, replace acc and gyro biases with values from VIO calibration
            if self.args.flags.debug_using_vio_bias:
                # TODO: check the unit of vio_ts here! Should be aligned with imu_ts!
                # NOTE: vio_ba and vio_bg not existing in dataStreamer!!
                #print(f"vio_ba: {self.dStreamer.vio_ba}")
                vio_ba = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_ba, axis=0)(imu_ts)
                vio_bg = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_bg, axis=0)(imu_ts)
                self.manager.filter.state.s_ba = np.atleast_2d(vio_ba).T
                self.manager.filter.state.s_bg = np.atleast_2d(vio_bg).T

            if self.manager.filter.is_initialized:
                # if the filter is already initialized, 
                # pass the IMU data to the filter for state estimation update
                # then add the updated data to logs
                # TODO: the function on_imu_measurement not implemented yet!
                did_update = self.manager.on_imu_measurement(imu_t_us, gyro_raw, acc_raw)
                self.add_data_to_be_logged(
                    imu_ts,
                    self.manager.last_acc_before_next_interp_time,
                    self.manager.last_gyro_before_next_interp_time,
                    did_update
                )

                # get the vio gt state at imu_ts
                vio_p = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_p, axis=0)(imu_ts)
                vio_R = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_R, axis=0)(imu_ts)

                # we ignore the visualization part here
                if i % 100 ==0 and self.visualizer is not None:
                    # update the pose of tlio
                    T_World_imu = np.eye(4)
                    T_World_imu[:3, :3] = self.manager.filter.state.s_R
                    T_World_imu[:3, 3:4] = self.manager.filter.state.s_p
                    # update the pose of vio gt
                    T_gt = np.eye(4)
                    T_gt[:3, :3] = vio_R.reshape((3, 3))
                    T_gt[:3, 3:4] = self.manager.filter.state.s_p.reshape((3, 1))

                    # calculate the position error for the visualizer
                    print(f"{i}-th gt position: {vio_p}")
                    print(f"{i}-th estimated position: {self.manager.filter.state.s_p.reshape((1, 3))}")
                    p_error = np.sqrt((vio_p - self.manager.filter.state.s_p.flatten())**2)
                    print(f"{i}-th per-axis position rmse error: {p_error}")
                    self.visualizer.update(
                        imu_t_us,
                        {"tlio": T_World_imu,
                         "vio_gt": T_gt},
                        {"tlio": [T_World_imu[:3, 3]],
                         "vio_gt": [T_gt[:3, 3]]}
                    )
            else:
                # if not using vio, run the filter anyway
                if not self.args.flags.initialize_with_vio:
                    self.manager.on_imu_measurement(imu_t_us, gyro_raw, acc_raw)
                # otherwise initialize the filter with vio data
                else:
                    # set initial biases default as zeros 
                    init_ba, init_bg = np.zeros((3, 1)), np.zeros((3, 1))
                    # initialize the biases with offline calibrations
                    if self.args.flags.initialize_with_offline_calib:
                        init_ba = self.manager.icalib.accBias
                        init_bg = self.manager.icalib.gyroBias
                    # ensure the imu timestamp is within the vio range
                    if imu_ts < self.dStreamer.vio_ts[0]:
                        logging.info(f"current imu_ts {imu_ts} smaller than first vio_ts {self.dStreamer.vio_ts[0]}")
                        continue

                    # interpolate vio states at imu timestamp imu_ts
                    vio_p = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_p, axis=0)(imu_ts)
                    vio_v = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_v, axis=0)(imu_ts)
                    vio_euler = interp1d(self.dStreamer.vio_ts, self.dStreamer.vio_euler, axis=0)(imu_ts)
                    vio_R = Rotation.from_euler("xyz", vio_euler, degrees=True).as_matrix()
                    # initialize the filter
                    # NOTE: seems a bug from the original code
                    # NOTE: you should give acc_raw and gyro_raw as last two inputs to the following func
                    # NOTE: instead it was given the init_ba and init_bg
                    self.manager.init_with_state_at_time(
                        imu_t_us,
                        vio_R,
                        np.atleast_2d(vio_v).T,
                        np.atleast_2d(vio_p).T,
                        init_ba,
                        init_bg
                    )
        # save the full state logs
        self.save_logs(self.args.flags.save_as_npy)
    
    def reset_filter_state_from_vio(self, this: FilterManager):
        """Rest filter states from VIO data as found in the input."""
        ref = self.dStreamer
        state = this.filter.state
        # reset the past states
        vio_ps, vio_Rs = [], []
        for i, t_init_us in enumerate(state.si_timestamps_us):
            t_init_sec = t_init_us * 1e-6
            ps = np.atleast_2d(interp1d(ref.vio_ts, ref.vio_p, axis=0)(t_init_sec)).T
            vio_ps.append(ps)

            vio_euler = interp1d(ref.vio_ts, ref.vio_euler, axis=0)(t_init_sec)
            vio_Rs.append(Rotation.from_euler("xyz", vio_euler, degrees=True).as_matrix())
        # reset the evolving state
        ts = state.s_timestamp_us * 1e-6
        vio_p = np.atleast_2d(interp1d(ref.vio_ts, ref.vio_p, axis=0)(ts)).T
        vio_v = np.atleast_2d(interp1d(ref.vio_ts, ref.vio_v, axis=0)(ts)).T
        vio_euler = interp1d(ref.vio_ts, ref.vio_euler, axis=0)(ts)
        vio_R = Rotation.from_euler("xyz", vio_euler, degrees=True).as_matrix()

        this.filter.reset_state_and_covariance(
            vio_Rs, vio_ps, 
            vio_R, vio_v, vio_p,
            state.s_ba, state.s_bg
        )

    def reset_filter_state_with_zeros(self):
        """Reset filter states of positions and velocities with zeros."""
        state = self.manager.filter.state()
        # add zeros to past states of positions
        past_ps = []
        for i in state.si_timestamps_us:
            past_ps.append(np.zeros((3, 1)))
        # add zeros to current position and velocity
        p_now = np.zeros((3, 1))
        v_now = np.zeros((3, 1))
        self.manager.filter.reset_state_and_covariance(
            state.si_Rs, past_ps, state.s_R, v_now, p_now, state.s_ba, state.s_bg
        )

            



