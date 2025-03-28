r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/scekf.py
"""
import numpy as np
from numba import jit
import logging
import utils.math_utils as maths
from filter_states import States

@jit(nopython=True, parallel=False, cache=True)
def state_propagation(R_k, v_k, p_k, bg_k, ba_k, gyro, acc, g, dt):
    """ 
    Propagates the rotation (R), velocity (v), position (p) and biases (ba, bg) of a system 
    while computing the state transition Jacobian (A).
    Inputs:
    - R_k:     3x3 Rotation matrix at time step k
    - v_k:     3x1 Velocity vector at time step k (m/s)
    - p_k:     3x1 Position vector at time step k (m)
    - b_gk:    3x1 Gyroscope bias (rad/s)
    - b_ak:    3x1 Accelerometer bias (m/s²)
    - gyro:    3x1 Measured gyroscope reading (rad/s)
    - acc:     3x1 Measured accelerometer reading (m/s²)
    - g:       3x1 Gravity vector (m/s²)
    - dt:      Time step (seconds)
    Outputs:
    - R_next:  Updated rotation matrix (3x3)
    - v_next:  Updated velocity vector (3x1)
    - p_next:  Updated position vector (3x1)
    - A:       15x15 Jacobian matrix of state transition
    More details: https://github.com/CathIAS/TLIO/issues/23#issuecomment-848313945
    NOTE: the gyro and acc biases are not updated!
    """
    def hat(vector):
        """
        Converts a 3x1 vector into a 3x3 skew-symmetric matrix.
        This is used for cross-product calculations.
        
        If v = [x, y, z], then:
        hat(v) = [[  0  -z   y ]
                  [  z   0  -x ]
                  [ -y   x   0 ]]
        """
        v = vector.flatten()
        return np.array([
            [0, -v[2], v[1]], 
            [v[2], 0, -v[0]], 
            [-v[1], v[0], 0]
        ])
    #---------Rotation update---------#
    dtheta = (gyro - bg_k) * dt # corrected angular displacement (rad)
    dR = maths.exponential_SO3_operation(dtheta) # convert small-angle rotation vector to rotation matrix
    R_next = R_k @ dR # update the rotation matrix

    #---------Velocity and position update---------#
    acc_corrected = acc - ba_k # remove accelerometer bias
    dv_world = R_k @ (acc_corrected * dt) # velocity change in world frame (m/s)
    dp_world = 0.5 * dv_world * dt # position change due to dv_world

    # gravity influence
    g_dt_vel = g * dt
    g_dt_pos = 0.5 * g_dt_vel * dt

    # update velocity and position
    v_next = v_k + dv_world + g_dt_vel # m/s
    p_next = p_k + v_k * dt + dp_world + g_dt_pos # m

    #---------Compute Jacobian (State Transition Matrix A)---------#
    A = np.eye(15)  # Initialize as identity matrix (15x15)
    A[3:6, :3] = -hat(dv_world) # -skew(R_k(a_k-ba_k))*dt
    A[6:9, :3] = -hat(dp_world) # -0.5*skew(R_k(a_k-ba_k))*dt^2

    A[6:9, 3:6] = np.eye(3) * dt # I*dt
    A[:3, 9:12] = -R_next @ maths.exponential_SO3_right_jacob(dtheta) * dt # -R_{k+1}*Jr((w_k-bg_k)*dt)*dt

    R_k_dt = R_k * dt
    A[3:6, 12:15] = -R_k_dt # -R_k*dt
    A[6:9, 12:15] = -0.5 * R_k_dt * dt # -0.5*R_k*dt^2
    return R_next, v_next, p_next, A

@jit(nopython=True, parallel=False, cache=True)
def state_cov_propagation(A_aug, B_aug, dt, Sigma, W, Q):
    
    return


def get_rotation_from_gravity(acc):
    """Estimate the orientation of the sensor relative to the gravity vector."""
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return maths.rot_2vec(acc, ig_w)

class MSCEKF:
    def __init__(self, config=None):
        # sanity check of std
        expected_attribute = [
            "sigma_na", # std of acc noise m/s^2
            "sigma_ng", # std of gyro noise rad/s
            "ita_ba", # acc bias random walk m/s^3
            "ita_bg", # gyro bias random walk rad/s^2
            "init_attitude_sigma", # std of attitude(roll/pitch) uncertainty rad
            "init_yaw_sigma", # std of yaw uncertainty rad
            "init_vel_sigma", # std of velocity uncertainty m/s
            "init_pos_sigma", # std of position uncertainty m
            "init_bg_sigma", # std of gyro bias uncertainty rad/s
            "init_ba_sigma", # std of acc bias uncertainty m/s^2
            "mahalanobis_fail_scale", # scaling factor for Mahalanobis distance test (outlier rejection)
        ]
        if not all(hasattr(config, attr) for attr in expected_attribute):
            logging.warning(
                "At least one filter parameter tuning will be left at its default value."
            )
        # gravity
        g_norm = getattr(config, "g_norm", 9.81)
        self.g = np.array([0, 0, -g_norm]).reshape((3, 1))

        # uncertainties of gyro and acc biases and noises
        self.sigma_na = getattr(config, "sigma_na", np.sqrt(1e-3))
        self.sigma_ng = getattr(config, "sigma_ng", np.sqrt(1e-4))
        self.ita_ba = getattr(config, "ita_ba", 1e-4)
        self.ita_bg = getattr(config, "ita_bg", 1e-6)

        # initial uncertainties of filter states
        self.init_attitude_sigma = getattr(config, "init_attitude_sigma", 10.0/180.0*np.pi)
        self.init_yaw_sigma = getattr(config, "init_yaw_sigma", 0.1/180.0*np.pi)
        self.init_vel_sigma = getattr(config, "init_vel_sigma", 1.0)
        self.init_pos_sigma = getattr(config, "init_pos_sigma", 0.001)
        self.init_bg_sigma = getattr(config, "init_bg_sigma", 0.0001)
        self.init_ba_sigma = getattr(config, "init_ba_sigma", 0.2)

        # measurement covariance related params
        self.meascov_scale = getattr(config, "meascov_scale", 1.0) # scaling factor for measurement covariance
        self.add_sim_meas_noise = getattr(config, "add_sim_meas_noise", False)
        if self.add_sim_meas_noise:
            self.sim_meas_cov_val = config.sim_meas_cov_val
            self.sim_meas_cov_val_z = config.sim_meas_cov_val_z
        # whether use constant covariance values
        self.use_const_cov = getattr(config, "use_const_cov", False)
        if self.use_const_cov:
            self.const_cov_val_x = config.const_cov_val_x
            self.const_cov_val_y = config.const_cov_val_y
            self.const_cov_val_z = config.const_cov_val_z
        
        # mahalanobis params
        self.mahalanobis_factor = 1 # mahalanobis factor
        self.mahalanobis_fail_scale = getattr(
            config, "mahalanobis_fail_scale", 0
        ) # handling cases when the mahalanobis test fails
        self.last_success_mahalanobis = None
        self.force_mahalanobis_until = None

        # state variables
        self.W = None # IMU measurement noise
        self.Q = None # stochastic noise (random walk)
        self.R = None # measurement noise
        self.Sigma = None # full state covariance
        self.Sigma15 = None # evolving state covariance
        self.state = States()
        self.last_timestamp_reset_us = None

        # IMU interpolated data online saving
        self.imu_data_interpolated = np.array([])

        # debug logs
        self.innovation = np.zeros((3, 1))
        self.meas = np.zeros((3, 1))
        self.pred = np.zeros((3, 1))
        self.meas_sigma = np.zeros((3, 1))
        self.inno_sigma = np.zeros((3, 1))

        # decision flags
        self.is_initialized = False
        self.is_converged = False
        self.is_first_update = True

    #---------------Initialize and Reset---------------#
    def initialize_state_covariance(self):
        """
        Prepare the full/evolving state covariance matrices.
        """
        # prepare the variance of states
        var_atti = np.power(self.init_attitude_sigma, 2.0)
        var_yaw = np.power(self.init_yaw_sigma, 2.0)
        var_vel = np.power(self.init_vel_sigma, 2.0)
        var_pos = np.power(self.init_pos_sigma, 2.0)
        var_bg_init = np.power(self.init_bg_sigma, 2.0)
        var_ba_init = np.power(self.init_ba_sigma, 2.0)

        # initialize the full state covariance matrix shape (6N+15, 6N+15)
        cov_state_full = np.zeros((15+6*self.state.N, 15+6*self.state.N))
        # fill each 6x6 diagonal blocks of covariance matrix with past state variances
        for i, _ in enumerate(self.state.si_timestamps_us):
            cov_state_full[6*i:, 6*i:][:3, :3] = np.diag([var_atti, var_atti, var_yaw])
            cov_state_full[6*i:, 6*i:][3:6, 3:6] = np.diag([var_pos, var_pos, var_pos])
        # now fill the 15x15 diagonal block (right bottom) with evolving state variances
        cov_state_evolving = cov_state_full[-15:, -15:] # no copy, ref
        cov_state_evolving[:3, :3] = np.diag(np.array([var_atti, var_atti, var_yaw]))
        cov_state_evolving[3:6, 3:6] = np.diag(np.array([var_vel, var_vel, var_vel]))
        cov_state_evolving[6:9, 6:9] = np.diag(np.array([var_pos, var_pos, var_pos]))
        cov_state_evolving[9:12, 9:12] = np.diag(np.array([var_bg_init, var_bg_init, var_bg_init]))
        cov_state_evolving[12:15, 12:15] = np.diag(np.array([var_ba_init, var_ba_init, var_ba_init]))

        # update the state covariance marices
        self.Sigma = cov_state_full
        self.Sigma15 = cov_state_full[-15:, -15:]


    def initialize_sensor_and_state_covariance(self):
        """
        Prepare the covariance matrices of sensor noises and biases.
        """
        # sensor noise variance
        var_na = np.power(self.sigma_na, 2.0)
        var_ng = np.power(self.sigma_ng, 2.0)
        # sensor bias random walk noise variance
        var_ba = np.power(self.ita_ba, 2.0)
        var_bg = np.power(self.ita_bg, 2.0)

        # prepare the sensor noise and bias covariance matrices
        self.W = np.diag(np.array([var_ng, var_ng, var_ng, var_na, var_na, var_na]))
        self.Q = np.diag(np.array([var_bg, var_bg, var_bg, var_ba, var_ba, var_ba]))
        self.initialize_state_covariance()

    
    
    def reset_state_and_covariance(self, Rs, ps, R, v, p, ba_init, bg_init):
        """
        Prepare both the full states and corresponding covariance matrices.
        """
        # ensure inputs are consistent with the past states size
        assert len(Rs) == self.state.N
        assert len(ps) == self.state.N

        # reset states and corresponding covariance matrices
        self.state.reset_state(Rs, ps, R, v, p, ba_init, bg_init)
        self.initialize_state_covariance()
        # update last reset timestamp
        self.last_timestamp_reset_us = self.state.s_timestamp_us

        # validate covariance matrix size
        expected_sigma_shape = (self.state.N*6+15, self.state.N*6+15)
        assert self.Sigma.shape == expected_sigma_shape
    
    
    def initialize_only_state(self, t_us, R, v, p, ba_init, bg_init):
        """
        Set the evolving state with given values.
        """
        self.state.initialize_state(t_us, R, v, p, ba_init, bg_init)
        self.last_timestamp_reset_us = t_us
    
    def initialize_covs_with_state(self, t_us, R, v, p, ba_init, bg_init):
        """
        Prepare the sensor and state covariance matrices and set the evolving state.
        """
        assert isinstance(t_us, int)
        self.initialize_sensor_and_state_covariance() # prepare the sensor covariance matrices
        self.initialize_only_state(t_us, R, v, p, ba_init, bg_init) # set the evolving state
        self.is_initialized = True # confirm initialization
        logging.info(f"Initialized filter with evolving state, sensor covariance and state covariance matrices!")
        
    
    def initialize_covs_with_zero_state(self, t_us, acc, ba_init, bg_init):
        """
        Prepare the sensor and state covariance matrices, 
        set the evolving state with zero vel and pos, and gravity rotation. 
        """
        assert isinstance(t_us, int)
        self.initialize_sensor_and_state_covariance()
        self.initialize_only_state(
            t_us,
            get_rotation_from_gravity(acc),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            ba_init,
            bg_init
        )
        self.is_initialized = True
        logging.info(f"Initialized filter with zeros evolving state.")
        logging.info(f"Initialized filter with sensor and state covaraince matrices.")
    
    #---------------Get---------------#
    def get_past_state(self, t_us):
        """
        Return a past state with a given timestamp.
        """
        assert isinstance(t_us, int)
        state_idx = self.state.si_timestamps_us.index(t_us)
        past_R = self.state.si_Rs[state_idx]
        past_p = self.state.si_ps[state_idx]
        return past_R, past_p
    

    def get_evolving_state(self):
        """
        Return the current evolving state.
        """
        R = self.state.s_R.copy()
        v = self.state.s_v.copy()
        p = self.state.s_p.copy()
        bg = self.state.s_bg.copy()
        ba = self.state.s_ba.copy()
        return R, v, p, ba, bg
    

    def get_covariance(self):
        """
        Return the full and evolving state covariance matrices.
        """
        Sigma = self.Sigma
        Sigma15 = self.Sigma15
        return Sigma, Sigma15
    
    def get_covariance_yaw_and_pos(self):
        """
        Return a submatrix of Sigma15 that includes the covariacne related to yaw and position.
        """
        return self.Sigma15[[2, 6, 7, 8], :][:, [2, 6, 7, 8]]
    
    def get_info_along_unobservable_shift(self):
        return np.diag(
            self.state.unobs_shift.T
            @ np.linalg.pinv(self.Sigma)
            @ self.state.unobs_shift
        )
    
    def get_debug(self):
        """
        Return debug variables.
        """
        return self.innovation, self.meas, self.pred, self.meas_sigma, self.inno_sigma
    
    #---------------Debug---------------#
    def check_filter_covariance(self):
        """
        Assume the filter has converged after 10 secs.
        """
        return
    
    def is_mahalanobis_activated(self):
        return
    
    #---------------Crucial EKF process---------------#
    def propagate(self, acc, gyro, t_us, t_augmentation_us=None):
        """
        Evolve the system state based on sensor readings, propagate the state forward in time.
        Optionally perform state augmentation if a time augmentation value is provided.
        """
        # initialize the current state
        R_k, v_k, p_k, ba_k, bg_k = (
            self.state.s_R,
            self.state.s_v,
            self.state.s_p,
            self.state.s_ba, 
            self.state.s_bg
        )
        # number of past states
        N = self.state.N
        # difference between current timestamp (t_us) and last timestamp for state update
        dt_us = t_us - self.state.s_timestamp_us 
        # prediction step: evolve current state based on IMU kinematics equations
        R_kp1, v_kp1, p_kp1, A_kp1 = state_propagation(
            R_k, v_k, p_k, bg_k, ba_k,
            gyro, acc, self.g, dt_us*1e-6
        )
        # IMU biases remain the same
        bg_kp1, ba_kp1 = bg_k, ba_k

        # calculate the matrix Bk (half)
        B = np.zeros((15, 6))
        B[:3, :3] = -A_kp1[:3, 9:12]
        B[3:6, 3:6] = -A_kp1[3:6, 12:15]
        B[6:9, 3:6] = -A_kp1[6:9, 12:15]

        # partial integration for state augmentation
        if t_augmentation_us:
            dt_aug_us = t_augmentation_us - self.state.s_timestamp_us
            Rd, vd, pd, Ad = state_propagation(
                R_k, v_k, p_k, bg_k, ba_k,
                gyro, acc, self.g, dt_aug_us*1e-6
            )

            # propagate the additional covariance block (only R and p)
            JA = np.zeros((6, 15))
            JA[:3, :] = Ad[:3, :] # rotation
            JA[3:6, :] = Ad[6:9, :] # position

            # arange the augmented matrix A_aug
            A_aug = np.zeros((15+6*(N+1), 15+6*N))
            A_aug[0:6*N, 0:6*N] = np.eye(6*N)
            A_aug[-15-6:-15, -15:] = JA
            A-A_aug[-15:, -15:] = A_kp1

            # arange the matrix B
            JB = np.zeros((6, 6))
            JB[:3, :3] = -Ad[:3, 9:12]
            JB[3:6, 3:6] = -Ad[6:9, 12:15]

            B_aug = np.zeros((15+6, 6))
            B_aug[-15-6:-15, -15:] = JB
            B_aug[-15:, :] = B

            # augment the past states
            assert Rd.shape == (3, 3), (f"Incorrect inserted past rotation state shape!")
            self.state.si_Rs.append(Rd)
            self.state.si_ps.append(pd)
            self.state.si_Rs_fej.append(Rd)
            self.state.si_ps_fej.append(pd)
            self.state.si_timestamps_us.append(t_augmentation_us)
            self.state.N += 1
        else:
            A_aug = np.eye((15+6*N))
            A_aug[-15:, -15:] = A_kp1

            B_aug = np.zeros((15, 6))
            B_aug[-15:, :] = B
        # propagate the state covariance matrix
        Sigma_kp1 = state_cov_propagation(
            A_aug, B_aug, dt_us*1e-6, self.Sigma, self.W, self.Q
        )

        # update states and covariance variables
        self.state.s_R = R_kp1
        self.state.s_v = v_kp1
        self.state.s_p = p_kp1
        self.state.s_ba = ba_kp1
        self.state.s_bg = bg_kp1
        self.state.s_timestamp_us = t_us

        self.Sigma = Sigma_kp1
        self.Sigma15 = Sigma_kp1[-15:, -15:]
        self.state.unobs_shift = A_aug @ self.state.unobs_shift # propagate unobs shift
    
    def update(self):
        return
    
    def marginalize(self):
        return
    













