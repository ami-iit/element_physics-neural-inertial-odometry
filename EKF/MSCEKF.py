import numpy as np
from numba import jit
import logging
import utils.math_utils as maths
from states import States

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
def state_cov_propagation(A_aug):
    return


def get_rotation_from_gravity(acc):
    """Estimate the orientation of the sensor relative to the gravity vector."""
    ig_w = np.array([0, 0, 1.0]).reshape((3, 1))
    return maths.rot_2vec(acc, ig_w)

class MSCEKF:
    def __init__(self, config=None):








