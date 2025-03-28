r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/scekf.py
"""
from utils import math_utils as maths
import numpy as np

class States(object):
    def __init__(self):
        super(States, self).__init__()
        self.s_R = None
        self.s_v = None
        self.s_p = None
        self.s_bg = None
        self.s_ba = None
        self.s_timestamp_us = -1 # microseconds
        self.N = 0 # number of past states
        self.si_Rs = []
        self.si_ps = []
        self.si_Rs_fej = []
        self.si_ps_fej = []
        self.si_timestamps_us = []
        self.unobs_shift = None

    def initialize_state(self, t_us, R, v, p, ba_init, bg_init):
        assert isinstance(t_us, int)
        # assign state variables
        self.s_R = R
        self.s_v = v # m/s
        self.s_p = p # m
        self.s_bg = bg_init # rad/s
        self.s_ba = ba_init # m/s^2

        # reset past state history
        self.si_Rs.clear()
        self.si_ps.clear()
        self.si_Rs_fej.clear()
        self.si_ps_fej.clear()
        self.si_timestamps_us.clear()

        # comute unobservable shift
        self.unobs_shift = self.generate_unobservable_shift()

    def reset_state(self, Rs, ps, R, v, p, ba_init, bg_init):
        self.s_R = R
        self.s_v = v  # m/s
        self.s_p = p  # m
        self.s_bg = bg_init  # rad/s
        self.s_ba = ba_init  # m/s^2

        self.si_Rs = Rs
        self.si_ps = ps
        self.si_Rs_fej = Rs
        self.si_ps_fej = ps

    def __repr__(self) -> str:
        """
        Returns a string representation of the state object for debugging.
        Ensures that all attributes are displayed in a readable format.
        """
        return (
            f"State(\n"
            f"  R:\n{repr(self.s_R) if self.s_R is not None else 'Not initialized'}\n"
            f"  v:\n{repr(self.s_v) if self.s_v is not None else 'Not initialized'}\n"
            f"  p:\n{repr(self.s_p) if self.s_p is not None else 'Not initialized'}\n"
            f"  bg:\n{repr(self.s_bg) if self.s_bg is not None else 'Not initialized'}\n"
            f"  ba:\n{repr(self.s_ba) if self.s_ba is not None else 'Not initialized'}\n"
            f")"
        )
    
    def update_state_with_corrections(self, dX):
        """
        Given the delta_X obtained with K(h(X)-nn_output), 
        return the updated state vector X = X + dX.
        """
        dX_past = dX[:-15] # N-steps past states of dim 6N
        dX_evol = dX[-15:] # current state of dim 15
        assert dX_past.flatten().shape[0] == (self.N * 6), (
            f"Number of past error states {dX_past.flatten().shape[0]} does not match the number of states {self.N * 6} in the filter!"
        )
        # if past states exist in the state vector, update them
        if self.N > 0:
            dX_past_temp = dX_past.reshape((self.N, 6))
            dps = np.expand_dims(dX_past_temp[:, 3:6], axis=2) # (N, 3, 1)
            dthetas = dX_past_temp[:, :3]
            dRs = maths.vec_exponential_SO3_operation(dthetas) # (N, 3, 3)
            # stack past states (Rs and ps) into an array
            Rs_past = np.stack(self.si_Rs, axis=0)
            ps_past = np.stack(self.si_ps, axis=0)
            # update the past states with delta corrections
            Rs_past_new = np.matmul(dRs, Rs_past)
            ps_past_new = ps_past + dps
            # reshape and store the corrected past states
            N = Rs_past.shape[0]
            self.si_Rs = np.split(Rs_past_new.reshape(N*3, 3), N, 0) # a list of N (3, 3) arrays
            self.si_ps = np.split(ps_past_new.reshape(N*3, 1), N, 0) # a list of N (3, 1) arrays
        
        # update the current state
        dtheta = dX_evol[:3]
        dR = maths.exponential_SO3_operation(dtheta)
        self.s_R = dR.dot(self.s_R)
        dv = dX_evol[3:6]
        self.s_v = self.s_v + dv
        dp = dX_evol[6:9]
        self.s_p = self.s_p + dp
        dbg = dX_evol[9:12]
        self.s_bg = self.s_bg + dbg
        dba = dX_evol[12:15]
        self.s_ba = self.s_ba + dba
    
    def cal_corrections(self, target_state):
        """
        Return the error state vector between the current state and given target state.
        NOTE: target_state is a State object.
        """
        assert target_state.N == self.N
        dX = np.zeros((self.N*6+15, 1))

        # update the corrections for past states
        for i in range(len(self.si_Rs)):
            # first append rotation correction of 3-dim
            dX[6*i:6*i+3] = maths.logarithm_map_R(
                target_state.si_Rs[i] @ (self.si_Rs[i].inverse())
            )
            # then append position correction of 3-dim
            dX[6*i+3:6*i+6] = target_state.si_ps[i] - self.si_ps[i]
        
        # update the corrections for current state
        dX[6*self.N:][:3, 0] = maths.logarithm_map_R(target_state.s_R @ self.s_R.T)
        dX[6*self.N:][3:6, :] = target_state.s_v - self.s_v
        dX[6*self.N:][6:9, :] = target_state.s_p - self.s_p
        dX[6*self.N:][9:12, :] = target_state.s_bg - self.s_bg
        dX[6*self.N:][12:15, :] = target_state.s_ba - self.s_ba

        return dX
    
    def generate_unobservable_shift(self):
        """
        Return a dX element along certain unobservable directions.
        """
        assert self.N == 0
        g = np.array([[0], [0], [1]])
        dX = np.zeros((15, 4))

        dX[0:3, [0]] = g
        dX[3:6, [0]] = -maths.hat(self.s_v) @ g
        dX[6:9, [0]] = -maths.hat(self.s_p) @ g
        dX[6:9, 1:4] = np.eye(3)
        return dX








    
