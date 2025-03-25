r"""Define the EKF states."""
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
        



    
