import numpy as np


class dotdict(dict):
    r"""
    A simple subclass of Python's built-in dict that allows attribute-style access
    to dict keys, e.g., obj.key instead of obj['key'].
    """
    def __setattr__(self, name, value):
        """Instead of storing attributes in self.__dict__, stors them as dict keys."""
        self[name] = value 
    def __getattr__(self, name):
        """
        If an attribute isn't found in the normal way, try to retrieve it from the dict,
        if the key does not exist, raise an error.
        """
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __repr__(self):
        """Return a string representation of the dict."""
        return super().__repr__()
    

def align_inertial_frames(T_est, T_gt):
    """
    Align two inertial frames using a single example, aligning by position and yaw
    """
    C = T_est[:3,:3] @ T_gt[:3,:3].T
    yaw = np.arctan2(C[0,1] - C[1,0], C[0,0] + C[1,1])
    R = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    p = T_gt[:3,3:4] - R @ T_est[:3,3:4]
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3:4] = p
    return T