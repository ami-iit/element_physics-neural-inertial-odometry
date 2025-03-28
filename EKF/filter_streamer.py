r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker_runner.py
"""
import os
import numpy as np
from progressbar import progressbar as pbar
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from imu_buffer_calib import IMUCalibrator
from filter_manager import FilterManager
from utils import general_utils as gut
