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
from data_streamer import DataStreamer
from utils import general_utils as gut


class FilterRunner:
    """
    FilterStreamer class is responsible for:
        - Going through a sequence of trajectory
        - Provide input readings to filter_manager
        - Log the output readings from filter_manager
    """
    def __init__(self, args, )