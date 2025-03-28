r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/tracker/imu_tracker.py
"""
import json
from typing import Optional
import numpy as np
from numba import jit
from filter import MSCEKF
