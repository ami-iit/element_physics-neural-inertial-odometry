r"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/main_filter.py
"""
import datetime
import json
import numpy as np
import os
import os.path as osp
import warnings
import numpy as np
from utils.general_utils import dotdict
from utils.logging import logging
from numba.core.errors import NumbaPerformanceWarning
from EKF.filter_runner import FilterRunner

debug_first = False
if not debug_first:
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

if __name__ == "__main__":
    #------------------- I/O params -------------------#
    pIO = dotdict({
        "root_dir": "./local_data/tlio_golden",
        "dataset_number": "145820422949970",
        "out_dir": "./local_data/out_dir_tlio",
        "model_path": None,
        "model_param_path": None,
        "out_filename": None,
        "sim_data_path": None,
        "start_from_ts": None,
        "erase_old_logs": None
    })
    #------------------- Filter and Network params -------------------#
    pFilter = dotdict({
        "cpu": None, # decide the device
        "g_norm": 9.81,
        "update_freq": 20.0, # Hz
        "sigma_na": np.sqrt(1e-3),
        "sigma_ng": np.sqrt(1e-4),
        "eta_ba": 1e-4,
        "eta_bg": 1e-6,
        "init_sigma_attitude": 1.0/180.0*np.pi,
        "init_sigma_yaw": 0.1/180.0*np.pi,
        "init_sigma_vel": 1.0, # m/s
        "init_sigma_pos": 0.001, # m
        "init_sigma_bg": 0.0001,
        "init_sigma_ba": 0.02,
        "meas_cov_scale": 10.0
    })
    #------------------- Decision params -------------------#
    pFlags = dotdict({
        "initialize_with_vio": True, # initialize filter state with gt data
        "initialize_with_offline_calib": True, # initialize bias state with offline calib or 0
        "use_const_cov": False,
        "use_vio_meas": True, # use gt delta pose to udpate filter
        "debug_using_vio_bias": False,
        "add_sim_meas_noise": False,
        "visualize": True,
        "log_full_state": True,
        "save_as_npy": True
    })
    #------------------- Constant params -------------------#
    pConsts = dotdict({
        "const_cov_val_x": 0.01,
        "const_cov_val_y": 0.01,
        "const_cov_val_z": 0.01,
        "sim_meas_cov_val": np.power(0.01, 2.0),
        "sim_meas_cov_val_z": np.power(0.01, 2.0),
        "mahalanobis_fail_scale": 0 
    })
    #------------------- Pack up params -------------------#
    # pack up the params
    args = dotdict({
        "io": pIO,
        "filter": pFilter,
        "flags": pFlags,
        "consts": pConsts
    })

    #------------------- Start tracking-------------------#
    # control the display of width of numpy arrays when printed
    np.set_printoptions(linewidth=2000)

    # if we want to run multiuple sequences
    data_list = []
    data_names = [pIO.dataset_number]

    # make sure the out_dir exists
    if not osp.exists(pIO.out_dir):
        os.mkdir(pIO.out_dir)

    # prepare the output params file to store
    if False:
        param_dict = vars(args)
        param_dict["date"] = str(datetime.datetime.now())
        with open(args.io.out_dir + "/parameters.json", "w") as f_params:
            f_params.write(json.dumps(param_dict, indent=4, sort_keys=True))
    
    try:
        # now for testing: run one single file
        logging.info(f"Testing on a single sequence: {args.io.dataset_number}")
        filterRunner = FilterRunner(args)
        filterRunner.run()
    except FileExistsError as e:
        print(e)
    except OSError as e:
        print(e)

