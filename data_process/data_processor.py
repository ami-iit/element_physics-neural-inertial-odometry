r"""Preprocess the iFeel data."""
import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.io.matlab import mat_struct
from scipy.io.matlab import MatReadError
import scipy.io as io
import h5py
from pathlib import Path
import pickle
import config as cfg


class DataProcessor:
    def __init__(self, dir):
        self.dir = dir
  
    def _check_keys(self, dict):
        r"""
        Chect if entries in dictionary are mat-objects. 
        If yes, _todict is called to change them to nested dictionaries.
        """
        for key in dict:
            if isinstance(dict[key], mat_struct):
                dict[key] = self._todict(dict[key])
        return dict
    
    def _todict(self, mat_obj):
        """
        A recursive function which constructs nested dictionaries from mat-objects
        """
        dict = {}
        for strg in mat_obj._fieldnames:
            elem = mat_obj.__dict__[strg]
            if isinstance(elem, mat_struct):
                dict[strg] = self._todict(elem)
            else:
                dict[strg] = elem
        return dict
            
    def _open(self, data_file):
        r"""Open a mat file, covering both v5 and v7.3 formats."""
        if not data_file.endswith('.mat'):
            raise ValueError(f"File {data_file} is not a mat file.")
        
        try:
            with open(join(self.dir, data_file), 'rb') as f:
                header = f.read(128)
            if header[:4] == b'MATL':
                mat_data = h5py.File(join(self.dir, data_file), 'r')
                print(f"Open {data_file} as HDF5-based MATLAB v7.3 format file.")
            return mat_data
        except (OSError, MatReadError):
                raise ValueError(f"{data_file} not a valid .mat file or unsupported format.")

    def read(self):
        r"""Read the full mat file."""
        self.full_dicts = {}
        for file in os.listdir(self.dir):
            self.file = file
            mat_data = self._open(self.file)
           
            mat_data_keys = sorted(mat_data.keys())
            full_data = self._check_keys(mat_data[mat_data_keys[1]])
            self.full_dicts[Path(self.file).stem[9:]] = full_data
        print(f"Full dicts: {self.full_dicts.keys()}")
    
    def get_imus(self, tasks=None, nodes=None, attris=None):
        r"""Retrieve the imu data of the selected nodes."""
        self.imus = {}
        if not hasattr(self, "full_dicts"):
            raise AttributeError("Not found full mat data!")
        
        if nodes is None:
            nodes = [f'node{i}' for i in range(1, 13)]
            print(f"Retrieve all nodes: {nodes}")

        if tasks is None:
            tasks = self.full_dicts.keys()
            print(f'Retrieve all tasks: {tasks}')
        
        for task in tasks:
            single_task_imu = {}
            for node in nodes:
                node_data = self.full_dicts[task][node]
                #print(f"Node {node} data keys: {node_data.keys()}")
                single_imu = {attri: np.array(node_data[attri]['data']) for attri in attris}
                single_task_imu[node] = single_imu
            self.imus[task] = single_task_imu
        # Debugging structure of the final dictionary
        task_example, node_example = tasks[0], nodes[0]
        print(f"IMUs: {list(self.imus.keys())} \n"
              f"Single task: {list(self.imus[task_example].keys())} \n"
              f"Single node: {list(self.imus[task_example][node_example].keys())}")
        return self.imus
    
    def get_states(
            self, tasks=None, 
            attris_joint=None, attris_base=None, attris_dynamics=None,
            is_joint=True, is_base=True, is_dynamics=False
        ):
        if not hasattr(self, "full_dicts"):
            raise AttributeError("Not found full mat data!")
        if tasks is None:
            tasks = self.full_dicts.keys()
            print(f'Retrieve all tasks: {tasks}')

        self.joint_states, self.base, self.dynamics = {}, {}, {}
        for task in tasks:
            joint_state_data = self.full_dicts[task]['joints_state']
            base_data = self.full_dicts[task]['human_state']
            dynamics_data = {
                'joint_torques': self.full_dicts[task]['human_dynamics']['joint_torques'],
                'wrenches': self.full_dicts[task]['human_wrench']['wrenches']
            }
            if is_joint:
                single_task_state = {attri: np.array(joint_state_data[attri]['data']) for attri in attris_joint}
                self.joint_states[task] = single_task_state
            if is_base:
                single_task_base = {attri: np.array(base_data[attri]['data']) for attri in attris_base}
                self.base[task] = single_task_base
            if is_dynamics:
                single_task_dynamics = {attri: np.array(dynamics_data[attri]['data']) for attri in attris_dynamics}
                self.dynamics[task] = single_task_dynamics
        return self.joint_states, self.base, self.dynamics

def save_data(data, dir, filename):
    r"""Save the retrieved data."""
    file_path = join(dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # config
    ifeel_dir = "./data/PredictionDataset_03_03_25"
    save_data_dir = "./data/processed"

    # load the data
    data_processor = DataProcessor(ifeel_dir)
    data_processor.read()
    imus = data_processor.get_imus(cfg.tasks, cfg.nodes, cfg.imu_attris)
    joint_states, base_states, _ = data_processor.get_states(
                                                    tasks=cfg.tasks, 
                                                    attris_joint=cfg.joint_attris,
                                                    attris_base=cfg.base_attris,
                                                    attris_dynamics=cfg.dynamics_attris,
                                                    is_joint=True, is_base=True, is_dynamics=False)
    
    # save the data
    save_data(imus, save_data_dir, "imu_data.npy")
    save_data(joint_states, save_data_dir, "joint_data.npy")
    save_data(base_states, save_data_dir, "base_data.npy")
    print(f"Successfully saved all data to {save_data_dir}.")

    
    







