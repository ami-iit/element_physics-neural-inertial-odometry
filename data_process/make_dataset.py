import numpy as np
import os
import math
from torch.utils.data import DataLoader
import config as cfg
import data_handler as daha
from calibrator import Calibrator
import maths

class CustomDataset:
    def __init__(self, data_dir, task, dofs):
        # load the raw data from the given dir
        """
        imus structure:
            - for each task 1~K
                - for each node 1~12
                    - angVel (angular velocity): (N, 1, 3)
                    - linAcc (linear acceleration): (N, 1, 3)
                    - orientation (link orientation, needs calibration): (N, 1, 4)
        joint states structure:
            - for each task 1~K
                - for each attribute (positions, velocities)
                    - numpy array of shape (N, 1, 31)
        base states structure:
            - for each task 1~K
                - for each attribute (vb, pb, wb, rb)
                    - numpy array of shape (N, 1, F)
        NOTE: dofs indicates the joint DoF in the provided data.
        """
        self.imus = np.load(os.path.join(data_dir, "imu_data.npy"), allow_pickle=True)
        self.joint_states = np.load(os.path.join(data_dir, "joint_data.npy"), allow_pickle=True)
        self.base_states = np.load(os.path.join(data_dir, "base_data.npy"), allow_pickle=True)
        self.task = task
        self.dofs = dofs

    def init_calib(self, start=None, end=None):
        r"""
        Return the ori calibration matrix for each node.
        """
        calib = Calibrator(cfg.link2imu_matrix, cfg.link2world_matrix, self.imus, self.task)
        calib.avg_raw_imu_ori(start, end)
        calib.from_imu_to_imuWorld()
        calib.from_imuWorld_to_World()
        # NOTE: calib has the structure {node_name: calib_matrix}
        self.calib_world_yaw_nodes = calib.calibrate_world_yaw()
    
    @staticmethod
    def check_length(arrs, feature, node_list):
        r"""Check if the measurements of each node has the same length."""
        lens = [arrs[node][feature].shape[0] for node in node_list]
        if len(set(lens)) > 1:
            min_length = min(lens)
            print(f"Inconsistent lenghts in {feature} arrays: {lens}.")
            print(f"Please use the minimal length: {min_length}.")
            return min_length
        else:
            same_length = min(lens)
            print(f"Each array in {feature} has the same length: {same_length}.")
            return same_length
    
    def get_features(self):
        r"""
        Get the I/O data and return them as dicts.
        Return:
            - x_lacc: (N, 3*5)
            - *x_lori: (N, 4*5), quaternion
            - y_pb: (N, 3)
            - *y_rb: (N, 3), rpy euler angles
            - y_vb: (N, 6)
            - y_s: (N, 31)
            - y_sdot: (N, 31)
        NOTE: here we have no access to link pose or twist!
        """
        # extract the single-task data
        self.task_imu_data = self.imus[self.task]
        self.task_joint_data = self.joint_states[self.task]
        self.task_base_data = self.base_states[self.task]

        # find the index of selected links
        nodes_list = [k for k, v in cfg.links.items() if v in cfg.input_links]
        print(f"min nodes list: {nodes_list}")

        lacc_length = self.check_length(self.task_imu_data, 'linAcc', nodes_list)
        self.x_lacc = np.concatenate(
            [self.task_imu_data[node]['linAcc'][:lacc_length].reshape((lacc_length, 3)) for node in nodes_list], 
            axis=-1
        )
        print(f"x lacc shape: {self.x_lacc.shape}")

        lori_length = self.check_length(self.task_imu_data, 'orientation', nodes_list)
        self.x_lori = np.concatenate(
            [self.task_imu_data[node]['orientation'][:lori_length].reshape((lori_length, 4)) for node in nodes_list], 
            axis=-1
        )
        print(f"x lori shape: {self.x_lori.shape}")

        # handle the joint states
        if self.task_joint_data['positions'].shape[-1] != self.dofs:
            raise ValueError(f"Expected joint DoF to be {self.dofs}, but got {self.task_joint_data['positions'].shape[-1]}")
        else:
            self.y_s = self.task_joint_data['positions'].reshape((-1, 31))
            print(f"Target joint position shape: {self.y_s.shape}")
        
        if self.task_joint_data['velocities'].shape[-1] != self.dofs:
            raise ValueError(f"Expected joint DoF to be {self.dofs}, but got {self.task_joint_data['velocities'].shape[-1]}")
        else:
            self.y_sdot = self.task_joint_data['velocities'].reshape((-1, 31))
            print(f"Target joint velocity shape: {self.y_sdot.shape}")

        # handle the base
        self.y_pb = self.task_base_data['base_position'].reshape((-1, 3))
        self.y_rb = self.task_base_data['base_orientation'].reshape((-1, 3))
        self.y_linear_vb = self.task_base_data['base_linear_velocity'].reshape((-1, 3))
        self.y_angular_vb = self.task_base_data['base_angular_velocity'].reshape((-1, 3))
        # concat the base velocity
        self.y_vb = np.concatenate((self.y_linear_vb, self.y_angular_vb), axis=-1)
        print(f"Base position shape: {self.y_pb.shape}| Base velocity shape: {self.y_rb.shape}| Base velocity shape: {self.y_vb.shape}")


    def calib_link_ori(self):
        r"""
        Calibrate the raw imu ori data. Return x_lori as a sequence of Rs (N, 5*9).
            - Step 1: convert the quaternion to rotation matrix.
            - Step 2: multiply the calibration matrix for each node.
        """
        # find the node name of each input link
        input_node_names = [key for key, value in cfg.links.items() if value in cfg.input_links]
        missing_nodes = [node for node in input_node_names if node not in self.calib_world_yaw_nodes]
        if missing_nodes:
            raise KeyError(f"Missing calibration matrices for nodes: {missing_nodes}")
        
        # initialize the calibrated link ori array
        n_frames = self.x_lori.shape[0]
        self.lori_calib = np.zeros((n_frames, 5*9))
        for i in range(n_frames):
            for j, node_name in enumerate(input_node_names):
                single_node_step_q = self.x_lori[i, j*4:(j+1)*4].reshape((4,))
                single_node_step_R = maths.quaternion_to_rotation_matrix(single_node_step_q)

                left_matrix = self.calib_world_yaw_nodes[node_name]
                right_matrix = cfg.link2imu_matrix[cfg.links[node_name]]

                single_node_step_R_calib = left_matrix @ single_node_step_R @ right_matrix
                self.lori_calib[i:i+1, j*9:(j+1)*9] = single_node_step_R_calib.reshape((1, 9))
        # debugging the calibrated lori shape
        print(f"Calibrated link ori shape: {self.lori_calib.shape}")


    def transform_base_ori(self):
        r"""
        Change the base orientation representation as rotation matrix.
        Return y_rb as a sequence of Rs (N, 9).
        """
        n_frames = self.y_rb.shape[0]
        self.rb_as_R = np.zeros((n_frames, 9))
        for i in range(n_frames):
            rb_step_as_euler = self.y_rb[i, :].reshape((3,))
            rb_step_as_R = maths.euler_to_rotation_matrix(rb_step_as_euler, False)
            self.rb_as_R[i:i+1, :] = rb_step_as_R.reshape((1, 9))
        print(f"Base orientation shape: {self.rb_as_R.shape}")

    def make_feature_dict(self):
        r"""
        Return I/O features as dicts.
            - x: {'lacc': (N, 3*5), 'lori': (N, 9*5)}
            - y: {'pb': (N, 3), 'rb': (N, 9), 'vb': (N, 6), 's': (N, 31), 'sdot': (N, 31)}
        """
        self.x = {
            'lacc': self.x_lacc,
            'lori': self.lori_calib
        }
        self.y = {
            'pb': self.y_pb,
            'rb': self.rb_as_R,
            'vb': self.y_vb,
            's': self.y_s,
            'sdot': self.y_sdot
        }
        print(f"I/O feature dicts are ready!")

    @staticmethod
    def split(ds, ratio):
        r"""Split the data into training and validation sets."""
        if not (0 < ratio < 1):
            raise ValueError(f"The splitting ratio must be between 0 and 1.")
        train_set, val_set = {}, {}
        for key, data in ds.items():
            split_index = math.floor(data.shape[0]*ratio)
            data_train = data[:split_index, :]
            data_val = data[split_index:, :]

            train_set[key] = data_train
            val_set[key] = data_val
        return train_set, val_set

    def get_splitted_datasets(self, ratio):
        r"""Return the splitted train/val datasets."""
        self.x_train, self.x_val = self.split(self.x, ratio)
        self.y_train, self.y_val = self.split(self.y, ratio)


    def generate(self, window_size, stride, output_steps):
        r"""Generate sliding windows through the whole sequence."""
        print(f"Preparing training windows...")
        self.train_windows = daha.DataHandler(
            x=self.x_train,
            y=self.y_train,
            window_size=window_size,
            stride=stride,
            output_steps=output_steps
        )
        print(f"Preparing evaluation windows...")
        self.val_windows = daha.DataHandler(
            x=self.x_val,
            y=self.y_val,
            window_size=window_size,
            stride=stride,
            output_steps=output_steps
        )

    def iterate(self, bs):
        r"""Return mini-batched iterators from all the sliding windows."""
        train_iterator = DataLoader(
            self.train_windows,
            batch_size=bs,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        
        val_iterator = DataLoader(
            self.val_windows,
            batch_size=bs,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        return train_iterator, val_iterator