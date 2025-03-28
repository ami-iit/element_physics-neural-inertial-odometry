import numpy as np
from scipy.spatial.transform import Rotation as rot

import config as cfg
import maths

class Calibrator:
    def __init__(self, link2imu, link2world, imus, task):
        # calibration matrix from link to imu
        self.link2imu = link2imu
        self.link2world = link2world
        self.imus = imus
        self.task = task

    def avg_raw_imu_ori(self, start, end):
        r"""Compute avg of ori from raw IMU data."""
        # get a seq of imu ori for each node
        self.avg_imu_ori_nodes = {}
        for _, node in enumerate(cfg.links):
            print(f"processing node: {node}")
            imu_ori_single_node = self.imus[self.task][node][cfg.imu_attris[-1]][start:end].reshape((end-start, 4))
            # compute the mean along rows
            mean_imu_ori_single_node = np.mean(imu_ori_single_node, axis=0, keepdims=True)
            print(f"avg ori as q: {mean_imu_ori_single_node}")
            self.avg_imu_ori_nodes[node] = mean_imu_ori_single_node
        print(f"Check single node avg ori: \n"
              f"shape: {self.avg_imu_ori_nodes['node1'].shape} \n"
              f"value: {self.avg_imu_ori_nodes['node1']}")
    
    def from_imu_to_imuWorld(self):
        r"""Get rotation matrix from avg ori (as q)"""
        self.avg_imu_R_nodes = {}
        for _, key in enumerate(cfg.links):
            avg_imu_q = self.avg_imu_ori_nodes[key].reshape((4,))
            if avg_imu_q.shape != (4,):
                raise ValueError(f"Expected quaternion shape (4,), but got {avg_imu_q.shape}")
            avg_imu_q_new = np.array([avg_imu_q[1], avg_imu_q[2], avg_imu_q[3], avg_imu_q[0]])
            self.avg_imu_R_nodes[key] = rot.from_quat(avg_imu_q_new).as_matrix()
        print(f"Check single node avg rotation matrix: \n"
              f"shape: {self.avg_imu_R_nodes['node1'].shape} \n"
              f"value: {self.avg_imu_R_nodes['node1']}")
        
    def from_imuWorld_to_World(self):
        r"""Compute the transformation matrix from imu_world to world for each node."""
        self.imuWorld_to_World_nodes = {}
        for _, key in enumerate(cfg.links):
            link2imu_matrix_inv = np.linalg.inv(self.link2imu[cfg.links[key]])
            imu2imuWorld_matrix_inv = np.linalg.inv(self.avg_imu_R_nodes[key])
            link2world_matrix = self.link2world[cfg.links[key]]
            transformation_matrix = imu2imuWorld_matrix_inv @ link2imu_matrix_inv @ link2world_matrix
            self.imuWorld_to_World_nodes[key] = transformation_matrix
        print("Finish computing transformation matrix for each node.")

    def calibrate_world_yaw(self):
        r"""Set the calibration matrix of the orientation to the offset in yaw only."""
        self.calib_matrix_WorldYaw_nodes = {}
        for _, key in enumerate(cfg.links):
            # convert the transformation matrix of each node into rpy angles
            calib_matrix = self.imuWorld_to_World_nodes[key]
            calib_rpy = maths.rotation_matrix_to_euler(calib_matrix)
            # apply the yaw calibration
            calib_rpy_new = np.array([0, 0, calib_rpy[-1]])
            # convert the rpy angles back to rotation matrix
            calib_matrix_yaw = maths.euler_to_rotation_matrix(calib_rpy_new, True)
            self.calib_matrix_WorldYaw_nodes[key] = calib_matrix_yaw
        print(f"Calibrate all nodes to the offset in World Yaw.")
        return self.calib_matrix_WorldYaw_nodes
