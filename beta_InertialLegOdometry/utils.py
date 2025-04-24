import numpy as np

def extend_joint_state_preds(s, sdot, old_joint_list, new_joint_list):
    r"""Return the extended joint predictions."""
    new_dof = len(new_joint_list)
    s_new, sdot_new = np.zeros(new_dof, ), np.zeros(new_dof, )
    joint_map = {joint: i for i, joint in enumerate(new_joint_list)}
    for i, joint in enumerate(old_joint_list):
        joint_index = joint_map[joint]
        s_new[joint_index] = s[i]
        sdot_new[joint_index] = sdot[i]
    return s_new, sdot_new