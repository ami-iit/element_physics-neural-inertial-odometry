from numba import jit
import numpy as np
import warnings

def hat(v):
    v = np.squeeze(v)
    R = np.array([
        [0, -v[2], v[1]], 
        [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]
    ])
    return R

@jit(nopython=True, parallel=False, cache=True)
def exponential_SO3_operation(omega):
    """
    Computes the matrix exponential of a 3D vector omega in so(3),
    returning a 3x3 rotation matrix in SO(3) using Rodrigues' formula.

    Parameters:
    omega (numpy.ndarray): A 3-element vector representing the axis-angle form.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix corresponding to the given omega.
    """
    if omega.shape != (3,):  # More robust check
        raise ValueError("Input must be a 3D vector with shape (3,)")
    def hat(v):
        """Computes the skew-symmetric matrix (hat operator) of a 3D vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]
                        ])
    angle = np.linalg.norm(omega)  # Compute rotation angle (magnitude of omega)
    if angle < 1e-10:  # Handle small angles with first-order approximation
        return np.identity(3) + hat(omega)  # Taylor expansion approximation
    
    axis = omega / angle  # Normalize omega to get unit rotation axis
    s, c = np.sin(angle), np.cos(angle)  # Compute sine and cosine of angle
    
    # Rodrigues' formula for the rotation matrix
    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)

# vectorize the function of exponential_SO3_operation
vec_exponential_SO3_operation = np.vectorize(exponential_SO3_operation, signature="(3)->(3,3)")


@jit(nopython=True, parallel=False, cache=True)
def exponential_SO3_right_jacob(omega):
    """
    Computes the right Jacobian of the SO(3) exponential map.

    Parameters:
    omega (numpy.ndarray): A 3-element vector representing an angular velocity in so(3).

    Returns:
    numpy.ndarray: A 3x3 right Jacobian matrix.
    """
    def hat(v):
        """Computes the skew-symmetric matrix (hat operator) of a 3D vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]
                        ])
    angle = np.linalg.norm(omega)  # Compute rotation angle (magnitude of omega)
    omega_hat = hat(omega)
    if angle < 1e-3: # Use Taylor series approximation for small angles
        J = np.eye(3) - 0.5 * omega_hat + (1.0 / 6.0) * (omega_hat @ omega_hat)
    else:
        angle_sq = angle ** 2 # angle squared
        angle_cub = angle ** 3 # angle cubed
        # compute the right Jacobian using the exact formula
        J = (
            np.eye(3)
            - ((1 - np.cos(angle)) / angle_sq) * omega_hat
            + ((angle - np.sin(angle)) / angle_cub) * (omega_hat @ omega_hat)
        )
    return J

@jit(nopython=True, parallel=False, cache=True)
def rot_2vec(a, b):
    """
    Computes the rotation matrix that aligns vector 'a' to vector 'b'.
    
    Parameters:
    a (numpy.ndarray): A 3x1 column vector.
    b (numpy.ndarray): A 3x1 column vector.
    
    Returns:
    numpy.ndarray: A 3x3 rotation matrix R such that R @ a = b (approximately).
    """
    # Ensure input vectors have the correct shape
    if a.shape != (3, 1) or b.shape != (3, 1):
        raise ValueError("Both input vectors must have shape (3,1)")
    
    def hat(v):
        """Computes the skew-symmetric matrix (hat operator) of a 3D vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]
                        ])
    
    # Normalize the input vectors to unit length
    a_n = np.linalg.norm(a)
    b_n = np.linalg.norm(b)
    if a_n == 0 or b_n == 0:
        raise ValueError("Input vectors must be non-zero")
    
    a_hat = a / a_n
    b_hat = b / b_n
    
    # Compute the rotation axis using the cross product
    omega = np.cross(a_hat.T, b_hat.T).T  # Ensure correct orientation
    
    # Compute the cosine of the angle between the vectors
    cos_theta = np.dot(a_hat.T, b_hat).item()
    
    # Handle edge cases: parallel or anti-parallel vectors
    if np.isclose(cos_theta, 1.0):
        return np.eye(3)  # Identity matrix (no rotation needed)
    elif np.isclose(cos_theta, -1.0):
        # 180-degree rotation: find an orthogonal vector for the axis
        perp = np.array([[1], [0], [0]]) if not np.isclose(a_hat[0, 0], 1.0) else np.array([[0], [1], [0]])
        omega = np.cross(a_hat.T, perp.T).T  # Compute a perpendicular axis
        omega /= np.linalg.norm(omega)  # Normalize axis
        return np.eye(3) + 2 * hat(omega) @ hat(omega)  # Rodrigues' formula for 180°
    
    # Compute the Rodrigues' rotation formula components
    c = 1.0 / (1 + cos_theta)  # Avoid division by zero
    R_ba = np.eye(3) + hat(omega) + c * hat(omega) @ hat(omega)
    
    return R_ba
 
def cal_q_from_R(R):
    """
    Return a quaternion q of order (x,y,z,w) from a given rotation matrix R.
    """
    is_single = False
    R = np.asarray(R, dtype=float)

    if R.ndim not in [2, 3] or R.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected `R` to have shape (3, 3) or "
            "(N, 3, 3), got {}".format(R.shape)
        )
    # If a single R is given, convert it to 3D 1 x 3 x 3 R but
    # set self._single to True so that we can return appropriate objects in
    # the `to_...` methods
    if R.shape == (3, 3):
        R = R.reshape((1, 3, 3))
        is_single = True
    num_rotations = R.shape[0]

    decision_R = np.empty((num_rotations, 4))
    decision_R[:, :3] = R.diagonal(axis1=1, axis2=2)
    decision_R[:, -1] = decision_R[:, :3].sum(axis=1)
    choices = decision_R.argmax(axis=1)

    quat = np.empty((num_rotations, 4))

    ind = np.nonzero(choices != 3)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3

    quat[ind, i] = 1 - decision_R[ind, -1] + 2 * R[ind, i, i]
    quat[ind, j] = R[ind, j, i] + R[ind, i, j]
    quat[ind, k] = R[ind, k, i] + R[ind, i, k]
    quat[ind, 3] = R[ind, k, j] - R[ind, j, k]

    ind = np.nonzero(choices == 3)[0]
    quat[ind, 0] = R[ind, 2, 1] - R[ind, 1, 2]
    quat[ind, 1] = R[ind, 0, 2] - R[ind, 2, 0]
    quat[ind, 2] = R[ind, 1, 0] - R[ind, 0, 1]
    quat[ind, 3] = 1 + decision_R[ind, -1]

    quat /= np.linalg.norm(quat, axis=1)[:, None]

    if is_single:
        return quat[0]
    else:
        return quat

_AXIS_TO_IND = {"x": 0, "y": 1, "z": 2}


def _elementary_basis_vector(axis):
    b = np.zeros(3)
    b[_AXIS_TO_IND[axis]] = 1
    return b


def compute_euler_from_matrix(matrix, seq, extrinsic=False):
    """
    The algorithm assumes intrinsic frame transformations. 
    The algorithm in the paper is formulated for rotation matrices which are transposition
    rotation matrices used within Rotation.
    Adapt the algorithm for our case by
    1. Instead of transposing our representation, use the transpose of the
       O matrix as defined in the paper, and be careful to swap indices
    2. Reversing both axis sequence and angles for extrinsic rotations
    """
    if extrinsic:
        seq = seq[::-1]

    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
    num_rotations = matrix.shape[0]

    # Step 0
    # Algorithm assumes axes as column vectors, here we use 1D vectors
    n1 = _elementary_basis_vector(seq[0])
    n2 = _elementary_basis_vector(seq[1])
    n3 = _elementary_basis_vector(seq[2])

    # Step 2
    sl = np.dot(np.cross(n1, n2), n3)
    cl = np.dot(n1, n3)

    # angle offset is lambda from the paper referenced in [2] from docstring of
    # `as_euler` function
    offset = np.arctan2(sl, cl)
    c = np.vstack((n2, np.cross(n1, n2), n1))

    # Step 3
    rot = np.array([[1, 0, 0], [0, cl, sl], [0, -sl, cl]])
    res = np.einsum("...ij,...jk->...ik", c, matrix)
    matrix_transformed = np.einsum("...ij,...jk->...ik", res, c.T.dot(rot))

    # Step 4
    angles = np.empty((num_rotations, 3))
    # Ensure less than unit norm
    positive_unity = matrix_transformed[:, 2, 2] > 1
    negative_unity = matrix_transformed[:, 2, 2] < -1
    matrix_transformed[positive_unity, 2, 2] = 1
    matrix_transformed[negative_unity, 2, 2] = -1
    angles[:, 1] = np.arccos(matrix_transformed[:, 2, 2])

    # Steps 5, 6
    eps = 1e-7
    safe1 = np.abs(angles[:, 1]) >= eps
    safe2 = np.abs(angles[:, 1] - np.pi) >= eps

    # Step 4 (Completion)
    angles[:, 1] += offset

    # 5b
    safe_mask = np.logical_and(safe1, safe2)
    angles[safe_mask, 0] = np.arctan2(
        matrix_transformed[safe_mask, 0, 2], -matrix_transformed[safe_mask, 1, 2]
    )
    angles[safe_mask, 2] = np.arctan2(
        matrix_transformed[safe_mask, 2, 0], matrix_transformed[safe_mask, 2, 1]
    )

    if extrinsic:
        # For extrinsic, set first angle to zero so that after reversal we
        # ensure that third angle is zero
        # 6a
        angles[~safe_mask, 0] = 0
        # 6b
        angles[~safe1, 2] = np.arctan2(
            matrix_transformed[~safe1, 1, 0] - matrix_transformed[~safe1, 0, 1],
            matrix_transformed[~safe1, 0, 0] + matrix_transformed[~safe1, 1, 1],
        )
        # 6c
        angles[~safe2, 2] = -(
            np.arctan2(
                matrix_transformed[~safe2, 1, 0] + matrix_transformed[~safe2, 0, 1],
                matrix_transformed[~safe2, 0, 0] - matrix_transformed[~safe2, 1, 1],
            )
        )
    else:
        # For instrinsic, set third angle to zero
        # 6a
        angles[~safe_mask, 2] = 0
        # 6b
        angles[~safe1, 0] = np.arctan2(
            matrix_transformed[~safe1, 1, 0] - matrix_transformed[~safe1, 0, 1],
            matrix_transformed[~safe1, 0, 0] + matrix_transformed[~safe1, 1, 1],
        )
        # 6c
        angles[~safe2, 0] = np.arctan2(
            matrix_transformed[~safe2, 1, 0] + matrix_transformed[~safe2, 0, 1],
            matrix_transformed[~safe2, 0, 0] - matrix_transformed[~safe2, 1, 1],
        )

    # Step 7
    if seq[0] == seq[2]:
        # lambda = 0, so we can only ensure angle2 -> [0, pi]
        adjust_mask = np.logical_or(angles[:, 1] < 0, angles[:, 1] > np.pi)
    else:
        # lambda = + or - pi/2, so we can ensure angle2 -> [-pi/2, pi/2]
        adjust_mask = np.logical_or(angles[:, 1] < -np.pi / 2, angles[:, 1] > np.pi / 2)

    # Dont adjust gimbal locked angle sequences
    adjust_mask = np.logical_and(adjust_mask, safe_mask)

    angles[adjust_mask, 0] += np.pi
    angles[adjust_mask, 1] = 2 * offset - angles[adjust_mask, 1]
    angles[adjust_mask, 2] -= np.pi

    angles[angles < -np.pi] += 2 * np.pi
    angles[angles > np.pi] -= 2 * np.pi

    # Step 8
    if not np.all(safe_mask):
        warnings.warn(
            "Gimbal lock detected. Setting third angle to zero since"
            " it is not possible to uniquely determine all angles."
        )
    # Reverse role of extrinsic and intrinsic rotations, but let third angle be
    # zero for gimbal locked cases
    if extrinsic:
        angles = angles[:, ::-1]
    return angles


def logarithm_map_R(R):
    """
    Compute the logarithm map of a rotation matrix R.
    Return a Lie algebra (so(3)) representation, a 3D rotation vector.
    """
    # compute the quaternion as xyzw
    q = cal_q_from_R(R)

    # extract the scalar part and the vector part
    q_w = q[-1]
    q_vec = q[:3]

    # compute the norm of the vector part
    q_vec_norm = np.linalg.norm(q_vec)

    # small threshold to prevent numerical instability
    epsilon = 1e-7

    # compute the logarithm map
    if q_vec_norm < epsilon:
        # when the q norm is very small, use a first-order approximation
        q_w_sq = q_w * q_w
        q_vec_norm_sq = q_vec_norm * q_vec_norm
        # approximation for small angles
        atn = 2.0/q_w - (2.0*q_vec_norm_sq) / (q_w*q_w_sq)
    else:
        if np.abs(q_w) < epsilon:
            # handle the special case when q_w is close to zero
            # this occurs when the rotation is close to 180 degrees
            atn = np.pi / q_vec_norm if q_w > 0 else -np.pi / q_vec_norm
        else:
            # general case: use the atan2 function for better numerical stability
            atn = 2.0 * np.arctan2(q_vec_norm, q_w) / q_vec_norm
    # compute the final rotation vector (axis-angle)
    tangent = atn * q_vec
    return tangent

def unwrap_rpy(rpys):
    """
    Unwrap RPY angles to ensure smooth transitions 
    when angles wrap around due to periodicity.
    Args:
        - rpys [N, 3] array: each row contains roll, pitch, yaw angles.
    Return:
        - a modified rpys array where discontinuities due to angle wrapping are removed.
    """
    # compute element-wise difference between consecutive rows
    # to detect sudden change in the angles
    diff = rpys[1:, :] - rpys[:-1, :]

    # initialize unwrapped array
    uw_rpys = np.zeros(rpys.shape)
    uw_rpys[0, :] = rpys[0, :] # the first row is copied

    # correct large jumps
    diff[diff > 300] = diff[diff > 300] - 360
    diff[diff < -300] = diff[diff < -300] + 360
    uw_rpys[1:, :] = uw_rpys[0, :] = np.cumsum(diff, axis=0)

    return uw_rpys


def wrap_rpy(uw_rpys, radians=False):
    """Inverse operation of unwrap rpy."""
    bound = np.pi if radians else 180
    rpys = uw_rpys
    while rpys/min() < -bound:
        rpys[rpys < bound] = rpys[rpys < -bound] + 2*bound
    while rpys.max() >= bound:
        rpys[rpys >= bound] = rpys[rpys >= bound] - 2*bound
    return rpys

def inv_SE3(T):
    """Return the inverse of a 4x4 transformation matrix."""
    invT = np.eye(4)
    invT[:3, :3] = T[:3, :3].T
    invT[:3, 3:4] = -T[:3, :3].T @ T[:3, 3:4]
    return invT