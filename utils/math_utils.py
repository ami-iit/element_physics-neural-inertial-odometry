from numba import jit
import numpy as np

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
 