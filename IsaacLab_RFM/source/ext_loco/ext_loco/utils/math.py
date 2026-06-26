import torch
import math


# math utils
def generate_sigmoid_scale(mu: float, decay_length: float, x: torch.Tensor):
    sigmoid_z = 5 / decay_length * (x - mu)
    return torch.sigmoid(sigmoid_z)


def relexed_barrier_func(x: torch.Tensor, mu: float = 0.05, delta: float = 0.1) -> torch.Tensor:
    """
    Compute the relaxed barrier function for the input tensor x.

    Args:
        x: Input tensor.
        eps: Small positive number.

    Returns:
        Relaxed barrier function.
    """
    if mu < 0 or delta < 0:
        raise ValueError("mu and delta must be non-negative.")
    result = torch.zeros_like(x)
    result[x > delta] = -mu * torch.log(x[x > delta])
    result[x <= delta] = mu / 2 * (((x[x <= delta] - 2 * delta) / delta) ** 2 - 1) - mu * math.log(delta)

    return result


# def compute_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
#     """Convert rotations given as quaternions to axis/angle.

#     Args:
#         quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
#         eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

#     Returns:
#         Rotations given as a vector in axis angle form. Shape is (..., 3).
#         The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.


#     Reference:
#         https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
#     """
#     # Modified to take in quat as [q_w, q_x, q_y, q_z]
#     # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
#     # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
#     # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
#     # When theta = 0, (sin(theta/2) / theta) is undefined
#     # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
#     assert (quat[..., 0:1] > 0).all()
#     # quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
#     mag = torch.norm(quat[..., 1:], dim=-1)
#     half_angle = torch.atan2(mag, quat[..., 0])
#     angle = 2.0 * half_angle
#     # check whether to apply Taylor approximation
#     # sin_half_angles_over_angles = torch.where(
#     #     angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
#     # )
#     assert (angle > 0.0).all(), "Angle must be positive."
#     return angle


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def compute_rotation_distance(input_quat, target_quat):
    Ee_target_R = quaternion_to_matrix(target_quat)
    Ee_R = quaternion_to_matrix(input_quat)

    # Calculate the rotation distance (Frobenius norm of log(R1^T * R2))

    R_rel = torch.matmul(torch.transpose(Ee_target_R, 1, 2), Ee_R)
    trace_R_rel = torch.einsum("bii->b", R_rel)

    # Clamping to avoid numerical issues with arccos
    trace_clamped = torch.clamp((trace_R_rel - 1) / 2, -1.0, 1.0)
    rotation_distance = torch.acos(trace_clamped)
    return rotation_distance

def command_duration_mask(time_left, duration):
    return time_left <= duration

