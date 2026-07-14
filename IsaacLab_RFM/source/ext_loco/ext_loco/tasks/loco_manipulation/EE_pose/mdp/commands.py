# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
import pickle
import numpy as np
from collections.abc import Sequence
from typing import Optional, Tuple
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_inv,
    quat_mul,
    quat_from_euler_xyz,
    quat_unique,
    quat_rotate_inverse,
    sample_uniform,
    quat_from_matrix,
)

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.envs.mdp import UniformPoseCommand
from isaaclab.envs import ManagerBasedEnv

from ext_loco.utils.math import generate_sigmoid_scale
from ext_loco.utils.math import compute_rotation_distance


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to rotation matrix using Rodrigues' formula.

    Args:
        axis_angle: Tensor of shape (..., 3) representing rotation axis * angle

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    angle = torch.norm(axis_angle, dim=-1)
    small_angle_mask = angle < 1e-8
    axis = axis_angle / (angle.unsqueeze(-1) + 1e-8)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    output_shape = list(axis_angle.shape[:-1]) + [3, 3]
    K = torch.zeros(output_shape, device=axis_angle.device, dtype=axis_angle.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    I = I.view(*([1] * (len(output_shape) - 2)), 3, 3).expand(output_shape)
    K_squared = torch.matmul(K, K)
    cos_angle = cos_angle.view(*output_shape[:-2], 1, 1)
    sin_angle = sin_angle.view(*output_shape[:-2], 1, 1)
    R = I + sin_angle * K + (1 - cos_angle) * K_squared
    R[small_angle_mask] = I[small_angle_mask]
    return R


class UniformWorldPoseCommand(UniformPoseCommand):
    """Command generator for pose commands in world frame, sampled uniformly.

    Positions are sampled uniformly in cartesian space relative to the robot root.
    Orientations are sampled from euler angles (roll-pitch-yaw) converted to quaternions (w, x, y, z).

    Note:
        Sampling euler angles uniformly is not equivalent to sampling rotations uniformly on SO(3).
    """

    def __init__(self, cfg: UniformWorldPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.decrease_vel = torch.zeros(self.num_envs, device=self.device)
        self.se3_distance_ref = torch.ones(self.num_envs, device=self.device) * 5.0
        self.decrease_vel_range = cfg.se3_decrease_vel_range
        self._env._loco_mani_scale = torch.ones(self.num_envs, device=self.device)  # type: ignore
        self.resampling_time_scale = cfg.resampling_time_scale
        self.resample_time_range = cfg.resampling_time_range
        self.optim_pos_distance = torch.zeros(self.num_envs, device=self.device)
        self.optim_orient_distance = torch.zeros(self.num_envs, device=self.device)
        self.pos_improvement = torch.zeros(self.num_envs, device=self.device)
        self.orient_improvement = torch.zeros(self.num_envs, device=self.device)
        # +1 means approach with the base front; -1 means approach in reverse.
        # The mode is fixed when a command is sampled so turning 180 degrees
        # cannot change a rear target into a forward-walking target.
        self.travel_direction = torch.ones(self.num_envs, device=self.device)

    def _refresh_pose_command_b(self):
        """Refresh pose_command_b by transforming world-frame command into robot base frame."""
        self.pose_command_b[:, :3] = quat_rotate_inverse(
            self.robot.data.root_link_quat_w,
            self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w,
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )

    def _set_travel_direction(self, env_ids: Sequence[int]):
        """Choose forward for the initial front half-plane and reverse for the rear."""
        self._refresh_pose_command_b()
        self.travel_direction[env_ids] = torch.where(
            self.pose_command_b[env_ids, 0] >= 0.0,
            1.0,
            -1.0,
        )

    def _update_metrics(self):
        self._refresh_pose_command_b()

        pos_error = self.pose_command_w[:, :3] - self.robot.data.body_link_state_w[:, self.body_idx, :3]
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.body_link_state_w[:, self.body_idx, 3:7]),
            self.pose_command_w[:, 3:],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = rot_error_angle

        self.se3_distance_ref -= self.decrease_vel * self._env.step_dt
        self.se3_distance_ref = torch.clamp(self.se3_distance_ref, min=0.0)
        self.pos_improvement = (self.optim_pos_distance - self.metrics["position_error"]).clip(min=0.0)
        self.orient_improvement = (self.optim_orient_distance - self.metrics["orientation_error"]).clip(min=0.0)
        self.optim_pos_distance[:] = torch.minimum(self.metrics["position_error"], self.optim_pos_distance)
        self.optim_orient_distance[:] = torch.minimum(self.metrics["orientation_error"], self.optim_orient_distance)

        self._env._loco_mani_scale = generate_sigmoid_scale(  # type: ignore
            mu=1.0, decay_length=1.0, x=self.se3_distance_ref
        )

    def _update_se3_ref(self, env_ids: Sequence[int]):
        self._refresh_pose_command_b()

        pos_error = self.pose_command_w[:, :3] - self.robot.data.body_link_state_w[:, self.body_idx, :3]
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.body_link_state_w[:, self.body_idx, 3:7]),
            self.pose_command_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = rot_error_angle

        self.se3_distance_ref[env_ids] = (
            2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
        )
        self.optim_pos_distance[env_ids] = self.metrics["position_error"][env_ids]
        self.optim_orient_distance[env_ids] = self.metrics["orientation_error"][env_ids]
        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = self.robot.data.root_link_pos_w[env_ids, 0] + r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = self.robot.data.root_link_pos_w[env_ids, 1] + r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.decrease_vel[env_ids] = sample_uniform(
            self.decrease_vel_range[0], self.decrease_vel_range[1], len(env_ids), device=self.device
        )
        self._set_travel_direction(env_ids)

    def _resample(self, env_ids):
        if len(env_ids) != 0:
            self._resample_command(env_ids)
            self._update_se3_ref(env_ids)
            se3_error = 2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
            random_scale = sample_uniform(
                self.resampling_time_scale[0], self.resampling_time_scale[1], len(env_ids), device=self.device
            )
            self.time_left[env_ids] = (se3_error * random_scale).clip(
                min=self.resample_time_range[0], max=self.resample_time_range[1]
            )
            self.command_counter[env_ids] += 1


@configclass
class UniformWorldPoseCommandCfg(UniformPoseCommandCfg):
    class_type: type = UniformWorldPoseCommand
    se3_decrease_vel_range: tuple[float, float] = (0.5, 1.4)
    resampling_time_scale: tuple[float, float] = (6.0, 15.0)


class PicklePoseSequenceCommand(UniformWorldPoseCommand):
    """Command generator that plays back EE pose sequences loaded from a pickle file.

    Includes built-in latency simulation: maintains a rolling history of EE poses
    and exposes a time-delayed pose via ``get_delayed_ee_pose()``.
    """

    def __init__(self, cfg: PicklePoseSequenceCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg

        # --- Latency buffer ---
        self.latency_seconds = cfg.pose_latency
        self.history_max_len = cfg.history_buffer_length
        # shape: (num_envs, history_len, 7)  [x, y, z, qw, qx, qy, qz]
        self.ee_pose_history = torch.zeros(self.num_envs, self.history_max_len, 7, device=self.device)
        self.ee_pose_history[..., 3] = 1.0  # identity quaternion (w=1)

        # --- Load pickle ---
        with open(cfg.file_path, "rb") as f:
            episodes = pickle.load(f)

        self.ee_pos = torch.from_numpy(
            np.stack([episode["ee_pos"] for episode in episodes], axis=0)
        ).to(self.device).float()

        ee_axis_angle = torch.from_numpy(
            np.stack([episode["ee_axis_angle"] for episode in episodes], axis=0)
        ).to(self.device).float()
        self.ee_rot_mat = axis_angle_to_matrix(ee_axis_angle)

        self.planar_center = cfg.planar_center
        self.added_random_height_range = cfg.add_random_height_range
        self.episode_length_s = cfg.episode_length_s

        cfg_sim = getattr(getattr(self._env, "cfg", None), "sim", None)
        self.sim_dt = getattr(cfg_sim, "dt", None) if cfg_sim is not None else None
        if self.sim_dt is None:
            raise AttributeError("sim.dt is not defined in env.cfg.sim, cannot align with UMI's 0.005s baseline")
        self.episode_length = int(self.episode_length_s / self.sim_dt)

        self.current_episode_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_start_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.added_heights = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.command_origin = torch.zeros(self.num_envs, 3, device=self.device)

        self.tip_offset_pos = torch.tensor(cfg.tip_offset_pos, device=self.device, dtype=torch.float)
        tip_quat = quat_from_euler_xyz(
            torch.tensor(cfg.tip_offset_rpy[0], device=self.device),
            torch.tensor(cfg.tip_offset_rpy[1], device=self.device),
            torch.tensor(cfg.tip_offset_rpy[2], device=self.device),
        )
        w, x, y, z = tip_quat
        self.tip_offset_rot_mat = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            device=self.device,
            dtype=torch.float,
        )
        self.tip_offset_rot_inv = self.tip_offset_rot_mat.transpose(0, 1)
        self.tip_offset_pos_inv = -self.tip_offset_rot_inv @ self.tip_offset_pos

        # Flag: on the first compute() after reset, update command_origin to actual EE position
        self._need_origin_update = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.planar_center:
            start_means = self.ee_pos[:, [1, 2, 3], :2].mean(dim=1, keepdim=True)
            self.ee_pos[..., :2] -= start_means

        self._pad_sequences()

    def _pad_sequences(self):
        """Pad sequences by repeating the last pose to fill episode_length."""
        if self.ee_pos.shape[1] < self.episode_length:
            padding_len = self.episode_length - self.ee_pos.shape[1]
            self.ee_pos = torch.cat([self.ee_pos, self.ee_pos[:, -1:, :].expand(-1, padding_len, -1)], dim=1)
            self.ee_rot_mat = torch.cat(
                [self.ee_rot_mat, self.ee_rot_mat[:, -1:, :, :].expand(-1, padding_len, -1, -1)], dim=1
            )

    def compute(self, dt: float):
        self.current_start_time += dt

        # On the first compute() after a reset, robot.data is already refreshed with the
        # new post-reset state. Update command_origin to the actual EE position so the
        # trajectory is anchored where the EE truly is after reset.
        if self._need_origin_update.any():
            update_ids = self._need_origin_update.nonzero(as_tuple=False).flatten()
            self.command_origin[update_ids] = (
                self.robot.data.body_link_state_w[update_ids, self.body_idx, :3].clone()
            )
            self._need_origin_update[update_ids] = False
            self._update_command_w(update_ids.tolist())
            self._set_travel_direction(update_ids)

        self._update_command_w(range(self.num_envs))
        # Parent handles metric updates and time_left countdown
        super().compute(dt)
        # Append current EE pose to latency history buffer
        current_pos = self.robot.data.body_link_state_w[:, self.body_idx, :3]
        current_quat = self.robot.data.body_link_state_w[:, self.body_idx, 3:7]
        self.ee_pose_history = torch.roll(self.ee_pose_history, shifts=-1, dims=1)
        self.ee_pose_history[:, -1, :] = torch.cat([current_pos, current_quat], dim=-1)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        # Reset history to current pose to avoid discontinuities after reset
        if len(env_ids) > 0:
            current_pos = self.robot.data.body_link_state_w[env_ids, self.body_idx, :3]
            current_quat = self.robot.data.body_link_state_w[env_ids, self.body_idx, 3:7]
            current_pose = torch.cat([current_pos, current_quat], dim=-1)
            self.ee_pose_history[env_ids] = current_pose.unsqueeze(1).expand(-1, self.history_max_len, -1)

    def get_delayed_ee_pose(self) -> torch.Tensor:
        """Returns the EE pose delayed by ``cfg.pose_latency`` seconds."""
        latency_steps = int(self.latency_seconds / self._env.step_dt)
        latency_steps = max(0, min(latency_steps, self.history_max_len - 1))
        return self.ee_pose_history[:, -1 - latency_steps, :]

    def _update_command_w(self, env_ids):
        """Update pose_command_w from the current position in the recorded sequence."""
        if len(env_ids) == 0:
            return
        indices = self.current_episode_indices[env_ids]
        step_indices = (self.current_start_time[env_ids] / self.sim_dt).long()
        step_indices = torch.clamp(step_indices, max=self.ee_pos.shape[1] - 1)

        target_pos = self.ee_pos[indices, step_indices]
        target_rot_mat = self.ee_rot_mat[indices, step_indices]
        target_pos[:, 2] += self.added_heights[env_ids]

        # Transform tip frame into tracked link frame: T_link = T_tip * (T_offset)^-1
        link_pos = target_pos + (target_rot_mat @ self.tip_offset_pos_inv)
        link_rot_mat = target_rot_mat @ self.tip_offset_rot_inv
        link_quat = quat_from_matrix(link_rot_mat)

        self.pose_command_w[env_ids, :3] = link_pos + self.command_origin[env_ids]
        self.pose_command_w[env_ids, 3:7] = link_quat

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample a new episode, reset timer, height, and anchor origin."""
        if len(env_ids) == 0:
            return
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(torch.randint(0, 100000, (1,)).item()))
        self.current_episode_indices[env_ids] = torch.randint(
            0, self.ee_pos.shape[0], (len(env_ids),), device=self.device, generator=generator
        )
        self.current_start_time[env_ids] = 0.0
        if self.added_random_height_range is not None:
            r = torch.rand((len(env_ids),), device=self.device)
            diff = self.added_random_height_range[1] - self.added_random_height_range[0]
            self.added_heights[env_ids] = r * diff + self.added_random_height_range[0]
        else:
            self.added_heights[env_ids] = 0.0
        # Use stale root pos as a temporary placeholder; command_origin will be corrected
        # to the actual post-reset EE position on the next compute() call.
        self.command_origin[env_ids] = self.robot.data.root_link_pos_w[env_ids].clone()
        self._need_origin_update[env_ids] = True
        # Initialize command immediately so metrics/time_left use fresh targets
        self._update_command_w(env_ids)
        self._set_travel_direction(env_ids)


@configclass
class PicklePoseSequenceCommandCfg(UniformWorldPoseCommandCfg):
    """Configuration for pose sequence command generator loaded from a pickle file."""

    class_type: type = PicklePoseSequenceCommand
    pose_latency: float = 0.1
    history_buffer_length: int = 100
    file_path: str = MISSING
    planar_center: bool = False
    add_random_height_range: Optional[Tuple[float, float]] = None
    # Random offset ranges added to the pickle trajectory
    add_random_pos_x_range: Optional[Tuple[float, float]] = None
    add_random_pos_y_range: Optional[Tuple[float, float]] = None
    add_random_orn_range: Optional[Tuple[float, float]] = None  # (roll, pitch, yaw) in rad
    episode_length_s: float = 10
    tip_offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tip_offset_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
