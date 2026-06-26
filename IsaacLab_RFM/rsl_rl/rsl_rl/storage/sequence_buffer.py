#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from dataclasses import MISSING
from rsl_rl.utils import split_and_pad_trajectories
from isaaclab.utils import configclass
from typing import Dict, Any


@configclass
class BufferConfig:
    """Configuration for each buffer tensor"""

    shape: tuple = MISSING  # Shape suffix after sequence_length and num_envs
    default_value: float = 0.0


class SequenceBuffer:
    def __init__(self, num_envs: int, sequence_length: int, obs_shape, device="cpu"):
        self.device = device
        # Core
        self.buffer_configs = {
            "observations": BufferConfig(shape=obs_shape),  # Adjust size as needed
            "external_wrench_b_all": BufferConfig(shape=(6,)),
            "base_lin_vel": BufferConfig(shape=(3,)),
            "feet_contact_ids": BufferConfig(shape=(4,)),
            "external_wrench_body_ids": BufferConfig(shape=(1,)),
            "external_wrench_part_ids": BufferConfig(shape=(1,)),
            "next_observations": BufferConfig(shape=obs_shape),  # Should match observations
            "over_contact_mask": BufferConfig(shape=(1,)),
            # "EE_external_wrench_b": BufferConfig(shape=(6,)),
        }

        self.select_idx = torch.zeros(num_envs, dtype=torch.int, device=self.device)
        self.sequence_length = sequence_length
        self.num_envs = num_envs
        self.step = torch.zeros(num_envs, dtype=torch.int, device=self.device)

        # Initialize buffers
        self.buffers: Dict[str, torch.Tensor] = {}
        self._initialize_buffers()

    def _initialize_buffers(self):
        """Initialize all buffers based on configurations"""
        for name, config in self.buffer_configs.items():
            shape = (self.sequence_length, self.num_envs) + config.shape
            self.buffers[name] = torch.full(shape, config.default_value, device=self.device)

    def add_obs(self, obs: torch.Tensor, ground_truth: Dict[str, torch.Tensor]):
        """Add observations and ground truth data to buffers"""
        if (self.step >= self.sequence_length).any():
            raise AssertionError("Rollout buffer overflow")

        # fifo deque
        self._forget()

        idx = torch.arange(self.num_envs, device=self.device)
        current_step = self.step

        # Update observations with noise
        self.buffers["observations"][current_step, idx] = obs.clone()
        torque_indices = slice(52, 74)
        torque_noise = torch.zeros_like(obs[:, torque_indices])
        self.buffers["observations"][current_step, idx, torque_indices] += torque_noise.uniform_(-0.05, 0.05)

        for gt_name in ground_truth.keys():
            if gt_name in self.buffers.keys():
                self.buffers[gt_name][current_step, idx] = (
                    ground_truth[gt_name]
                    .clone()
                    .to(self.buffers[gt_name].dtype)
                    .view_as(self.buffers[gt_name][current_step, idx])
                )
            else:
                raise ValueError(f"Ground truth key {gt_name} not found in buffer keys")

        # self.buffers["EE_external_wrench_b"][current_step, idx] = critic_obs[:, 116:122].clone()

    def _forget(self):
        """Shift all buffer data forward by one step"""
        if len(self.full_index) == 0:
            return

        for buffer in self.buffers.values():
            buffer[:-1, self.full_index] = buffer[1:, self.full_index]
            buffer[-1, self.full_index] = 0

    def clear(self, dones):
        """Clear buffers for completed episodes"""
        reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if len(reset_ids) == 0:
            return

        self.step[reset_ids] = 0
        for buffer in self.buffers.values():
            buffer[:, reset_ids] = 0

    def add_next_obs(self, next_obs):
        if (self.step >= self.sequence_length).any():
            raise AssertionError("Rollout buffer overflow")

        idx = torch.arange(self.num_envs, device=self.device)
        current_step = self.step
        self.buffers["next_observations"][current_step, idx] = next_obs.clone()

    @property
    def full_index(self):
        """Get indices of environments that have completed their sequence"""
        return (self.step == (self.sequence_length - 1)).to(torch.float).nonzero(as_tuple=False).flatten()

    def add_buffer(self, name: str, shape: tuple, default_value: float = 0.0):
        """Dynamically add a new buffer"""
        self.buffer_configs[name] = BufferConfig(shape=shape, default_value=default_value)
        buffer_shape = (self.sequence_length, self.num_envs) + shape
        self.buffers[name] = torch.full(buffer_shape, default_value, device=self.device)

    def progress(self):
        self.select_idx = self.step.clone()
        self.step = torch.where(self.step < self.sequence_length - 1, self.step + 1, self.step)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("Batch size too small for number of mini batches")
        transposed_buffers = {}
        for name, tensor in self.buffers.items():
            assert (~torch.isinf(tensor)).all() and (~torch.isnan(tensor)).all()  # and (tensor != 0).any()
            transposed_buffers[name] = tensor.transpose(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                batch = {
                    name: tensor[batch_idx] for name, tensor in transposed_buffers.items()  # if ~((tensor == 0).all())
                }
                batch["select_ids"] = self.select_idx[batch_idx]
                yield batch
