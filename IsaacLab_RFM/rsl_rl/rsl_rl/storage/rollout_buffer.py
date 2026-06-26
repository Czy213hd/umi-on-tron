from __future__ import annotations

import torch

from rsl_rl.storage.rollout_storage import RolloutStorage

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import Dict, Any

from .sequence_buffer import BufferConfig


class RolloutBuffer:
    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple,
        actor_input_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
        cn_obs_hist_shape: tuple,
        next_obs_shape: tuple,
        gru_latent_shape,
        device="cpu",
    ):
        self.buffer_configs = {
            "observations": BufferConfig(shape=obs_shape),  # Adjust size as needed
            "actor_inputs": BufferConfig(shape=actor_input_shape),
            "cn_obs_history": BufferConfig(shape=cn_obs_hist_shape),
            "next_obs_gt": BufferConfig(shape=next_obs_shape),
            # "base_lin_vel_gt": BufferConfig(shape=(3,)),
            "privileged_observations": BufferConfig(shape=critic_obs_shape),
            "actions": BufferConfig(shape=action_shape),
            "gru_latent": BufferConfig(shape=gru_latent_shape),
            "rewards": BufferConfig(shape=(1,)),
            "dones": BufferConfig(shape=(1,)),
            "values": BufferConfig(shape=(1,)),
            "actions_log_prob": BufferConfig(shape=(1,)),  # Should match observations
            "action_mean": BufferConfig(shape=action_shape),
            "action_sigma": BufferConfig(shape=action_shape),
            "fm_old_logprob": BufferConfig(shape=(1,)),
            "fm_t": BufferConfig(shape=(1,)),
            "fm_noise": BufferConfig(shape=action_shape),
            "fm_condition_mask": BufferConfig(shape=actor_input_shape),
            "returns": BufferConfig(shape=(1,)),
            "advantages": BufferConfig(shape=(1,)),
        }
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        # Initialize buffers
        self.buffers: Dict[str, torch.Tensor] = {}

        self.reset_transition()
        self._initialize_buffers()

    def reset_transition(self):
        self.transition: Dict[str, torch.Tensor] = {
            "observations": None,
            "actor_inputs": None,
            "privileged_observations": None,
            "cn_obs_history": None,
            "next_obs_gt": None,
            # "base_lin_vel_gt": None,
            "actions": None,
            "rewards": None,
            "gru_latent": None,
            "dones": None,
            "values": None,
            "actions_log_prob": None,
            "action_mean": None,
            "action_sigma": None,
            "fm_old_logprob": None,
            "fm_t": None,
            "fm_noise": None,
            "fm_condition_mask": None,
        }

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.buffers["values"][step + 1]
            next_is_not_terminal = 1.0 - self.buffers["dones"][step].float()
            delta = (
                self.buffers["rewards"][step]
                + next_is_not_terminal * gamma * next_values
                - self.buffers["values"][step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.buffers["returns"][step] = advantage + self.buffers["values"][step]
        # Compute and normalize the r_advantages
        self.buffers["advantages"] = self.buffers["returns"] - self.buffers["values"]
        self.buffers["advantages"] = (self.buffers["advantages"] - self.buffers["advantages"].mean()) / (
            self.buffers["advantages"].std() + 1e-8
        )

    def _initialize_buffers(self):
        """Initialize all buffers based on configurations"""
        for name, config in self.buffer_configs.items():
            shape = (self.num_transitions_per_env, self.num_envs) + config.shape
            self.buffers[name] = torch.full(shape, config.default_value, device=self.device)

    def clear(self):
        self.step = 0

    def add_transitions(self):
        if self.step > self.num_transitions_per_env - 1:
            raise AssertionError("Rollout buffer overflow")
        for name in self.transition.keys():
            assert self.transition[name] is not None, f"Transition key {name} is None"
            if name in self.buffers.keys():
                self.buffers[name][self.step] = (
                    self.transition[name].clone().to(self.buffers[name].dtype).view_as(self.buffers[name][self.step])
                )
            else:
                raise ValueError(f"Transition key {name} not found in buffer keys")
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        flatten_buffers = {}
        for name, tensor in self.buffers.items():
            assert (~torch.isinf(tensor)).all() and (~torch.isnan(tensor)).all()  # and (tensor != 0).any()
            flatten_buffers[name] = tensor.flatten(0, 1)

        for _ in range(num_epochs):
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                batch = {
                    name: tensor[batch_idx] for name, tensor in flatten_buffers.items()  # if ~((tensor == 0).all())
                }
                yield batch
