#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn


class GRUWrapper(nn.Module):

    def __init__(
        self,
        gru_latent_dim,
        gru_input_dim,
        num_envs,
        gru_batch_first=True,
        device="cpu",
    ):
        super().__init__()
        # gru init
        self.gru_batch_first = gru_batch_first
        self.gru = nn.GRU(gru_input_dim, gru_latent_dim, batch_first=gru_batch_first)
        self.gru_output_layer = nn.Linear(gru_latent_dim, gru_latent_dim)
        # self.norm = nn.LayerNorm(gru_latent_dim, eps=1e-5, bias=True)
        self.gru_latent_dim = gru_latent_dim
        # register hidden state as a buffer so that .to(device) moves it together with the module
        self.register_buffer("hidden_state", torch.zeros(1, num_envs, gru_latent_dim, device=device))
        self.last_input: torch.Tensor | None = None

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.gru_batch_first:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] == 1:
            return x
        if x.dim() == 3:
            raise ValueError("Input must have a time dimension of 1")
        raise ValueError(f"Input must have 2 or 3 dimension, got {x.dim()}")

    def gru_forward(self, x, hx):
        x = self._normalize_input(x)

        if self.last_input is None:
            self.last_input = x.clone()
        elif torch.all(self.last_input == x):
            raise ValueError("Input is the same as last input")
        else:
            self.last_input = x.clone()

        # make sure hidden state is on the same device as the input
        if hx is None:
            hx = self.hidden_state
        if hx.device != x.device:
            hx = hx.to(x.device)

        hidden_state, h_n = self.gru(
            input=x, hx=hx
        )  # hidden_state shape: (batch_size, 1num_sequence=1, hidden_size), h_n shape: (num_layers =1, batch_size, hidden_size)
        # Update hidden state
        self.hidden_state = h_n.clone().detach()
        return x.squeeze(1) + self.gru_output_layer(hidden_state.squeeze(1))

    def gru_forward_without_memory_with_hidden(self, x, hx):
        x = self._normalize_input(x)
        hidden_state, h_n = self.gru(input=x, hx=hx)
        return x.squeeze(1) + self.gru_output_layer(hidden_state.squeeze(1)), h_n

    def gru_forward_without_memory(self, x, hx):
        x = self._normalize_input(x)
        hidden_state, _ = self.gru(input=x, hx=hx)
        return x.squeeze(1) + self.gru_output_layer(hidden_state.squeeze(1))

    def get_hidden_state(self):
        return self.hidden_state

    def reset_hidden_states(self, dones=None):
        if dones is None:
            # reset all hidden states
            self.hidden_state.zero_()
            return

        indices_to_reset = torch.nonzero(dones).squeeze(1)
        if indices_to_reset.numel() > 0:
            replacement_tensor = torch.zeros_like(self.hidden_state[:, 0, :])
            self.hidden_state[:, indices_to_reset, :] = replacement_tensor
