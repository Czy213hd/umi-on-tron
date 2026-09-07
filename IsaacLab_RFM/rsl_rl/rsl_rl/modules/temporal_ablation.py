"""Minimal encoders for Transformer/GRU architecture ablations."""

from __future__ import annotations

import torch
import torch.nn as nn

from .MLP_enc_dec import MLPEncoderDecoder


class LastObservationEncoder(nn.Module):
    """Project only the newest observation before the recurrent GRU stage.

    This intentionally contains no temporal mixing: all temporal state in the
    GRU-only ablation is carried by ``GRUWrapper``.  The linear projection is
    only the dimension adapter required by the existing [velocity, mu, logvar]
    interface and auxiliary decoder.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        next_obs_decoder_output_dim: int,
        next_obs_decoder_input_dim: int,
        next_obs_decoder_hidden_dims=(256, 128),
        next_obs_decoder_activation: str = "elu",
        **_: object,
    ) -> None:
        super().__init__()
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.next_obs_decoder = MLPEncoderDecoder(
            next_obs_decoder_input_dim,
            next_obs_decoder_output_dim,
            activation=next_obs_decoder_activation,
            hidden_dim=next_obs_decoder_hidden_dims,
        )

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        if observation_history.ndim != 3:
            raise ValueError(
                "LastObservationEncoder expects [batch, history, features], got "
                f"{tuple(observation_history.shape)}"
            )
        return self.output_layer(observation_history[:, -1, :])
