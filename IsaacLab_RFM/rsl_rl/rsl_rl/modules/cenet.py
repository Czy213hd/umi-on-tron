"""DreamWaQ-style history encoder used by the encoder ablation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .MLP_enc_dec import MLPEncoderDecoder


class CENet(nn.Module):
    """Encode a fixed proprioceptive history with an MLP.

    The output contract deliberately matches ``SimplifiedContactNetModel`` so
    the policy input and auxiliary objectives stay unchanged in the ablation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        next_obs_decoder_output_dim: int,
        next_obs_decoder_input_dim: int,
        history_length: int = 10,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        next_obs_decoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        next_obs_decoder_activation: str = "elu",
        **_: object,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        layers: list[nn.Module] = []
        feature_dim = input_dim * history_length
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(feature_dim, hidden_dim), nn.ELU()))
            feature_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        # DreamWaQ uses separate velocity, latent-mean, and latent-logvar
        # heads from the shared 64-D feature. Concatenating those heads keeps
        # this repository's existing [velocity, mu, logvar] contract.
        latent_dim = (output_dim - 3) // 2
        if 3 + 2 * latent_dim != output_dim:
            raise ValueError(f"CENet output_dim must equal 3 + 2*latent_dim, got {output_dim}")
        self.velocity_head = nn.Linear(feature_dim, 3)
        self.latent_mean_head = nn.Linear(feature_dim, latent_dim)
        self.latent_logvar_head = nn.Linear(feature_dim, latent_dim)
        self.next_obs_decoder = MLPEncoderDecoder(
            next_obs_decoder_input_dim,
            next_obs_decoder_output_dim,
            activation=next_obs_decoder_activation,
            hidden_dim=next_obs_decoder_hidden_dims,
        )

    def forward(self, observation_history: torch.Tensor) -> torch.Tensor:
        if observation_history.ndim != 3:
            raise ValueError(
                "CENet expects [batch, history, features], got "
                f"{tuple(observation_history.shape)}"
            )
        if observation_history.shape[1] != self.history_length:
            raise ValueError(
                f"CENet expects history_length={self.history_length}, got "
                f"{observation_history.shape[1]}"
            )
        features = self.encoder(observation_history.flatten(start_dim=1))
        return torch.cat(
            (
                self.velocity_head(features),
                self.latent_mean_head(features),
                self.latent_logvar_head(features),
            ),
            dim=-1,
        )


class IdentityGRUWrapper(nn.Module):
    """GRU-compatible identity stage for the CENet architecture ablation."""

    def __init__(
        self,
        gru_latent_dim: int,
        num_envs: int,
        gru_input_dim: int | None = None,
        device: str = "cpu",
        **_: object,
    ) -> None:
        super().__init__()
        self.gru_latent_dim = gru_latent_dim
        self.input_dim = gru_input_dim or gru_latent_dim
        self.register_buffer("hidden_state", torch.zeros(1, num_envs, gru_latent_dim, device=device))

    def gru_forward(self, x: torch.Tensor, hx: torch.Tensor | None) -> torch.Tensor:
        self.hidden_state.zero_()
        return x

    def gru_forward_without_memory(self, x: torch.Tensor, hx: torch.Tensor | None) -> torch.Tensor:
        return x

    def gru_forward_without_memory_with_hidden(
        self, x: torch.Tensor, hx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x, torch.zeros_like(hx)

    def reset_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        if dones is None:
            self.hidden_state.zero_()
            return
        indices = torch.nonzero(dones).squeeze(1)
        if indices.numel() > 0:
            self.hidden_state[:, indices, :] = 0.0
