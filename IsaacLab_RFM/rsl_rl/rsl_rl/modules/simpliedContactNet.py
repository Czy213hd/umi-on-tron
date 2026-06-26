import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP_enc_dec import MLPEncoderDecoder

import math

from torch.distributions import Normal

from copy import deepcopy


class SimpliedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(SimpliedTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src):
        x = src
        if self.norm_first:
            x = x[:, -1, :].unsqueeze(1) + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x[:, -1, :].unsqueeze(1) + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x, attn_weights = self.self_attn(x[:, -1, :].unsqueeze(1), x, x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SimplifiedContactNetModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_dim,
        num_layers,
        num_heads,
        dim_feedforward,
        next_obs_decoder_output_dim,
        next_obs_decoder_input_dim,
        next_obs_decoder_hidden_dims=[256, 256],
        next_obs_decoder_activation="elu",
        dropout=0.1,
    ):
        super(SimplifiedContactNetModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # Create Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers - 1)
        self.s_transformer_layer = SimpliedTransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # self.s_transformer_encoder = nn.ModuleList([deepcopy(self.transformer_layer) for i in range(num_layers)])

        self.next_obs_decoder: MLPEncoderDecoder = MLPEncoderDecoder(
            next_obs_decoder_input_dim,
            next_obs_decoder_output_dim,
            activation=next_obs_decoder_activation,
            hidden_dim=next_obs_decoder_hidden_dims,
        )

        self.output_layer = nn.Linear(model_dim, output_dim)

    def generate_mask(self, seq_len):
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - self.n + 1)
            mask[i, start : i + 1] = False
        return mask

    def forward(self, x):
        # Create mask
        seq_len = x.size(1)
        # mask = self.generate_mask(seq_len).to(x.device)
        # Embed input
        x = self.embedding(x) * math.sqrt(self.model_dim)

        position_encoding = self.get_positional_encoding(seq_len=seq_len)
        position_encoding = position_encoding.to(x.device)
        x = x + position_encoding

        x = self.dropout(x)
        # Pass through Transformer layers
        x = self.transformer_encoder(x)
        x = self.s_transformer_layer(x)
        # Output layer
        return self.output_layer(x).squeeze(1)

    def get_positional_encoding(self, seq_len, device="cuda"):
        # Create a positional encoding tensor
        positional_encoding = torch.zeros(seq_len, self.model_dim, device=device)

        # Create a position array (0, 1, 2, ..., seq_len-1) on the specified device
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)

        # Compute the div_term array
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.model_dim)
        )

        # Apply the sine to even indices
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine to odd indices
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding

    # def update_data(self):
    #     self.base_lin_vel_est = self._cn_output[:, :3]
    #     self.ext_wrench_est = self._cn_output[:, 3:9]
    #     self.feet_contact_ids_est = self._cn_output[:, 9:17]
    #     self.part_ids_est = self._cn_output[:, 17 : 17 + self.part_num]
    #     self.body_ids_est = self._cn_output[:, 17 + self.part_num : 17 + self.part_num + self.body_num]
    #     # self.over_contact = self._cn_output[:, 37:39]
    #     assert self._cn_output.shape[1] == 17 + self.part_num + self.body_num + 2 * self.latent_dim
    #     # self.part_ids_est = self._cn_output[:, 17:19]
    #     # self.body_ids_est = self._cn_output[:, 19:21]
    #     self.mu = self._cn_output[:, -2 * self.latent_dim : -self.latent_dim]
    #     self.logvar = self._cn_output[:, -self.latent_dim :]
    #     if self.standard_gaussian is None:
    #         self.standard_gaussian = Normal(torch.zeros_like(self.mu), torch.ones_like(self.logvar))
    #     contactNet_latent = self.standard_gaussian.sample() * (self.logvar.exp() + 1e-5).sqrt() + self.mu
    #     # contactNet_latent = self.gaussian.sample()
    #     self.contactNet_output = torch.cat((self._cn_output[:, : -(2 * self.latent_dim)], contactNet_latent), dim=1)
    #     self.next_obs_est = self.next_obs_decoder(self.contactNet_output)
