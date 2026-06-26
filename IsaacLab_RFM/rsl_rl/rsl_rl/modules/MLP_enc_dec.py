from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from .actor_critic import get_activation


class MLPEncoderDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation="elu",
        orthogonal_init=False,
        hidden_dim=[256, 256],
        dropout=None,
        normalization=False,
    ):
        super(MLPEncoderDecoder, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        activation = get_activation(activation)
        self.orthogonal_init = orthogonal_init
        self.output_dim = output_dim
        MLP_layers = []
        MLP_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(MLP_layers[-1].weight, np.sqrt(2))
        MLP_layers.append(activation)
        if dropout is not None:
            MLP_layers.append(nn.Dropout(dropout))
        if normalization:
            MLP_layers.append(nn.LayerNorm(hidden_dim[0]))
        for l in range(len(hidden_dim)):
            if l == len(hidden_dim) - 1:
                MLP_layers.append(nn.Linear(hidden_dim[l], self.output_dim))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(MLP_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(MLP_layers[-1].bias, 0.0)
            else:
                MLP_layers.append(
                    nn.Linear(
                        hidden_dim[l],
                        hidden_dim[l + 1],
                    )
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(MLP_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(MLP_layers[-1].bias, 0.0)
                MLP_layers.append(activation)
                if dropout is not None:
                    MLP_layers.append(nn.Dropout(dropout))
                if normalization:
                    MLP_layers.append(nn.LayerNorm(hidden_dim[l + 1]))
        self.MLP = nn.Sequential(*MLP_layers)

        print(f"Generated MLP: {self.MLP}")

    def forward(self, input):
        output = self.MLP(input)
        return output
