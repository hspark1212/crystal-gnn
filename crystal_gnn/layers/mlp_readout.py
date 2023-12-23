"""https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/mlp_readout_layer.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NONLINEAR = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": F.leaky_relu,
    "silu": F.silu,
}


class MLPReadout(nn.Module):
    """
    Readout function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        L: int = 2,
        bias: bool = True,
        nonlinear: str = "relu",
    ):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2**l, input_dim // 2 ** (l + 1), bias=bias)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2**L, output_dim, bias=bias))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.nonlinear = NONLINEAR[nonlinear]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)  # [B, H/2**l]
            y = self.nonlinear(y)
        y = self.FC_layers[self.L](y)  # [B, O]
        return y
