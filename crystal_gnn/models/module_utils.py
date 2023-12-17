from typing import Optional

import numpy as np
import torch
from torch import nn


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale**2)

    @torch.no_grad()
    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


# TODO: depreciated
class Normalizer(object):
    """
    normalize for regression
    """

    def __init__(self, mean: float, std: float) -> None:
        if mean is not None and std is not None:
            self._norm_func = lambda tensor: (tensor - mean) / std
            self._denorm_func = lambda tensor: tensor * std + mean
        else:
            self._norm_func = lambda tensor: tensor
            self._denorm_func = lambda tensor: tensor

        self.mean = mean
        self.std = std

    def encode(self, tensor) -> torch.Tensor:
        return self._norm_func(tensor)

    def decode(self, tensor) -> torch.Tensor:
        return self._denorm_func(tensor)
