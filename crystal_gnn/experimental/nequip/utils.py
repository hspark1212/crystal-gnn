import math
from pydantic import BaseModel, field_validator

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Irrep


class IrrepsArray(BaseModel):
    irreps: Irreps
    array: torch.Tensor

    class Config:
        arbitrary_types_allowed = True

    @field_validator("irreps")
    def convert_irreps(cls, v):
        return Irreps(v)

    @field_validator("array")
    def check_array(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("array must be a torch.Tensor")
        return v


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:  # move to utils.py
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class BesselBasis(nn.Module):
    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        num_basis : int
            Number of Bessel Basis functions

        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def reset_parameters(self):
        if self.trainable:
            nn.init.constant_(self.bessel_weights, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.

        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))
