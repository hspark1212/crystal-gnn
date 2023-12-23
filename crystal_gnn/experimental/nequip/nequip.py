"""modified from nequip.py in 
https://github.com/google-deepmind/materials_discovery
"""
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Union

import operator
import functools


import e3nn
from e3nn.o3 import Irreps, Irrep, TensorProduct, Linear, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate
from torch_runstats.scatter import scatter

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from crystal_gnn.experimental.nequip.utils import (
    IrrepsArray,
    BesselBasis,
    tp_path_exists,
)  # TODO: update path
from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.models.module_utils import NONLINEARITY


# Types
Array = torch.Tensor


def prod(xs):
    """From e3nn_jax/util/__init__.py."""
    return functools.reduce(operator.mul, xs, 1)


class NequIPConvolution(nn.Module):
    """NequIP Convolution.

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

    Args:
        hidden_irreps: irreducible representation of hidden/latent features
        use_sc: use self-connection in network (recommended)
        nonlinearities: nonlinearities to use for even/odd irreps
        radial_net_nonlinearity: nonlinearity to use in radial MLP
        radial_net_n_hidden: number of hidden neurons in radial MLP
        radial_net_n_layers: number of hidden layers for radial MLP
        num_basis: number of Bessel basis functions to use
        n_neighbors: constant number of per-atom neighbors, used for internal
        normalization
        scalar_mlp_std: standard deviation of weight init of radial MLP

      Returns:
        Updated node features h after the convolution.
    """

    def __init__(
        self,
        hidden_irreps: Irreps,
        sh_irreps: Irreps,
        node_irreps: Irreps,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        radial_net_nonlinearity: str = "raw_swish",
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        num_basis: int = 8,
        n_neighbors: float = 1.0,
        # scalar_mlp_std: float = 4.0, # TODO: check
    ):
        super().__init__()
        # config
        self.hidden_irreps = hidden_irreps
        self.sh_irreps = sh_irreps
        self.node_irreps = node_irreps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.n_neighbors = n_neighbors
        # self.scalar_mlp_std = scalar_mlp_std

        # layers
        irreps_scalars = []
        irreps_nonscalars = []
        irreps_gate_scalars = []

        # get scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if Irrep(irrep).l == 0 and tp_path_exists(
                self.hidden_irreps, self.sh_irreps, irrep
            ):
                irreps_scalars += [(multiplicity, irrep)]

        irreps_scalars = Irreps(irreps_scalars)

        # get non-scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if Irrep(irrep).l > 0 and tp_path_exists(
                self.hidden_irreps, self.sh_irreps, irrep
            ):
                irreps_nonscalars += [(multiplicity, irrep)]

        irreps_nonscalars = Irreps(irreps_nonscalars)

        # get gate scalar irreps
        if tp_path_exists(self.hidden_irreps, self.sh_irreps, "0e"):
            gate_scalar_irreps_type = "0e"
        else:
            gate_scalar_irreps_type = "0o"

        for multiplicity, _ in irreps_nonscalars:
            irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]

        irreps_gate_scalars = Irreps(irreps_gate_scalars)

        # final layer output irreps are all three
        # note that this order is assumed by the gate function later, i.e.
        # scalars left, then gate scalar, then non-scalars
        h_out_irreps = irreps_scalars + irreps_gate_scalars + irreps_nonscalars

        if self.use_sc:
            self.self_connection = FullyConnectedTensorProduct(
                irreps_in1=self.hidden_irreps,
                irreps_in2=self.node_irreps,
                irreps_out=h_out_irreps,
            )

        self.lin_1 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # we gather the instructions for the tp as well as the tp output irreps
        mode = "uvu"
        trainable = "True"
        irreps_after_tp = []
        instructions = []

        # iterate over both arguments, i.e. node irreps and edge irreps
        # if they have a valid TP path for any of the target irreps,
        # add to instructions and put in appropriate position
        # we use uvu mode (where v is a single-element sum) and weights will
        # be provide externally by the scalar MLP
        # this triple for loop is similar to the one used in e3nn and nequip
        for i, (mul_in1, irreps_in1) in enumerate(self.hidden_irreps):
            for j, (_, irreps_in2) in enumerate(self.sh_irreps):
                for curr_irreps_out in irreps_in1 * irreps_in2:
                    if curr_irreps_out in h_out_irreps:
                        k = len(irreps_after_tp)
                        irreps_after_tp += [(mul_in1, curr_irreps_out)]
                        instructions += [(i, j, k, mode, trainable)]

        # we will likely have constructed irreps in a non-l-increasing order
        # so we sort them to be in a l-increasing order
        irreps_after_tp, p, _ = Irreps(irreps_after_tp).sort()

        # if we sort the target irreps, we will have to sort the instructions
        # acoordingly, using the permutation indices
        sorted_instructions = []

        for irreps_in1, irreps_in2, irreps_out, mode, trainable in instructions:
            sorted_instructions += [
                (
                    irreps_in1,
                    irreps_in2,
                    p[irreps_out],
                    mode,
                    trainable,
                )
            ]

        # TP between spherical harmonics embedding of the edge vector
        # Y_ij(\hat{r}) and neighbor node h_j, weighted on a per-atom basis
        # by the radial newtork R(r_ij)
        self.tp = TensorProduct(
            irreps_in1=self.hidden_irreps,
            irreps_in2=self.sh_irreps,
            irreps_out=irreps_after_tp,
            instructions=sorted_instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # scalar radial network, number of output neurons is the total number of
        # tensor product paths, nonlinearity must have f(0)=0 and MLP must no
        # have biases
        n_tp_weights = 0

        # get output dim of radial MLP / number of TP weights
        for ins in self.tp.instructions:
            if ins.has_weight:
                n_tp_weights += prod(ins.path_shape)
        # build radial MLP R(r) that maps from interatomic distance to TP weights
        # must not use biases to that R(0) = 0
        self.fc = FullyConnectedNet(
            (self.num_basis,)
            + (self.radial_net_n_hidden,) * self.radial_net_n_layers
            + (n_tp_weights,),
            NONLINEARITY[self.radial_net_nonlinearity],
        )

        # linear_2
        self.lin_2 = Linear(
            irreps_in=irreps_after_tp.simplify(),
            irreps_out=h_out_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # gate nonlinearity, applied to gate data, consisting of:
        # a) regular scalars,
        # b) gate scalars, and
        # c) non-scalars to be gated
        # in this order
        nonlinearities = {
            1: self.nonlinearities["e"],
            -1: self.nonlinearities["o"],
        }
        self.equivariant_nonlin = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[
                NONLINEARITY[nonlinearities[ir.p]] for _, ir in irreps_scalars
            ],
            irreps_gates=irreps_gate_scalars,
            act_gates=[
                NONLINEARITY[nonlinearities[ir.p]] for _, ir in irreps_gate_scalars
            ],
            irreps_gated=irreps_nonscalars,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_1.weight.data.normal_(mean=0.0, std=0.02)
        self.tp.weight.data.normal_(mean=0.0, std=0.02)
        for l in self.fc:
            l.weight.data.normal_(mean=0.0, std=0.02)  # TODO: std=4 (google-deepmind)
        self.self_connection.weight.data.normal_(mean=0.0, std=0.02)
        self.lin_2.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        node_features: IrrepsArray,
        node_attributes: IrrepsArray,
        edge_sh: IrrepsArray,
        edge_src: Array,
        edge_dst: Array,
        edge_embedded: Array,
    ) -> IrrepsArray:
        if self.use_sc:
            self_connection = self.self_connection(
                node_features.array, node_attributes.array
            )
        h = node_features.array  # [B_n, hidden_irreps]
        #  # linear_1, stays in current h-space
        h = self.lin_1(h)

        # map node features onto edges for tp
        edge_features = h[edge_src]  # [n_edges, hidden_irreps]

        # the TP weights (v dimension) are given by the FC
        weight = self.fc(edge_embedded)  # [n_edges, n_tp_weights]

        edge_features = self.tp(
            edge_features,
            edge_sh.array,
            weight,
        )  # [n_edges, h_out_irreps]

        # aggregate edges onto node after tp
        h = scatter(
            edge_features,
            edge_dst,
            dim=0,
            dim_size=h.shape[0],
        )  # [n_nodes, h_out_irreps]

        # normalize by the average (not local) number of neighbors
        h = h / self.n_neighbors  # TODO: check

        # Linear_2, now we create extra gate scalars by mapping to h-out
        h = self.lin_2(h)

        # self-connection, similar to a resnet-update that sums the output from
        # the TP to chemistry-weighted h
        if self.use_sc:
            h = h + self_connection
        # gate
        h = self.equivariant_nonlin(h)

        h_node = IrrepsArray(irreps=self.equivariant_nonlin.irreps_out, array=h)

        return h_node


class NEQUIP(BaseModule):
    """NequIP.

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

        Args:
            graph_net_steps: number of NequIP convolutional layers
            use_sc: use self-connection in network (recommended)
            nonlinearities: nonlinearities to use for even/odd irreps
            n_element: number of chemical elements in input data
            hidden_irreps: irreducible representation of hidden/latent features
            sh_irreps: irreducible representations on the edges
            num_basis: number of Bessel basis functions to use
            r_max: radial cutoff used in length units
            radial_net_nonlinearity: nonlinearity to use in radial MLP
            radial_net_n_hidden: number of hidden neurons in radial MLP
            radial_net_n_layers: number of hidden layers for radial MLP
            shift: per-atom energy shift
            scale: per-atom energy scale
            n_neighbors: constant number of per-atom neighbors, used for internal
                normalization
            scalar_mlp_std: standard deviation of weight init of radial MLP

        Returns:
            Potential energy of the inputs.
    """

    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)
        # config
        self.graph_net_steps: int = _config["num_conv"]
        self.use_sc: bool = _config["residual"]
        self.nonlinearities: Union[str, Dict[str, str]] = {
            "e": "raw_swish",
            "o": "tanh",
        }
        self.n_elements: int = 103

        self.hidden_irreps: str = "128x0e + 64x1e + 4x2e"
        self.sh_irreps: str = "1x0e + 1x1e + 1x2e"

        self.num_basis: int = 8
        self.r_max: float = _config["cutoff"]

        self.radial_net_nonlinearity: str = "raw_swish"
        self.radial_net_n_hidden: int = 64
        self.radial_net_n_layers: int = 2

        self.n_neighbors: float = 1.0
        # self.n_neighbors: float = 10.0 # TODO: check it
        # self.scalar_mlp_std: float = 4.0  # TODO: check it

        # layers
        self.node_irreps = Irreps(f"{self.n_elements}x0e")
        self.lin_1 = Linear(
            irreps_in=Irreps(f"{self.n_elements}x0e"),
            irreps_out=Irreps(self.hidden_irreps),
            internal_weights=True,
            shared_weights=True,
        )
        # edge embedding
        self.bassel_basis = BesselBasis(
            r_max=self.r_max, num_basis=self.num_basis, trainable=True
        )
        # conv layers
        self.conv_layers = nn.ModuleList(
            [
                NequIPConvolution(
                    hidden_irreps=Irreps(self.hidden_irreps),
                    sh_irreps=Irreps(self.sh_irreps),
                    node_irreps=Irreps(self.node_irreps),
                    use_sc=self.use_sc,
                    nonlinearities=self.nonlinearities,
                    radial_net_nonlinearity=self.radial_net_nonlinearity,
                    radial_net_n_hidden=self.radial_net_n_hidden,
                    radial_net_n_layers=self.radial_net_n_layers,
                    num_basis=self.num_basis,
                    n_neighbors=self.n_neighbors,
                    # scalar_mlp_std=self.scalar_mlp_std,
                )
                for _ in range(self.graph_net_steps)
            ]
        )
        # output block, two Linear layers that decay dimensions from h to h//2 to 1
        for mul, ir in Irreps(self.hidden_irreps):
            if ir == Irrep("0e"):
                mul_second_to_final = mul // 2

        second_to_final_irreps = Irreps(f"{mul_second_to_final}x0e")
        final_irreps = Irreps("1x0e")
        self.lin_2 = Linear(
            irreps_in=Irreps(self.hidden_irreps),
            irreps_out=second_to_final_irreps,
        )
        self.lin_3 = Linear(
            irreps_in=second_to_final_irreps,
            irreps_out=final_irreps,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_1.weight.data.normal_(mean=0.0, std=0.02)
        self.bassel_basis.reset_parameters()
        for conv_layer in self.conv_layers:
            conv_layer.reset_parameters()
        self.lin_2.weight.data.normal_(mean=0.0, std=0.02)
        self.lin_3.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        # calculate relative vectors
        edge_src, edge_dst = data.edge_index[0], data.edge_index[1]  # [B_e], [B_e]

        # convert str to Irreps
        hidden_irreps = Irreps(self.hidden_irreps)
        sh_irreps = Irreps(self.sh_irreps)

        # node features

        atomic_numbers = data.x  # [n_nodes, 1]
        one_hot = torch.zeros((atomic_numbers.shape[0], self.n_elements)).to(
            self.device
        )
        one_hot[torch.arange(atomic_numbers.shape[0]), atomic_numbers] = 1.0
        node_attrs = IrrepsArray(
            irreps=self.node_irreps, array=one_hot
        )  # [B_n, n_elements]

        # edge embeddings
        dR = data.relative_vec
        edge_sh = e3nn.o3.spherical_harmonics(
            sh_irreps.ls, dR, True, normalization="component"
        )
        edge_sh = IrrepsArray(irreps=sh_irreps, array=edge_sh)  # [B_e, num_sh_coeff]

        scalar_dr_edge = torch.norm(dR, dim=-1)
        # add threshold to avoid division by zero
        scalar_dr_edge = torch.where(
            scalar_dr_edge < 1e-5, 1e-5, scalar_dr_edge
        )  # [B_e]

        # bedge embedding (essel basis functions)
        embedding_dr_edge = self.bassel_basis(scalar_dr_edge)  # [B_e, num_basis]

        # embedding layer
        h_node = self.lin_1(node_attrs.array)  # [n_nodes, hidden_irreps]
        h_node = IrrepsArray(
            irreps=hidden_irreps, array=h_node
        )  # [n_nodes, hidden_irreps]
        # conv layers
        for conv_layer in self.conv_layers:
            h_node = conv_layer(
                h_node,
                node_attrs,
                edge_sh,
                edge_src,
                edge_dst,
                embedding_dr_edge,
            )  # [B_n, hidden_irreps]
        # readout
        h_node_mid = self.lin_2(h_node.array)
        atomic_output = self.lin_3(h_node_mid)  # [B_n, 1]
        # pooling (aggregate atomic output to graph output)
        graph_output = scatter(
            atomic_output,
            data.batch,
            dim=0,
            reduce="sum",  # TODO:check pooling original code is sum
        )  # [B, 1]
        return graph_output
