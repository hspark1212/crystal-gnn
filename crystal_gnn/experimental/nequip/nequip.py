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


# Types
Array = torch.Tensor

NONLINEARITY = {
    "none": lambda x: x,
    "relu": nn.functional.relu,
    # "swish": BetaSwish(),
    "raw_swish": nn.functional.silu,
    "tanh": nn.functional.tanh,
    "sigmoid": nn.functional.sigmoid,
    "silu": nn.functional.silu,  # TODO: check if this is the same as raw_swish
}


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
        self.hidden_irreps = hidden_irreps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.n_neighbors = n_neighbors
        # self.scalar_mlp_std = scalar_mlp_std

    def forward(
        self,
        node_features: IrrepsArray,
        node_attributes: IrrepsArray,
        edge_sh: IrrepsArray,
        edge_src: Array,
        edge_dst: Array,
        edge_embedded: Array,
    ) -> IrrepsArray:
        irreps_scalars = []
        irreps_nonscalars = []
        irreps_gate_scalars = []

        # get scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if Irrep(irrep).l == 0 and tp_path_exists(
                node_features.irreps, edge_sh.irreps, irrep
            ):
                irreps_scalars += [(multiplicity, irrep)]

        irreps_scalars = Irreps(irreps_scalars)

        # get non-scalar target irreps
        for multiplicity, irrep in self.hidden_irreps:
            # need the additional Irrep() here for the build, even though irrep is
            # already of type Irrep()
            if Irrep(irrep).l > 0 and tp_path_exists(
                node_features.irreps, edge_sh.irreps, irrep
            ):
                irreps_nonscalars += [(multiplicity, irrep)]

        irreps_nonscalars = Irreps(irreps_nonscalars)

        # get gate scalar irreps
        if tp_path_exists(node_features.irreps, edge_sh.irreps, "0e"):
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
            self_connection = FullyConnectedTensorProduct(
                irreps_in1=node_features.irreps,
                irreps_in2=node_attributes.irreps,
                irreps_out=h_out_irreps,
            )(node_features.array, node_attributes.array)

        h = node_features.array

        # first linear, stays in current h-space
        h = Linear(
            irreps_in=node_features.irreps,
            irreps_out=node_features.irreps,
            internal_weights=True,
            shared_weights=True,
        )(h)

        # map node features onto edges for tp
        edge_features = h[edge_src]  # [n_edges, hidden_irreps]

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
        for i, (mul_in1, irreps_in1) in enumerate(node_features.irreps):
            for j, (_, irreps_in2) in enumerate(edge_sh.irreps):
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
        tp = TensorProduct(
            irreps_in1=node_features.irreps,
            irreps_in2=edge_sh.irreps,
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
        for ins in tp.instructions:
            if ins.has_weight:
                n_tp_weights += prod(ins.path_shape)

        # build radial MLP R(r) that maps from interatomic distance to TP weights
        # must not use biases to that R(0) = 0
        fc = FullyConnectedNet(
            (edge_embedded.shape[-1],)  # TODO: check this.
            + (self.radial_net_n_hidden,) * self.radial_net_n_layers
            + (n_tp_weights,),
            NONLINEARITY[self.radial_net_nonlinearity],
        )

        # the TP weights (v dimension) are given by the FC
        weight = fc(edge_embedded)  # [n_edges, n_tp_weights]

        #
        edge_features = tp(
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
        h = h / self.n_neighbors

        # second linear, now we create extra gate scalars by mapping to h-out
        h = Linear(
            irreps_in=irreps_after_tp.simplify(),
            irreps_out=h_out_irreps,
            internal_weights=True,
            shared_weights=True,
        )(h)

        # self-connection, similar to a resnet-update that sums the output from
        # the TP to chemistry-weighted h
        if self.use_sc:
            h = h + self_connection

        # gate nonlinearity, applied to gate data, consisting of:
        # a) regular scalars,
        # b) gate scalars, and
        # c) non-scalars to be gated
        # in this order
        nonlinearities = {
            1: self.nonlinearities["e"],
            -1: self.nonlinearities["o"],
        }
        equivariant_nonlin = Gate(
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

        h = equivariant_nonlin(h)

        h_node = IrrepsArray(irreps=equivariant_nonlin.irreps_out, array=h)

        return h_node


class NequIPEnergyModel(nn.Module):
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
        super().__init__()
        self.graph_net_steps: int = _config["graph_net_steps"]
        self.use_sc: bool = _config["use_sc"]
        self.nonlinearities: Union[str, Dict[str, str]] = _config["nonlinearities"]
        self.n_elements: int = _config["n_elements"]

        self.hidden_irreps: str = _config["hidden_irreps"]
        self.sh_irreps: str = _config["sh_irreps"]

        self.num_basis: int = _config["num_basis"]
        self.r_max: float = _config["r_max"]

        self.radial_net_nonlinearity: str = _config["radial_net_nonlinearity"]
        self.radial_net_n_hidden: int = _config["radial_net_n_hidden"]
        self.radial_net_n_layers: int = _config["radial_net_n_layers"]

        self.shift: float = 0.0
        self.scale: float = 1.0
        self.n_neighbors: float = 1.0
        self.scalar_mlp_std: float = 4.0

    def forward(self, graph: Union[Data, Batch]) -> torch.Tensor:
        """Forward pass of NequIP.

        Args:
            graph: input graph

        Returns:
            Potential energy of the inputs.
        """
        # calculate relative vectors
        edge_src, edge_dst = graph.edge_index[0], graph.edge_index[1]

        # convert str to Irreps
        hidden_irreps = Irreps(self.hidden_irreps)
        sh_irreps = Irreps(self.sh_irreps)

        # node features
        node_irreps = Irreps(f"{self.n_elements}x0e")
        atomic_numbers = graph.x  # [n_nodes, 1]
        one_hot = torch.zeros((atomic_numbers.shape[0], self.n_elements))
        one_hot[torch.arange(atomic_numbers.shape[0]), atomic_numbers] = 1.0
        node_attrs = IrrepsArray(
            irreps=node_irreps, array=one_hot
        )  # [n_nodes, n_elements]

        # edge embeddings
        dR = graph.relative_vec
        edge_sh = e3nn.o3.spherical_harmonics(
            sh_irreps.ls, dR, True, normalization="component"
        )
        edge_sh = IrrepsArray(
            irreps=sh_irreps, array=edge_sh
        )  # [n_edges, num_sh_coeff]

        scalar_dr_edge = torch.norm(dR, dim=-1)
        # add threshold to avoid division by zero
        scalar_dr_edge = torch.where(scalar_dr_edge < 1e-5, 1e-5, scalar_dr_edge)

        # bessel basis functions
        embedding_dr_edge = BesselBasis(
            r_max=self.r_max, num_basis=self.num_basis, trainable=True
        )(
            scalar_dr_edge
        )  # [n_edges, num_basis]

        # embedding layer
        linear = e3nn.o3.Linear(
            irreps_in=node_irreps,
            irreps_out=hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        h_node = linear(node_attrs.array)
        h_node = IrrepsArray(
            irreps=hidden_irreps, array=h_node
        )  # [n_nodes, hidden_irreps]

        # convolutional layers
        for _ in range(self.graph_net_steps):
            h_node = NequIPConvolution(
                hidden_irreps=hidden_irreps,
                use_sc=self.use_sc,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis,
                n_neighbors=self.n_neighbors,
                # scalar_mlp_std=self.scalar_mlp_std,
            )(
                h_node,
                node_attrs,
                edge_sh,
                edge_src,
                edge_dst,
                embedding_dr_edge,
            )

        # output block, two Linear layers that decay dimensions from h to h//2 to 1
        for mul, ir in h_node.irreps:
            if ir == Irrep("0e"):
                mul_second_to_final = mul // 2

        second_to_final_irreps = Irreps(f"{mul_second_to_final}x0e")
        final_irreps = Irreps("1x0e")

        h_node_mid = Linear(
            irreps_in=h_node.irreps,
            irreps_out=second_to_final_irreps,
        )(h_node.array)
        atomic_output = Linear(
            irreps_in=second_to_final_irreps,
            irreps_out=final_irreps,
        )(h_node_mid)

        # shift + scale atomic energies
        atomic_output = atomic_output * self.scale + self.shift  # TODO: normalizer

        # aggregate atomic output to graph output
        graph_output = scatter(
            atomic_output,
            graph.batch,
            dim=0,
            reduce="sum",
        )

        return graph_output
