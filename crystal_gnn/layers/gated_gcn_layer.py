import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class GatedGCNLayer(MessagePassing):
    """ResGatedGCN: Residual Gated Graph ConvNets

    "Residual Gated Graph ConvNets"
    ICLR (2018)
    https://openreview.net/forum?id=HyXBcYg0b
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        h = node_feats  # [B_n, H]
        Ah = self.A(node_feats)  # [B_n, H]
        Bh = self.B(node_feats)  # [B_n, H]
        Dh = self.D(node_feats)  # [B_n, H]
        Eh = self.E(node_feats)  # [B_n, H]
        e = edge_feats  # [B_e, H]
        Ce = self.C(edge_feats)  # [B_e, H]

        out_h, out_e = self.propagate(
            edge_index, h=h, Ah=Ah, Bh=Bh, Dh=Dh, Eh=Eh, e=e, Ce=Ce
        )
        # batch norm
        if self.batch_norm:
            out_h = self.bn_node_h(out_h)
            out_e = self.bn_node_e(out_e)
        # nonlinearity
        out_h = F.silu(out_h)
        out_e = F.silu(out_e)
        # residual connection
        if self.residual:
            out_h = h + out_h
            out_e = e + out_e
        # dropout
        out_h = F.dropout(out_h, p=self.dropout, training=self.training)
        out_e = F.dropout(out_e, p=self.dropout, training=self.training)

        return out_h, out_e

    def message(
        self,
        Bh_i: Tensor,
        Dh_j: Tensor,
        Eh_j: Tensor,
        Ce: Tensor,
    ) -> Tensor:
        # update edge features {e^_ij}
        e = Dh_j + Eh_j + Ce  # [B_e, H]
        # sigma
        sigma = torch.sigmoid(e)  # [B_e, H]
        # numerator
        sigma_h = Bh_i * sigma  # [B_e, H]
        return sigma_h, sigma, e

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
    ) -> Tensor:
        sigma_h, sigma, e = inputs
        return sigma_h, sigma, e, index

    def update(self, aggr_out: Tensor, Ah: Tensor) -> Tensor:
        sigma_h, sigma, out_e, index = aggr_out  # [B_e, H], [B_e, H], [B_e, H], [B_e]
        # aggregate
        sum_sigma_h = scatter(
            sigma_h, index, dim=0, dim_size=Ah.size(0), reduce="sum"
        )  # [B_n, H]
        sum_sigma = scatter(
            sigma, index, dim=0, dim_size=Ah.size(0), reduce="sum"
        )  # [B_n, H]
        # update
        out_h = Ah + sum_sigma_h / (sum_sigma + 1e-6)  # [B_n, H]
        return out_h, out_e
