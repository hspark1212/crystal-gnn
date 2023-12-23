from typing import Union, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.models.module_utils import RBFExpansion
from crystal_gnn.layers.mlp_readout import MLPReadout

fn_softplus = nn.Softplus()


class CGCNN(BaseModule):
    """CGCNN model.

    "Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties"
    Phys. Rev. Lett. (2018).
    https://doi.org/10.1103/PhysRevLett.120.145301
    """

    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)
        # config
        self.num_conv = _config["num_conv"]
        self.hidden_dim = _config["hidden_dim"]
        self.rbf_distance_dim = _config["rbf_distance_dim"]
        self.batch_norm = _config["batch_norm"]
        self.dropout = _config["dropout"]
        self.residual = _config["residual"]
        self.cutoff = _config["cutoff"]
        # layers
        self.node_embedding = nn.Embedding(103, self.hidden_dim)
        self.rbf_expansion = RBFExpansion(
            vmin=0, vmax=self.cutoff, bins=self.rbf_distance_dim
        )
        self.conv_layers = nn.ModuleList(
            [
                CGCNNlayer(
                    hidden_dim=self.hidden_dim,
                    edge_feat_dim=self.rbf_distance_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.avg_pool = global_mean_pool
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.readout = MLPReadout(self.hidden_dim, self.readout_dim, bias=True)

        self.apply(self._init_weights)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        # node embedding
        node_attrs = data.x  # [B_n]
        node_feats = self.node_embedding(node_attrs)  # [B_n, H]
        # edge embedding
        distances = torch.norm(data.relative_vec, dim=-1)  # [B_e]
        edge_feats = self.rbf_expansion(distances)  # [B_e, D]
        # conv layers
        for conv_layer in self.conv_layers:
            node_feats = conv_layer(node_feats, edge_feats, data.edge_index)  # [B_n, H]
        # pooling
        node_feats = self.avg_pool(node_feats, data.batch)  # [B, H]
        # readout
        node_feats = fn_softplus(self.lin(fn_softplus(node_feats)))  # [B, H]
        node_feats = self.readout(node_feats)  # [B, O]
        return node_feats


class CGCNNlayer(MessagePassing):
    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        batch_norm: bool,
        residual: bool,
        dropout: float,
    ) -> None:
        super().__init__(aggr="add")
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        z_dim = hidden_dim * 2 + edge_feat_dim
        self.lin = nn.Linear(z_dim, 2 * hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: features of nodes [B_n, H]
            edge_feats: features of edges [B_e, D]
            edge_index: edge index [2, B_e]
        Returns:
            torch.Tensor: updated node features [B_n, H]
        """
        out = self.propagate(edge_index, h=node_feats, edge_feats=edge_feats)
        # dropout
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def message(self, h_i: Tensor, h_j: Tensor, edge_feats: Tensor) -> Tensor:
        z = torch.cat([h_i, h_j, edge_feats], dim=1).to(h_i.dtype)  # [B_e, 2H+D]
        z = self.lin(z)  # [B_e, 2H]
        nbr_filter, nbr_core = torch.chunk(z, 2, dim=1)  # [B_e, H]
        nbr_filter = torch.sigmoid(nbr_filter)  # [B_e, H]
        nbr_core = fn_softplus(nbr_core)  # [B_e, H]
        return nbr_filter * nbr_core  # [B_e, H]

    def update(self, aggr_out: Tensor, h: Tensor) -> Tensor:
        # batch norm
        if self.batch_norm:
            aggr_out = self.bn(aggr_out)  # [B_n, H]
        # residual connection
        if self.residual:
            aggr_out = aggr_out + h
        # aggr_out = fn_softplus(aggr_out)
        return aggr_out  # [B_n, H]
