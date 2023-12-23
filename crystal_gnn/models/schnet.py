from typing import Union, Dict, Any
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models.schnet import InteractionBlock

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.models.module_utils import RBFExpansion, ShiftedSoftplus


class SCHNET(BaseModule):
    """SCHNET Model.

    "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions"
    Arxiv (2017).
    https://arxiv.org/abs/1706.08566
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
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=self.hidden_dim,
                    num_gaussians=self.rbf_distance_dim,
                    num_filters=self.hidden_dim,
                    cutoff=self.cutoff,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.sum_pool = global_add_pool
        self.lin_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.lin_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.shift_softplus = ShiftedSoftplus()
        self.readout_lin_1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2, bias=True)
        self.readout_lin_2 = nn.Linear(
            self.hidden_dim // 2, self.readout_dim, bias=True
        )

        self.apply(self._init_weights)

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        # node embedding
        node_attrs = data.x  # [B_n]
        node_feats = self.node_embedding(node_attrs)  # [B_n, H]
        # edge embedding
        distances = torch.norm(data.relative_vec, dim=-1)  # [B_e]
        edge_feats = self.rbf_expansion(distances)  # [B_e, D]
        # conv layers
        for interaction_block in self.interaction_blocks:
            node_feats = interaction_block(
                node_feats,
                data.edge_index,
                distances,
                edge_feats,
            )  # [B_n, H]
            if self.residual:
                node_feats += node_feats
        # TODO: add batch norm, dropout
        node_feats = self.lin_1(node_feats)  # [B, H]
        node_feats = self.shift_softplus(node_feats)  # [B, H]
        node_feats = self.lin_2(node_feats)  # [B, H]
        # pool
        node_feats = self.sum_pool(node_feats, data.batch)  # [B, H]
        # readout
        node_feats = self.readout_lin_1(node_feats)  # [B, H//2]
        node_feats = self.shift_softplus(node_feats)  # [B, H//2]
        node_feats = self.readout_lin_2(node_feats)  # [B, O]
        return node_feats


# TODO: deprecated
class InteractionBlock_old(nn.Module):
    """Interaction block of SchNet."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        batch_norm: bool,
        residual: bool,
        dropout: float,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        self.cfconv = CFconv(
            hidden_dim,
            edge_feat_dim,
            cutoff,
            batch_norm,
            residual,
            dropout,
        )
        self.shift_softplus = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        distances: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_feats: features of nodes [B_n, H]
            edge_feats: features of edges [B_e, D]
            distances: distances of edges [B_e]
            edge_index: edge index [2, B_e]
        Returns:
            torch.Tensor: updated node features [B_n, H]
        """
        orig_node_feats = node_feats.clone()
        # conv
        node_feats = self.cfconv(
            node_feats, edge_feats, distances, edge_index
        )  # [B_n, H]
        node_feats = self.shift_softplus(node_feats)  # [B_n, H]
        node_feats = self.lin(node_feats)  # [B_n, H]
        # batch norm
        if self.batch_norm:
            node_feats = self.bn(node_feats)  # [B_n, H]
        # residual connection
        if self.residual:
            node_feats += orig_node_feats  # [B_n, H]
        # dropout
        node_feats = F.dropout(
            node_feats, p=self.dropout, training=self.training
        )  # [B_n, H]
        return node_feats


class CFconv(MessagePassing):
    """Implemented in torch_geometric.nn.models.schnet."""

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        cutoff: float,
        batch_norm: bool,
        residual: bool,
        dropout: float,
    ) -> None:
        super().__init__(aggr="add")
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        self.cutoff = cutoff
        self.mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lin_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.lin_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        node_feats: Tensor,
        edge_feats: Tensor,
        distances: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """
        Args:
            node_feats: features of nodes [B_n, H]
            edge_feats: features of edges [B_e, D]
            distances: distances of edges [B_e]
            edge_index: edge index [2, B_e]
        Returns:
            node_feats: updated node features [B_n, H]
        """
        C = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)  # [B_e]
        edge_feats = self.mlp(edge_feats) * C.view(-1, 1)  # [B_e, H]
        node_feats = self.lin_1(node_feats)  # [B_n, H]
        node_feats = self.propagate(
            edge_index, h=node_feats, edge_feats=edge_feats
        )  # [B_n, H]
        node_feats = self.lin_2(node_feats)  # [B_n, H]
        # batch norm
        if self.batch_norm:
            node_feats = self.bn(node_feats)
        # residual connection
        if self.residual:
            node_feats += node_feats
        # dropout
        node_feats = F.dropout(node_feats, p=self.dropout, training=self.training)
        return node_feats

    def message(self, h_j: Tensor, edge_feats: Tensor) -> Tensor:
        return h_j * edge_feats  # [B_e, H]
