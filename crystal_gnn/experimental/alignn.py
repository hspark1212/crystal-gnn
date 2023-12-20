from typing import Union, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import LineGraph

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.models.module_utils import RBFExpansion
from crystal_gnn.layers.mlp_readout import MLPReadout
from crystal_gnn.layers.gated_gcn_layer import GatedGCNLayer


class ALIGNN(BaseModule):
    """ALIGNN model.

    "Atomistic Line Graph Neural Network
    for improved materials property predictions"
    npj Comput. Mater. (2021).
    https://www.nature.com/articles/s41524-021-00650-1
    """

    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)
        # config
        self.num_conv = _config["num_conv"]
        self.hidden_dim = _config["hidden_dim"]
        self.rbf_distance_dim = _config["rbf_distance_dim"]
        self.rbf_triplet_dim = _config[
            "rbf_distance_dim"
        ]  # suppose rbf_triplet_dim = rbf_distance_dim
        self.batch_norm = _config["batch_norm"]
        self.dropout = _config["dropout"]
        self.residual = _config["residual"]
        self.cutoff = _config["cutoff"]
        # layers
        self.line_graph = LineGraph(force_directed=True)
        self.node_embedding = nn.Embedding(103, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.rbf_distance_dim, self.hidden_dim)
        self.angle_embedding = nn.Linear(self.rbf_triplet_dim, self.hidden_dim)
        self.rbf_expansion_distance = RBFExpansion(
            vmin=0, vmax=self.cutoff, bins=self.rbf_distance_dim
        )
        self.rbf_expansion_triplet = RBFExpansion(
            vmin=-1, vmax=1, bins=self.rbf_triplet_dim
        )
        self.conv_layers = nn.ModuleList(
            [
                ALIGNNlayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.gated_gcn_layer = nn.ModuleList(
            [
                GatedGCNLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.avg_pool = global_mean_pool
        self.lin = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.readout = MLPReadout(self.hidden_dim, self.readout_dim, bias=False)

    def reset_parameters(self) -> None:
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.angle_embedding.reset_parameters()
        self.rbf_expansion_distance.reset_parameters()
        self.rbf_expansion_triplet.reset_parameters()
        for conv_layer in self.conv_layers:
            conv_layer.reset_parameters()
        for gated_gcn_layer in self.gated_gcn_layer:
            gated_gcn_layer.reset_parameters()
        self.lin.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        if isinstance(data, Batch):
            data_lg = self.convert_line_graph(data)
        else:
            data_lg = self.line_graph(data)
        # node embedding
        node_attrs = data.x  # [B_n]
        node_feats = self.node_embedding(node_attrs)  # [B_n, H]
        # edge embedding
        distances = torch.norm(data.relative_vec, dim=-1)  # [B_e]
        edge_feats = self.rbf_expansion_distance(distances)  # [B_e, D]
        edge_feats = self.edge_embedding(edge_feats)  # [B_e, H]
        # angle embedding
        angles = calculate_angle(data_lg)  # [B_lg_e]
        angle_feats = self.rbf_expansion_triplet(angles)  # [B_lg_e, D]
        angle_feats = self.angle_embedding(angle_feats)  # [B_lg_e, H]
        # conv layers
        for conv_layer in self.conv_layers:
            node_feats, edge_feats, angle_feats = conv_layer(
                data, data_lg, node_feats, edge_feats, angle_feats
            )  # [B_n, H], [B_e, H], [B_lg_e, H]
        # gated gcn layers
        for gated_gcn_layer in self.gated_gcn_layer:
            node_feats, _ = gated_gcn_layer(
                node_feats, edge_feats, data.edge_index
            )  # [B_n, H], [B_e, H]
        # pooling
        node_feats = self.avg_pool(node_feats, data.batch)  # [B, H]
        # readout
        node_feats = self.lin(node_feats)  # [B, H]
        node_feats = F.silu(node_feats)
        out = self.readout(node_feats)  # [B, O]
        return out

    def convert_line_graph(self, data: Batch):
        d_lg_list = []
        for d in data.to_data_list():
            d_lg = self.line_graph(d)
            d_lg_list.append(d_lg)
        return Batch.from_data_list(d_lg_list)


class ALIGNNlayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_update = GatedGCNLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            batch_norm=batch_norm,
            residual=residual,
            dropout=dropout,
        )
        self.edge_update = GatedGCNLayer(
            input_dim=output_dim,
            output_dim=input_dim,
            batch_norm=batch_norm,
            residual=residual,
            dropout=dropout,
        )

    def forward(
        self,
        graph: Union[Data, Batch],
        line_graph: Union[Data, Batch],
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        angle_feats: torch.Tensor,
    ) -> torch.Tensor:
        # node update
        node_feats, edge_feats = self.node_update(
            node_feats, edge_feats, graph.edge_index
        )  # [B_n, H], [B_e, H]
        # edge update
        edge_feats, angle_feats = self.edge_update(
            edge_feats, angle_feats, line_graph.edge_index
        )  # [B_e, H], [B_lg_e, H]

        return node_feats, edge_feats, angle_feats


def calculate_angle(data_lg: Union[Data, Batch]) -> Tensor:
    """Calculate angle between two bonds

    Args:
        data_lg: line graph data

    Returns:
        angle: angle between two bonds
    """
    # data_lg.x is relative_vec in line graph
    idx_src, idx_dst = data_lg.edge_index
    r1 = -data_lg.relative_vec[idx_src]
    r2 = data_lg.relative_vec[idx_dst]
    # calculate cosine of angle
    bond_consine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1) + 1e-8
    )
    angles = torch.clamp(bond_consine, -1, 1)
    return angles
