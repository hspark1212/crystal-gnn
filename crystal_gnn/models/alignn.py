from typing import Any, Dict

from dgl.nn.pytorch.glob import AvgPooling
import torch
import torch.nn as nn

from crystal_gnn.layers import ALIGNNLayer, GatedGCNLayer, MLPReadout
from crystal_gnn.layers.utils import RBFExpansion
from crystal_gnn.models.base_module import BaseModule


class ALIGNN(BaseModule):
    """ALIGNN model.

    "Atomistic Line Graph Neural Network
    for improved materials property predictions"
    npj Comput. Mater. (2021).
    https://www.nature.com/articles/s41524-021-00650-1
    """

    def __init__(self, _config: Dict[str, Any]):
        super().__init__(_config)
        # config
        self.num_conv = _config["num_conv"]
        self.hidden_dim = _config["hidden_dim"]
        self.rbf_distance_dim = _config["rbf_distance_dim"]
        self.rbf_triplet_dim = _config["rbf_triplet_dim"]
        self.batch_norm = _config["batch_norm"]
        self.dropout = _config["dropout"]
        self.residual = _config["residual"]

        # layers
        self.node_embedding = nn.Embedding(103, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.rbf_distance_dim, self.hidden_dim)
        self.angle_embedding = nn.Linear(self.rbf_triplet_dim, self.hidden_dim)
        self.rbf_expansion_distance = RBFExpansion(
            vmin=0, vmax=8, bins=self.rbf_distance_dim
        )
        self.rbf_expansion_triplet = RBFExpansion(
            vmin=-1, vmax=1, bins=self.rbf_triplet_dim
        )
        self.conv_layers = nn.ModuleList(
            [
                ALIGNNLayer(
                    input_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.gated_gcn_layers = nn.ModuleList(
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
        self.pooling = AvgPooling()
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.readout = MLPReadout(self.hidden_dim, self.readout_dim)

        self.apply(self._init_weights)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward propagation.

        Args:
            batch (Dict[str, Any]): batch data including graph, line graph (optional)
            and target

        Returns:
            torch.Tensor: predicted target values (logits)
        """
        graph = batch["graph"]
        line_graph = batch["line_graph"]

        # node embedding
        node_attrs = graph.ndata["atomic_number"]
        node_feats = self.node_embedding(node_attrs)
        # edge embedding
        edge_attrs = self.rbf_expansion_distance(graph.edata["distance"])
        edge_feats = self.edge_embedding(edge_attrs)
        # angle (edge pair) embedding
        angle_attrs = self.rbf_expansion_triplet(line_graph.edata["angle"])
        angle_feats = self.angle_embedding(angle_attrs)
        # conv layers
        for conv in self.conv_layers:
            node_feats, edge_feats, angle_feats = conv(
                graph, line_graph, node_feats, edge_feats, angle_feats
            )
        # gated gcn layers
        for gated_gcn in self.gated_gcn_layers:
            node_feats, _ = gated_gcn(graph, node_feats, edge_feats)
        # pooling
        node_feats = self.pooling(graph, node_feats)
        # readout
        node_feats = self.fc(node_feats)
        node_feats = self.readout(node_feats)
        return node_feats
