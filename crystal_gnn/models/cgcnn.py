from typing import Any, Dict

from dgl.nn.pytorch.glob import AvgPooling
import torch
import torch.nn.functional as F
import torch.nn as nn

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.layers import CGCNNLayer, MLPReadout
from crystal_gnn.layers.utils import RBFExpansion


class CGCNN(BaseModule):
    """CGCNN model.

    "Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties"
    Phys. Rev. Lett. (2018).
    https://doi.org/10.1103/PhysRevLett.120.145301
    """

    def __init__(self, _config: Dict[str, Any]):
        super().__init__(_config)
        # config
        self.num_conv = _config["num_conv"]
        self.hidden_dim = _config["hidden_dim"]
        self.rbf_distance_dim = _config["rbf_distance_dim"]
        self.batch_norm = _config["batch_norm"]
        self.dropout = _config["dropout"]
        self.residual = _config["residual"]
        # layers
        self.node_embedding = nn.Embedding(103, self.hidden_dim)
        self.rbf_expansion = RBFExpansion(bins=self.rbf_distance_dim)
        self.conv_layers = nn.ModuleList(
            [
                CGCNNLayer(
                    hidden_dim=self.hidden_dim,
                    edge_feat_dim=self.rbf_distance_dim,
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
        # node embedding
        node_attrs = graph.ndata["atomic_number"]
        node_feats = self.node_embedding(node_attrs)
        # edge embedding
        edge_feats = self.rbf_expansion(graph.edata["distance"])
        # conv layers
        for conv_layer in self.conv_layers:
            node_feats = conv_layer(graph, node_feats, edge_feats)
        # pooling
        node_feats = self.pooling(graph, node_feats)
        # readout
        node_feats = F.softplus(self.fc(F.softplus(node_feats)))
        node_feats = self.readout(node_feats)
        return node_feats
