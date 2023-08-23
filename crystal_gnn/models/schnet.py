from typing import Any, Dict

from dgl.nn.pytorch.glob import AvgPooling
import torch
import torch.nn as nn

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.layers.utils import RBFExpansion
from crystal_gnn.layers import SCHNETLayer, MLPReadout
from crystal_gnn.layers.schnet_layer import shifted_softplus


class SCHNET(BaseModule):
    """SCHNET Model.

    "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions"
    Arxiv (2017).
    https://arxiv.org/abs/1706.08566
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
                SCHNETLayer(
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
        self.fc = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.readout = MLPReadout(int(self.hidden_dim / 2), self.readout_dim)

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

        #######################################
        # the following code is from origin paper, but poor performance.
        # readout
        # node_feats = self.fc(node_feats)
        # node_feats = shifted_softplus(node_feats)
        # node_feats = self.readout(node_feats)
        # sum pooling
        # node_feats = self.pooling(graph, node_feats)
        #######################################
        # pooling
        node_feats = self.pooling(graph, node_feats)
        # readout
        node_feats = self.fc(node_feats)
        node_feats = shifted_softplus(node_feats)
        node_feats = self.readout(node_feats)
        return node_feats
