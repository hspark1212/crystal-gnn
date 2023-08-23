from typing import Any, Dict

from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from dgl.nn import Set2Set
import torch
import torch.nn as nn

from crystal_gnn.models.base_module import BaseModule
from crystal_gnn.layers.utils import RBFExpansion
from crystal_gnn.layers import MLPReadout
from crystal_gnn.experimental.megnet.megnet_layer import MEGNETLayer


class MEGNET(BaseModule):
    """MEGNET model.

    For simplicity, dimensions of node, edge, state embeddings are
    all set to hidden_dim, which is different from the original paper.

    "Graph Networks as a Universal Machine Learning Framework
    for Molecules and Crystals"
    Chem. Mater. (2019)
    https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
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
        self.edge_embedding = nn.Linear(self.rbf_distance_dim, self.hidden_dim)
        self.state_embedding = nn.Linear(2, self.hidden_dim)
        self.rbf_expansion = RBFExpansion(bins=self.rbf_distance_dim)
        self.conv_layers = nn.ModuleList(
            [
                MEGNETLayer(
                    hidden_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    residual=self.residual,
                    dropout=self.dropout,
                )
                for _ in range(self.num_conv)
            ]
        )
        self.node_set2set = Set2Set(self.hidden_dim, n_layers=1, n_iters=2)

        self.edge_set2set = EdgeSet2Set(self.hidden_dim, n_layers=1, n_iters=2)
        self.fc = nn.Linear(self.hidden_dim * 5, self.hidden_dim)
        self.readout = MLPReadout(self.hidden_dim, self.readout_dim)

        self.apply(self._init_weights)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward propagation.

        Args:
            batch (Dict[str, Any]):  batch data including graph, line graph (optional)
            and target

        Returns:
            torch.Tensor: predicted target values (logits)
        """
        graph = batch["graph"]
        # node embedding
        node_attrs = graph.ndata["atomic_number"]
        node_feats = self.node_embedding(node_attrs)
        # edge embedding
        edge_attrs = self.rbf_expansion(graph.edata["distance"])
        edge_feats = self.edge_embedding(edge_attrs)
        # global state embedding
        state_attrs = torch.zeros(graph.batch_size, 2).to(self.device)  # placeholder
        state_feats = self.state_embedding(state_attrs)  # [N', hidden_dim]
        # conv layers
        for conv in self.conv_layers:
            node_feats, edge_feats, state_feats = conv(
                graph, node_feats, edge_feats, state_feats
            )
        # pooling
        node_feats = self.node_set2set(graph, node_feats)  # [batch_size, hidden_dim*2]
        edge_feats = self.edge_set2set(graph, edge_feats)  # [batch_size, hidden_dim*2]
        # readout
        total_feats = torch.cat([node_feats, edge_feats, state_feats], dim=1)
        total_feats = self.fc(total_feats)
        total_feats = self.readout(total_feats)
        return total_feats


class EdgeSet2Set(nn.Module):
    """Implemented in MATGL
    https://github.com/materialsvirtuallab/matgl/blob/main/matgl/layers/_core.py#L127
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """:param input_dim: The size of each input sample.
        :param n_iters: The number of iterations.
        :param n_layers: The number of recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, g: DGLGraph, feat: torch.Tensor):
        """Defines the computation performed at every call.

        :param g: Input graph
        :param feat: Input features.
        :return: One hot vector
        """
        with g.local_scope():
            batch_size = g.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(g, q)).sum(dim=-1, keepdim=True)
                g.edata["e"] = e
                alpha = softmax_edges(g, "e")
                g.edata["r"] = feat * alpha
                readout = sum_edges(g, "r")
                q_star = torch.cat([q, readout], dim=-1)

            return q_star
