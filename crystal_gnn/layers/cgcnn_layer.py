from typing import Dict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNLayer(nn.Module):
    """CGCNN layer.

    "Crystal Graph Convolutional Neural Networks for an Accurate and
    Interpretable Prediction of Material Properties"
    Phys. Rev. Lett. (2018).
    https://doi.org/10.1103/PhysRevLett.120.145301
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_feat_dim: int,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim (int): dimension of hidden features
            edge_feat_dim (int): dimension of edge attributes (RBF expansion of distances)
            batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
            residual (bool, optional): whether to use residual connection. Defaults to False.
            dropout (float, optional): a ratio of dropout. Defaults to 0.0.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        z_dim = hidden_dim * 2 + edge_feat_dim
        self.fc = nn.Linear(z_dim, hidden_dim * 2)
        self.bn = nn.BatchNorm1d(hidden_dim * 2)

    def message_func(self, edges: dgl.udf.EdgeBatch) -> Dict[str, torch.Tensor]:
        z = torch.cat([edges.src["h"], edges.dst["h"], edges.data["e"]], dim=1)
        z = self.fc(z)
        if self.batch_norm:
            z = self.bn(z)
        nbr_filter, nbr_core = torch.chunk(z, 2, dim=1)
        nbr_filter = torch.sigmoid(nbr_filter)
        nbr_core = F.softplus(nbr_core)
        return {"nbr_filter": nbr_filter, "nbr_core": nbr_core}

    def reduce_func(self, nodes: dgl.udf.NodeBatch) -> Dict[str, torch.Tensor]:
        h = torch.sum(nodes.mailbox["nbr_filter"] * nodes.mailbox["nbr_core"], dim=1)
        if self.residual:
            h = nodes.data["h"] + h
        h = F.softplus(h)
        return {"h": h}

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embeded edge features

        Returns:
            torch.Tensor: updated node features
        """
        g.ndata["h"] = h
        g.edata["e"] = e
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]

        h = F.dropout(h, self.dropout, training=self.training)
        return h
