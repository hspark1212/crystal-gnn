import dgl
from dgl import function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCHNETLayer(nn.Module):
    """SCHNET layer. Interaction block in SCHNET.

    "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions"
    Arxiv (2017).
    https://arxiv.org/abs/1706.08566
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
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        self.node_func = nn.Linear(hidden_dim, hidden_dim)
        self.edge_func_1 = nn.Linear(edge_feat_dim, hidden_dim)
        self.edge_func_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

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
        h_in = h  # for residual connection
        # node update
        h = self.node_func(h)
        # edge update
        e = self.edge_func_1(e)
        e = shifted_softplus(e)
        e = self.edge_func_2(e)
        e = shifted_softplus(e)
        # message passing
        g.ndata["h"] = h
        g.edata["e"] = e
        g.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
        h = g.ndata["h"]
        # projection
        if self.batch_norm:
            h = self.bn(h)

        h = self.fc_1(h)
        h = shifted_softplus(h)
        h = self.fc_2(h)

        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h


def shifted_softplus(x: torch.Tensor, shift: float = 2.0) -> torch.Tensor:
    return F.softplus(x) - torch.log(torch.tensor(shift, device=x.device))
