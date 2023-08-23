from typing import Tuple, Dict
import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MEGNETLayer(nn.Module):
    """MEGNET model.

    "Graph Networks as a Universal Machine Learning Framework
    for Molecules and Crystals"
    Chem. Mater. (2019)
    https://pubs.acs.org/doi/10.1021/acs.chemmater.9b01294
    """

    def __init__(
        self,
        hidden_dim: int,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim (int): dimension of hidden features
            batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
            residual (bool, optional): whether to use residual connection. Defaults to False.
            dropout (float, optional): a ratio of dropout. Defaults to 0.0.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        self.node_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )
        self.edge_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )
        self.state_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )

        self.node_update_func = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )
        self.edge_update_func = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )

        self.state_update_func = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            SoftPlus2(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            SoftPlus2(),
        )

        self.bn_node_h = nn.BatchNorm1d(hidden_dim)
        self.bn_node_e = nn.BatchNorm1d(hidden_dim)
        self.bn_node_s = nn.BatchNorm1d(hidden_dim)

    def edge_update(self, g: dgl.DGLGraph) -> torch.Tensor:
        def update_func(edges: dgl.udf.EdgeBatch) -> Dict[str, torch.Tensor]:
            _m = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["e"], edges.src["n_s"]],
                dim=1,
            )
            m = self.edge_update_func(_m)
            return {"m": m}

        g.apply_edges(update_func)
        return g.edata.pop("m")

    def node_update(self, g: dgl.DGLGraph) -> torch.Tensor:
        g.update_all(fn.copy_e("e", "m"), fn.mean("m", "hh"))
        _h = torch.cat([g.ndata["hh"], g.ndata["h"], g.ndata["n_s"]], dim=1)
        return self.node_update_func(_h)

    def state_update(self, g: dgl.DGLGraph, s) -> torch.Tensor:
        u_edge = dgl.readout_edges(g, "e", op="mean")
        u_node = dgl.readout_nodes(g, "h", op="mean")
        _s = torch.cat([s, u_edge, u_node], dim=1)
        return self.state_update_func(_s)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embedded edge features
            s (torch.Tensor): embedded global state features

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            updated node features, updated edge features, updated global state features
        """
        # residual connection
        h_in = h
        e_in = e
        s_in = s

        h = self.node_func(h)
        e = self.edge_func(e)
        s = self.state_func(s)

        # update with MEGNet convolution
        g.ndata["h"] = h
        g.edata["e"] = e
        g.ndata["n_s"] = dgl.broadcast_nodes(g, s)

        e = self.edge_update(g)
        g.edata["e"] = e
        h = self.node_update(g)
        g.ndata["h"] = h
        s = self.state_update(g, s)

        # if self.batch_norm:
        #     h = self.bn_node_h(h)
        #     e = self.bn_node_e(e)
        #     s = self.bn_node_s(s)

        # h = F.selu(h)
        # e = F.selu(e)
        # s = F.selu(s)

        if self.residual:
            h = h_in + h
            e = e_in + e
            s = s_in + s

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        s = F.dropout(s, self.dropout, training=self.training)

        return h, e, s


class SoftPlus2(nn.Module):
    """Implemented in MEGNet paper
    SoftPlus2 activation function:
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x) - math.log(2.0)
