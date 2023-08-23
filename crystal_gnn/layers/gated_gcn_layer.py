"""implemented in Benchmarking-GNNs 
(https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/layers/gated_gcn_layer.py)"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedGCNLayer(nn.Module):
    """ResGatedGCN: Residual Gated Graph ConvNets

    "An Experimental Study of Neural Networks for Variable Graph"
    ICLR (2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim)
        self.B = nn.Linear(input_dim, output_dim)
        self.C = nn.Linear(input_dim, output_dim)
        self.D = nn.Linear(input_dim, output_dim)
        self.E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
    ):
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embedded edge features
        Return:
            h (torch.Tensor): updated node features
            e (torch.Tensor): updated edge features
        """
        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)

        g.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))
        g.edata["e"] = g.edata["DEh"] + g.edata["Ce"]  # updated edge features (e^_ij)
        g.edata["sigma"] = torch.sigmoid(g.edata["e"])  # sigma(e^_ij)
        # numerator
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        # denominator
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["Ah"] + g.ndata["sum_sigma_h"] / (
            g.ndata["sum_sigma"] + 1e-6
        )  # updated node features
        h = g.ndata["h"]
        e = g.edata["e"]

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        h = F.silu(h)
        e = F.silu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e
