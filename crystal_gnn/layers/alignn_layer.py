from typing import Tuple

import dgl
import torch
import torch.nn as nn

from crystal_gnn.layers.gated_gcn_layer import GatedGCNLayer


class ALIGNNLayer(nn.Module):
    """ALIGNN layer.

    "Atomistic Line Graph Neural Network
    for improved materials property predictions"
    npj Comput. Mater. (2021).
    https://www.nature.com/articles/s41524-021-00650-1
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim (int): dimension of input features
            output_dim (int): dimension of output features
            batch_norm (bool, optional): whether to use batch normalization. Defaults to False.
            residual (bool, optional): whether to use residual connection. Defaults to False.
            dropout (float, optional): a ratio of dropout. Defaults to 0.0.
        """
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
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        h: torch.Tensor,
        e: torch.Tensor,
        l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            g (dgl.DGLGraph): DGLGraph
            lg (dgl.DGLGraph): line graph of g
            h (torch.Tensor): embedded node features
            e (torch.Tensor): embedded edge features
            l (torch.Tensor): embedded line graph features (edge pair)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            updated node features, updated edge features, updated line graph features
        """
        # node update
        h, m = self.node_update(g, h, e)
        # edge update
        e, l = self.edge_update(lg, m, l)

        return h, e, l
