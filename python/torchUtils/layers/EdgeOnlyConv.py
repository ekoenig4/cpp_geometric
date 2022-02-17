from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import Adj, PairTensor


class EdgeOnlyConv(torch.nn.Module):
    def __init__(self, n_in_node, n_in_edge, n_out):
        super().__init__()
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge, n_out)

    def forward(self, x, edge_index, edge_attr):
        src, dest = edge_index
        out = torch.cat([x[src], x[dest], edge_attr], dim=1)
        return self.linear(out)
