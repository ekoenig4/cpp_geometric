from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import Adj, PairTensor


class GCNConv(MessagePassing):
    def __init__(self, n_in_node, n_in_edge, n_out):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge, n_out)

    def forward(self, x, edge_index, edge_attr):
        self.messages = []
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        out = torch.cat([x_i, x_j-x_i, edge_attr], dim=1)
        out = self.linear(out)
        self.messages.append((x_i,x_j,edge_attr,out))
        return out

    def update(self, aggr_out):
        return aggr_out
