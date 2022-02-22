from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import Adj, PairTensor


class GCNConv(MessagePassing):
    def __init__(self, n_in = None, n_out=None, n_in_node=None, n_in_edge=None, ):
        super(GCNConv, self).__init__(aggr='add')
        if n_in is None: n_in = 2*n_in_node + n_in_edge
        self.linear = Linear(n_in, n_out)
        self.weight = self.linear.weight 
        self.bias = self.linear.bias

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        self.msg = torch.cat([x_i, x_j-x_i, edge_attr], dim=1)
        self.msg = self.linear(self.msg)
        return self.msg

    def update(self, aggr_out):
        return aggr_out
    
class GCNConvMSG(GCNConv):
    def __init__(self,n_in = None, n_out=None, n_in_node=None, n_in_edge=None, ):
        super(GCNConvMSG, self).__init__(n_in, n_out, n_in_node, n_in_edge)
        self.edge_aggr = lambda tensor: tensor.max(dim=-1)[0]

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        x = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        
        edge_attr = torch.cat([x[edge_index[1]], x[edge_index[0]], self.msg], dim=-1)
        
        # edge_attr = torch.cat([x[edge_index[1]][:, :, None], x[edge_index[0]][:, :, None], self.msg[:, :, None]], dim=-1)
        # edge_attr = self.edge_aggr(edge_attr)
        return x,edge_attr