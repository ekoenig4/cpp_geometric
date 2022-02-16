from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import Adj, PairTensor


class EdgeConv(MessagePassing):
    aggr_funcs = dict(
        max=lambda tensor: tensor.max(dim=-1)[0],
        min=lambda tensor: tensor.min(dim=-1)[0],
        mean=lambda tensor: tensor.mean(dim=-1),
    )

    def __init__(self, nn: Callable, aggr: str = 'max', edge_aggr: str = 'max', return_with_edges: bool = False, return_only_edges: bool = False, **kwargs):
        super().__init__(aggr, **kwargs)
        self.nn = nn 
        self.edge_x: Tensor = Tensor

        assert edge_aggr in ['max', 'mean', 'min', None]
        self.edge_aggr = edge_aggr

        self.return_with_edges = return_with_edges
        self.return_only_edges = return_only_edges

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_x: Optional[Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        x = self.propagate(edge_index, x=x, edge_x=edge_x)
        if self.return_with_edges or self.return_only_edges:
            if self.edge_aggr is None:
                edge_x = torch.cat(
                    [x[edge_index[1]], x[edge_index[0]], self.edge_x], dim=-1)
            else:
                edge_x = torch.cat(
                    [x[edge_index[1]][:, :, None], x[edge_index[0]][:, :, None], self.edge_x[:, :, None]], dim=-1)
                edge_x = self.aggr_funcs[self.edge_aggr](edge_x)

            if self.return_with_edges:
                return x, edge_x
            if self.return_only_edges:
                return edge_x

        return x

    def message(self, x_i: Tensor, x_j: Tensor, edge_x: Tensor) -> Tensor:
        self.edge_x = self.nn(torch.cat([x_i, x_j - x_i, edge_x], dim=-1))
        return self.edge_x


class GCNConv(MessagePassing):
    def __init__(self,n_in_node,n_in_edge,n_out):
        super(GCNConv,self).__init__(aggr='add')
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge,n_out);
    def forward(self, x, edge_index, edge_attr):
        # self.messages = []
        return self.propagate(edge_index,x=x,edge_attr=edge_attr)
    
    def message(self,x_i, x_j, edge_attr):
        out = torch.cat([x_i,x_j-x_i,edge_attr],dim=1)
        out = self.linear(out)
        # self.messages.append((x_i,x_j,edge_attr,out))
        return out
    
    def update(self,aggr_out):
        return aggr_out
    
class EdgeConCat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,edge_index,edge_attr):
        src,dest = edge_index
        out = torch.cat([x[src],x[dest],edge_attr],dim=1)
        return out

class EdgeOnlyConv(torch.nn.Module):
    def __init__(self,n_in_node,n_in_edge,n_out):
        super().__init__()
        self.linear = torch.nn.Linear(2*n_in_node+n_in_edge,n_out)
    def forward(self,x,edge_index,edge_attr):
        src,dest = edge_index
        out = torch.cat([x[src],x[dest],edge_attr],dim=1)
        return self.linear(out)