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


class EdgeConvONNX(EdgeConv):
    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        def aggr_input(inputs):
            return inputs.max(dim=0)[0]
        
        size,nfeatures = inputs.size()
        outputs = inputs.new_zeros((dim_size,nfeatures))
        
        padding = inputs.new_zeros((1,nfeatures))
        for i in range(dim_size):   
            outputs[i] = aggr_input(
                torch.cat([inputs[index==i],padding],dim=0)
                )
        self.debug = (inputs,index,outputs)
        return outputs