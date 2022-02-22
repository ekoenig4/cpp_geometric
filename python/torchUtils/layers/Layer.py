from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch_geometric.typing import Adj, PairTensor

class NodeLinear(Linear):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return super(NodeLinear,self).forward(x),edge_attr

class EdgeLinear(Linear):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return x,super(EdgeLinear,self).forward(edge_attr)
    
class GCNRelu(Module):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return torch.relu(x), torch.relu(edge_attr)
    
class GCNSoftmax(Module):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return torch.softmax(x), torch.softmax(edge_attr)
    