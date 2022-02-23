from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, Module, functional as F
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
    
class GCNLogSoftmax(Module):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)    