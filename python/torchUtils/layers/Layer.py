from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.typing import Adj, PairTensor

class NodeLinear(Linear):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return super(NodeLinear,self).forward(x)

class EdgeLinear(Linear):
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return super(EdgeLinear,self).forward(edge_attr)