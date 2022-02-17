from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.typing import Adj, PairTensor


class EdgeConCat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr):
        src, dest = edge_index
        out = torch.cat([x[src], x[dest], edge_attr], dim=1)
        return out
