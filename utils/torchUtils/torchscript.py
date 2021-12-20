import torch
from typing import List


@torch.jit.script
class GraphDataBase:
    def __init__(self, x, y, edge_index, edge_attr, edge_y):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_y = edge_y


@torch.jit.script
def get_fully_connected(n_nodes: int) -> torch.Tensor:
    nodes = torch.arange(0, n_nodes)
    return torch.stack(torch.meshgrid([nodes, nodes]), dim=0).reshape(2, n_nodes*n_nodes)


@torch.jit.script
def build_graph(node_attrs: torch.Tensor, node_targs: torch.Tensor, edge_attrs: torch.Tensor, edge_targs: torch.Tensor) -> GraphDataBase:
    n_nodes = node_attrs.shape[0]
    edge_index = get_fully_connected(n_nodes)

    assert node_attrs.shape[0] == node_targs.shape[0], f"Expected node_attrs and node_targs shapes to match, but got {node_attrs.shape[0],node_targs.shape[0]}"
    assert edge_attrs.shape[0] == edge_targs.shape[0], f"Expected edge_attrs and edge_targs shapes to match, but got {edge_attrs.shape[0],edge_targs.shape[0]}"
    assert edge_attrs.shape[0] == edge_index.shape[1], f"Expected edge_attrs and edge_index shapes to match, but got {edge_attrs.shape[0],edge_index.shape[1]}"

    return GraphDataBase(node_attrs, node_targs, edge_index, edge_attrs, edge_targs)


@torch.jit.script
def build_dataset(node_attrs: torch.Tensor, node_targs: torch.Tensor, node_slices: torch.Tensor, edge_attrs: torch.Tensor, edge_targs: torch.Tensor, edge_slices: torch.Tensor) -> List[GraphDataBase]:
    data_list: List[GraphDataBase] = []
    for node_lo, node_hi, edge_lo, edge_hi in zip(node_slices[:-1], node_slices[1:], edge_slices[:-1], edge_slices[1:]):
        data_list.append(
            build_graph(
                node_attrs[node_lo:node_hi],
                node_targs[node_lo:node_hi],
                edge_attrs[edge_lo:edge_hi],
                edge_targs[edge_lo:edge_hi]
            )
        )
    return data_list
