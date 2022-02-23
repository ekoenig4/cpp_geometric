import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
 
def _load_nodes(root):
    node_shape = np.loadtxt(f'{root}/node_shape.txt',dtype=float).astype(int)
    node_x = np.loadtxt(f'{root}/node_x.txt',dtype=float)
    node_x = ak.unflatten(node_x,node_shape[:,0])
    return node_x

def _load_edges(root):
    edge_shape = np.loadtxt(f'{root}/edge_shape.txt',dtype=float).astype(int)
    edge_index = np.loadtxt(f'{root}/edge_index.txt',dtype=float).astype(int)
    edge_attr = np.loadtxt(f'{root}/edge_attr.txt',dtype=float)

    edge_attr = ak.unflatten(edge_attr,edge_shape[:,0])
    edge_index = ak.unflatten(edge_index,edge_shape[:,0])
    edge_index = ak.Array( [index.to_numpy().T for index in edge_index] )
    return edge_index,edge_attr


def _load_extra_nodes(root,tag):
    node_shape = np.loadtxt(f'{root}/{tag}_node_shape.txt',dtype=float).astype(int)
    node_x = np.loadtxt(f'{root}/{tag}_node_x.txt',dtype=float)
    node_x = ak.unflatten(node_x,node_shape[:,0])
    return node_x

def _load_extra_edges(root,tag):
    edge_shape = np.loadtxt(f'{root}/{tag}_edge_shape.txt',dtype=float).astype(int)
    edge_attr = np.loadtxt(f'{root}/{tag}_edge_attr.txt',dtype=float)
    edge_attr = ak.unflatten(edge_attr,edge_shape[:,0])
    return edge_attr

def _build_graph(node_x,edge_index,edge_attr):
    node_x = torch.Tensor(node_x)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.Tensor(edge_attr).reshape(-1,1)
    return Data(x=node_x,edge_index=edge_index,edge_attr=edge_attr)
    
 
class Dataset(list):
    def __init__(self,root):
        self.root = root
        expected = ['node_x.txt','node_shape.txt','edge_attr.txt','edge_shape.txt','edge_index.txt']
        
        
        node_x = _load_nodes(root)
        edge_index,edge_attr = _load_edges(root)
        
        for graph in zip(node_x,edge_index,edge_attr):
            self.append(_build_graph(*graph) )
            
    def load_extra(self,tag):
        node_x = _load_extra_nodes(self.root,tag)
        edge_attr = _load_extra_edges(self.root,tag)
        
        for graph,x,attr in zip(self,node_x,edge_attr):
            setattr(graph,f"{tag}_x",x)
            setattr(graph,f"{tag}_edge_attr",attr)
            
        