
import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from ..selectUtils import *
from ..utils import get_collection
from .torchscript import build_dataset
from .gnn import *


class ScaleAttrs:
    def fit(self, x):
        self.nfeatures = x[0].to_numpy().shape[-1]
        self.minims = np.array([ak.min(x[:, :, i])
                               for i in range(self.nfeatures)])
        self.maxims = np.array([ak.max(x[:, :, i])
                               for i in range(self.nfeatures)])
        return self

    def transform(self, x):
        n_nodes = ak.num(x, axis=1)
        x = ak.flatten(x, axis=1).to_numpy()
        x = (x - self.minims)/(self.maxims - self.minims)
        return ak.unflatten(x, n_nodes)

    def inverse(self, x):
        n_nodes = ak.num(x, axis=1)
        x = ak.flatten(x, axis=1).to_numpy()
        x = (self.maxims-self.minims)*x + self.minims
        return ak.unflatten(x, n_nodes)


def graph_to_torch(graph):
    return Data(x=graph.x, y=graph.y, edge_index=graph.edge_index, edge_attr=graph.edge_attr, edge_y=graph.edge_y)


def prepare_features(attrs, targs):
    slices = torch.from_numpy(ak.num(attrs, axis=1).to_numpy())
    slices = torch.cat([torch.Tensor([0]), slices.cumsum(dim=0)]).long()
    attrs = torch.from_numpy(ak.flatten(attrs, axis=1).to_numpy()).float()
    targs = torch.from_numpy(ak.flatten(targs, axis=1).to_numpy()).long()
    return attrs, targs, slices


def get_node_attrs(jets, attrs=["m", "pt", "eta", "phi", "btag"]):
    features = ak.concatenate([attr[:, :, None]
                              for attr in ak.unzip(jets[attrs])], axis=-1)
    return features


def get_node_targs(jets):
    targets = 1*(jets.signalId > -1)
    return targets


def get_edge_attrs(jets, attrs=["dr"]):
    dr = calc_dr(jets.eta[:, :, None], jets.phi[:, :, None],
                 jets.eta[:, None], jets.phi[:, None])[:, :, :, None]
    return ak.flatten(dr, axis=2)


def get_edge_targs(jets):
    diff = np.abs(jets.signalId[:, None] - jets.signalId[:, :, None])
    add = jets.signalId[:, None] + jets.signalId[:, :, None]
    mod2 = add % 2

    paired = (diff*mod2 == 1) & ((add == 1) | (add == 5) | (add == 9))
    return ak.flatten(1*paired, axis=2)


def build_node_features(jets, node_attr_names=["m", "pt", "eta", "phi", "btag"]):
    node_attrs = get_node_attrs(jets, node_attr_names)
    node_targs = get_node_targs(jets)
    return node_attrs, node_targs, node_attr_names


def build_edge_features(jets, edge_attr_names=["dr"]):
    edge_attrs = get_edge_attrs(jets, edge_attr_names)
    edge_targs = get_edge_targs(jets)
    return edge_attrs, edge_targs, edge_attr_names


def build_features(tree, node_attr_names=["m", "pt", "eta", "phi", "btag"], edge_attr_names=["dr"]):
    jets = get_collection(tree, 'jet', False)
    return build_node_features(jets, node_attr_names), build_edge_features(jets, edge_attr_names)


def scale_attrs(attrs, scaler=None):
    if scaler is None:
        scaler = ScaleAttrs().fit(attrs)
    attrs = scaler.transform(attrs)
    return attrs, scaler


def get_class_weights(node_targs, edge_targs):
    pos_node_targs = np.sum(node_targs == 1)
    neg_node_targs = np.sum(node_targs == 0)
    num_nodes = pos_node_targs + neg_node_targs
    node_class_weights = max(neg_node_targs, pos_node_targs) / \
        np.array([neg_node_targs, pos_node_targs])

    pos_edge_targs = np.sum(edge_targs == 1)/2
    neg_edge_targs = (np.sum(edge_targs == 0)-num_nodes)/2
    num_edges = pos_edge_targs + neg_edge_targs
    edge_class_weights = max(neg_edge_targs, pos_edge_targs) / \
        np.array([neg_edge_targs, pos_edge_targs])

    type_class_weights = max(num_nodes, num_edges) / \
        np.array([num_nodes, num_edges])

    return node_class_weights, edge_class_weights, type_class_weights

class Transform:
    def __init__(self,*args):
        self.transforms = args
    def __call__(self,graph):
        for transform in self.transforms: 
            graph = transform(graph)
        return graph

def to_uptri_graph(graph):
    edge_index, edge_attr, edge_y = graph.edge_index, graph.edge_attr, graph.edge_y
    uptri = edge_index[0] < edge_index[1]
    edge_index = torch.stack([edge_index[0][uptri], edge_index[1][uptri]])
    edge_attr = edge_attr[uptri]
    edge_y = edge_y[uptri]
    return Data(x=graph.x, y=graph.y, edge_index=edge_index, edge_attr=edge_attr, edge_y=edge_y)

def to_numpy(graph):
    return Data(x=graph.x.numpy(),y=graph.y.numpy(),edge_index=graph.edge_index.numpy(),edge_attr=graph.edge_attr.numpy(),edge_y=graph.edge_y.numpy())

def to_long(graph,precision=1e6):
    return Data(x=(precision*graph.x).long(),y=graph.y.long(),edge_index=graph.edge_index.long(),edge_attr=(precision*graph.edge_attr).long(),edge_y=graph.edge_y.long())

class Dataset(InMemoryDataset):
    def __init__(self, root, tree=None, template=None, transform=None):
        self.tree = tree

        self.node_scaler = template.node_scaler if template is not None else None
        self.edge_scaler = template.edge_scaler if template is not None else None
        self.node_class_weights = template.node_class_weights if template is not None else None
        self.edge_class_weights = template.edge_class_weights if template is not None else None
        self.type_class_weights = template.type_class_weights if template is not None else None
        self.node_attr_names = template.node_attr_names if template is not None else None
        self.edge_attr_names = template.edge_attr_names if template is not None else None

        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.node_attr_names, self.edge_attr_names = torch.load(
            self.processed_paths[1])
        self.node_class_weights, self.edge_class_weights, self.type_class_weights = torch.load(
            self.processed_paths[2])
        self.filelist = torch.load(self.processed_paths[3])
        self.node_scaler, self.edge_scaler = torch.load(
            self.processed_paths[4])

    @property
    def processed_file_names(self):
        return ['graphs.pt', 'attr_names.pt', 'class_weights.pt', 'filelist.pt', 'scalers.pt']

    def process(self):
        # Read data into huge `Data` list.
        filelist = list(map(lambda f: f.fname, self.tree.filelist))

        print("Building Features...")
        jets = get_collection(self.tree, 'jet', False)

        node_kwargs = {}
        if self.node_attr_names is not None:
            node_kwargs['node_attr_names'] = self.node_attr_names
        edge_kwargs = {}
        if self.edge_attr_names is not None:
            edge_kwargs['edge_attr_names'] = self.edge_attr_names

        node_attrs, node_targs, node_attr_names = build_node_features(
            jets, **node_kwargs)
        edge_attrs, edge_targs, edge_attr_names = build_edge_features(
            jets, **edge_kwargs)
        node_attrs, node_scaler = scale_attrs(node_attrs, self.node_scaler)
        edge_attrs, edge_scaler = scale_attrs(edge_attrs, self.edge_scaler)

        if self.node_class_weights is None or self.edge_class_weights is None or self.type_class_weights is None:
            node_class_weights, edge_class_weights, type_class_weights = get_class_weights(
                node_targs, edge_targs)
        else:
            node_class_weights, edge_class_weights, type_class_weights = self.node_class_weights, self.edge_class_weights, self.type_class_weights

        node_attrs, node_targs, node_slices = prepare_features(
            node_attrs, node_targs)
        edge_attrs, edge_targs, edge_slices = prepare_features(
            edge_attrs, edge_targs)

        assert node_attrs.type(
        ) == 'torch.FloatTensor', f"Expected node_attrs of type torch.FloatTensor, but got {node_attrs.type()}"
        assert node_targs.type(
        ) == 'torch.LongTensor', f"Expected node_targs of type torch.LongTensor, but got {node_targs.type()}"
        assert node_slices.type(
        ) == 'torch.LongTensor', f"Expected node_slices of type torch.LongTensor, but got {node_slices.type()}"

        assert edge_attrs.type(
        ) == 'torch.FloatTensor', f"Expected edge_attrs of type torch.FloatTensor, but got {edge_attrs.type()}"
        assert edge_targs.type(
        ) == 'torch.LongTensor', f"Expected edge_targs of type torch.LongTensor, but got {edge_targs.type()}"
        assert edge_slices.type(
        ) == 'torch.LongTensor', f"Expected edge_slices of type torch.LongTensor, but got {edge_slices.type()}"

        print("Building Dataset...")
        data_list = build_dataset(
            node_attrs, node_targs, node_slices, edge_attrs, edge_targs, edge_slices)

        data_list = [graph_to_torch(graph) for graph in data_list]

        data, slices = self.collate(data_list)

        print("Saving Dataset...")
        torch.save((data, slices), self.processed_paths[0])
        torch.save((node_attr_names, edge_attr_names), self.processed_paths[1])
        torch.save((node_class_weights, edge_class_weights, type_class_weights),
                   self.processed_paths[2])
        torch.save(filelist, self.processed_paths[3])
        torch.save((node_scaler, edge_scaler), self.processed_paths[4])

    def build_graphs(self, tree):
        (node_attrs, node_targs, node_attr_names), (edge_attrs,
                                                    edge_targs, edge_attr_names) = build_features(tree, self.node_attr_names, self.edge_attr_names)
        node_attrs = self.node_scaler.transform(node_attrs)
        edge_attrs = self.edge_scaler.transform(edge_attrs)

        node_attrs, node_targs, node_slices = prepare_features(
            node_attrs, node_targs)
        edge_attrs, edge_targs, edge_slices = prepare_features(
            edge_attrs, edge_targs)
        data_list = build_dataset(
            node_attrs, node_targs, node_slices, edge_attrs, edge_targs, edge_slices)

        if self.transform is not None:
            return [self.transform(graph_to_torch(graph)) for graph in data_list]
        return [graph_to_torch(graph) for graph in data_list]
