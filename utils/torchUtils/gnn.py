import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sklearn.metrics as metrics

from ..selectUtils import *
from ..utils import *


def to_tensor(tensor, gpu=False):
    if not torch.is_tensor(tensor):
        tensor = torch.Tensor(tensor)
    if gpu:
        return tensor.to('cuda:0')
    return tensor


def get_uptri(edge_index, edge_attr, return_index=False):
    uptri = edge_index[0] < edge_index[1]
    edge_index = torch.stack([edge_index[0][uptri], edge_index[1][uptri]])
    edge_attr = edge_attr[uptri]
    if return_index:
        return edge_attr, edge_index
    return edge_attr


def train_test_split(dataset, test_split):
    size = len(dataset)
    train_size = int(size*(1-test_split))
    test_size = size - train_size
    return random_split(dataset, [train_size, test_size])


def graph_pred(model, g):
    edge_pred = model.predict(g)
    if type(edge_pred) is tuple:
        _, edge_pred = edge_pred

    g_pred = Data(x=g.x, edge_index=g.edge_index,
                  edge_attr=g.edge_attr, y=g.y, edge_y=edge_pred)
    return g_pred


def get_wp(wp_fpr, fpr, tpr, thresholds):
    wp_index = np.where(fpr > wp_fpr)[0][0]
    wp_tpr = tpr[wp_index]
    wp_threshold = thresholds[wp_index]
    return np.array([wp_fpr, wp_tpr, wp_threshold])


def predict_dataset_edges(model, dataset, batch_size=50):
    if type(dataset) is not DataLoader:
        dataset = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    edge_scores = torch.cat([model.predict_edges(data) for data in dataset])
    return edge_scores.numpy()


def predict_dataset_nodes(model, dataset, batch_size=50):
    if type(dataset) is not DataLoader:
        dataset = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    edge_scores = torch.cat([model.predict_nodes(data) for data in dataset])
    return edge_scores.numpy()


class ROCMetric:
    def __init__(self, true, pred):
        self.true = ak.flatten(true,axis=None).to_numpy()
        self.pred = ak.flatten(pred,axis=None).to_numpy()
        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
            self.true, self.pred)
        self.auc = metrics.auc(self.fpr, self.tpr)

    def get_wps(self, fpr_wps=[0.2, 0.1, 0.05]):
        self.wps = np.stack(
            [get_wp(fpr_wp, self.fpr, self.tpr, self.thresholds) for fpr_wp in fpr_wps])
        return self.wps

    def get_values(self): return self.fpr, self.tpr, self.auc


def get_model_roc(model, dataloader, batch_size=50):
    if type(dataloader) is not DataLoader:
        dataloader = DataLoader(dataloader, batch_size=batch_size, num_workers=4)

    node_true = torch.cat([data.y for data in dataloader]).numpy()
    node_pred = predict_dataset_nodes(model, dataloader)

    edge_true = torch.cat([data.edge_y for data in dataloader]).numpy()
    edge_pred = predict_dataset_edges(model, dataloader)

    node_metrics = ROCMetric(node_true, node_pred)
    edge_metrics = ROCMetric(edge_true, edge_pred)

    return node_metrics, edge_metrics
