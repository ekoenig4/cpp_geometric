import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
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


class ROCMetric:
    def __init__(self):
        self.true = torch.Tensor()
        self.pred = torch.Tensor()

    def update(self, true, pred):
        self.true = torch.cat([self.true, true])
        self.pred = torch.cat([self.pred, pred])

    def process(self):
        self.true = self.true.numpy()
        self.pred = self.pred.numpy()

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
            self.true, self.pred)
        self.auc = metrics.auc(self.fpr, self.tpr)

    def get_wps(self, fpr_wps=[0.1, 0.01, 0.001]):
        self.wps = np.stack(
            [get_wp(fpr_wp, self.fpr, self.tpr, self.thresholds) for fpr_wp in fpr_wps])
        return self.wps

    def get_values(self): return self.fpr, self.tpr, self.auc


def get_model_roc(model, dataloader):
    node_metrics = ROCMetric()
    edge_metrics = ROCMetric()

    for data in dataloader:
        node_true, edge_true = data.y, data.edge_y
        node_pred, edge_pred = model.predict(data)

        edge_true = get_uptri(data.edge_index, edge_true)
        # edge_pred = get_uptri(data.edge_index, edge_pred)

        node_metrics.update(node_true, node_pred)
        edge_metrics.update(edge_true, edge_pred)
    node_metrics.process()
    edge_metrics.process()

    return node_metrics, edge_metrics
