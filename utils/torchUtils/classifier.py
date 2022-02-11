from torch_geometric.nn import Linear
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .gnn import to_tensor, get_uptri
from .layers import EdgeConv, EdgeOnlyConv, GCNConv

useGPU = True 
useGPU = useGPU and torch.cuda.is_available()


class GCN(pl.LightningModule):
    def __init__(self, dataset, for_onnx=False, nn1_out=32, nn2_out=128, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters('nn1_out', 'nn2_out')
        self.node_weights = to_tensor(dataset.node_class_weights, useGPU)
        self.edge_weights = to_tensor(dataset.edge_class_weights, useGPU)
        self.type_weights = to_tensor(dataset.type_class_weights, useGPU)

        nn1 = torch.nn.Sequential(
            Linear(2*dataset.num_node_features +
                   dataset.num_edge_features, nn1_out),
            torch.nn.ELU()
        )

        self.conv1 = EdgeConv(nn1, edge_aggr=None, return_with_edges=True)

        nn2 = torch.nn.Sequential(
            Linear(5*nn1_out, nn2_out),
            torch.nn.ELU()
        )

        self.conv2 = EdgeConv(nn2, edge_aggr=None, return_with_edges=True)

        self.edge_seq = torch.nn.Sequential(
            Linear(3*nn2_out, 2),
        )

        self.node_seq = torch.nn.Sequential(
            Linear(nn2_out, 2),
        )

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x, edge_attr = self.conv1(x, edge_index, edge_attr)
        x, edge_attr = self.conv2(x, edge_index, edge_attr)
        x, edge_attr = self.node_seq(x), self.edge_seq(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)

    def predict(self, data):    
        with torch.no_grad():
            node_pred, edge_pred = self(data)
        return torch.exp(node_pred)[:,1],torch.exp(edge_pred)[:,1]

    def predict_nodes(self, data):
        node_pred, edge_pred = self.predict(data)
        return node_pred

    def predict_edges(self, data):
        node_pred, edge_pred = self.predict(data)
        return edge_pred

    def shared_step(self, batch, batch_idx, tag=None):
        node_o, edge_o = self(batch)
        node_o, edge_o = to_tensor(
            node_o, useGPU), to_tensor(edge_o, useGPU)
        node_y, edge_y = to_tensor(
            batch.y, useGPU), to_tensor(batch.edge_y, useGPU)

        node_loss = F.nll_loss(node_o, node_y, self.node_weights)

        edge_o = get_uptri(batch.edge_index,edge_o)
        edge_y = get_uptri(batch.edge_index,edge_y)

        edge_loss = F.nll_loss(
            edge_o, edge_y, self.edge_weights)

        loss = self.type_weights[0]*node_loss + self.type_weights[1]*edge_loss
        acc = accuracy(torch.cat([node_o, edge_o]).argmax(
            dim=1), torch.cat([node_y, edge_y]))
        metrics = dict(loss=loss, acc=acc)
        if tag is not None:
            metrics = {f'{tag}_{key}': value for key, value in metrics.items()}

        self.log_dict(metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer],dict(
            scheduler=scheduler,
            monitor='val_loss'
        )
        

class GoldenGCN(pl.LightningModule):
    def __init__(self, dataset, nn1_out=32, nn2_out=128, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters('nn1_out', 'nn2_out')
        self.node_weights = to_tensor(dataset.node_class_weights, useGPU)
        self.edge_weights = to_tensor(dataset.edge_class_weights, useGPU)
        self.type_weights = to_tensor(dataset.type_class_weights, useGPU)
        
        self.node_conv1 = GCNConv(5, nn1_out)
        self.edge_conv1 = EdgeOnlyConv(5,1,nn1_out)
        self.node_conv2 = GCNConv(nn1_out,nn2_out)
        self.edge_conv2 = EdgeOnlyConv(nn1_out,nn1_out,nn2_out)
        
        self.node_linear = Linear(nn2_out,2)
        self.edge_linear = Linear(nn2_out,2)
        

    def forward(self, data):
        if type(data) is list:
            data = data[0]
        if type(data) is not tuple:
            data = (data.x, data.edge_index, data.edge_attr)
        x, edge_index, edge_attr = data

        x = self.node_conv1(x, edge_index, edge_attr)
        edge_attr = self.edge_conv1(x,edge_index,edge_attr)
        x,edge_attr = torch.relu(x),torch.relu(edge_attr)
        
        x = self.node_conv2(x, edge_index, edge_attr)
        edge_attr = self.edge_conv2(x,edge_index,edge_attr)
        x,edge_attr = torch.relu(x),torch.relu(edge_attr)
        
        x = self.node_linear(x)
        edge_attr = self.node_linear(edge_attr)
        return F.log_softmax(x, dim=1), F.log_softmax(torch.cat(edge_attr,dim=-1),dim=1)

    def predict(self, data):    
        with torch.no_grad():
            node_pred, edge_pred = self(data)
        return torch.exp(node_pred)[:,1],torch.exp(edge_pred)[:,1]

    def predict_nodes(self, data):
        node_pred, edge_pred = self.predict(data)
        return node_pred

    def predict_edges(self, data):
        node_pred, edge_pred = self.predict(data)
        return edge_pred

    def shared_step(self, batch, batch_idx, tag=None):
        node_o, edge_o = self(batch)
        node_o, edge_o = to_tensor(
            node_o, useGPU), to_tensor(edge_o, useGPU)
        node_y, edge_y = to_tensor(
            batch.y, useGPU), to_tensor(batch.edge_y, useGPU)

        node_loss = F.nll_loss(node_o, node_y, self.node_weights)

        edge_o = get_uptri(batch.edge_index,edge_o)
        edge_y = get_uptri(batch.edge_index,edge_y)

        edge_loss = F.nll_loss(
            edge_o, edge_y, self.edge_weights)

        loss = self.type_weights[0]*node_loss + self.type_weights[1]*edge_loss
        acc = accuracy(torch.cat([node_o, edge_o]).argmax(
            dim=1), torch.cat([node_y, edge_y]))
        metrics = dict(loss=loss, acc=acc)
        if tag is not None:
            metrics = {f'{tag}_{key}': value for key, value in metrics.items()}

        self.log_dict(metrics)
        return metrics

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, tag='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer],dict(
            scheduler=scheduler,
            monitor='val_loss'
        )

