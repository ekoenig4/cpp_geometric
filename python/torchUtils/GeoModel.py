import torch 
import numpy as np 
import awkward as ak 

from configparser import ConfigParser

from . import layers as model_layers


def _get_weights(root, shapes):
    f_weights = np.loadtxt(f"{root}/weights.txt")
    weight_index = [0] + list(shapes.prod(axis=1).cumsum())
    weights = [f_weights[lo:hi]
               for lo, hi in zip(weight_index[:-1], weight_index[1:])]
    weights = [weight.reshape(shape) for weight, shape in zip(weights, shapes)]
    return weights


def _get_biases(root, shapes):
    f_biases = np.loadtxt(f"{root}/bias.txt")
    biases_index = [0] + list(shapes[:, 0].cumsum())
    biases = [f_biases[lo:hi]
              for lo, hi in zip(biases_index[:-1], biases_index[1:])]
    return biases


class GeoModel:
    def __init__(self, root):
        self.root = root
        self.cfg = ConfigParser()

        with open(f"{self.root}/model.cfg", "r") as f:
            self.cfg.read_file(f)
        layers = self.cfg["model"]["layers"].split(' ')
        shapes = self.cfg["model"]["layer_shapes"].split(' ')
        shapes = np.array(list(zip(shapes[::2], shapes[1::2])), dtype=int)

        weights = _get_weights(self.root, shapes)
        biases = _get_biases(self.root, shapes)

        # self.debug = (layers, shapes, weights, biases)
        self.sequence = [
            getattr(model_layers, layer)(*shape[::-1])
            if shape.sum() > 0
            else
            getattr(model_layers, layer)()

            for shape, layer in zip(shapes, layers)
        ]

        for layer,weight,bias in zip(self.sequence,weights,biases):
            if weight.shape == (0,0): continue
            layer.weight.data = torch.Tensor(weight)
            layer.bias.data = torch.Tensor(bias)

    def __call__(self, x, edge_index, edge_attr):
        for layer in self.sequence:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr
