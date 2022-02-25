import os
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
        layers = self.cfg["model"]["layers"].split(',')
        shapes = self.cfg["model"]["layer_shapes"].split(',')
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

def export_array(array):
    return ",".join(map(str,np.array(array).flatten()))

def export_layer(layer):
    state = {param.replace("linear.", ""): array.numpy()
             for param, array in layer.state_dict().items()}

    if any(state):
        return {
            "class": type(layer).__name__,
            "shape": state["weight"].shape,
            "weight": state["weight"],
            "bias": state["bias"]
        }
    else:
        return  {
            "class": type(layer).__name__,
            "shape": np.array([0,0]),
            "weight": np.array([]),
            "bias": np.array([])
        }

def export_model(model,template,outdir):
    print(f"Exporting model to: {outdir}")
    
    layers = [
        export_layer(layer)
        for _, layer in model.named_children()
    ]
    
    cfg = {
        "model": {
            "num_node_features": template.num_node_features,
            "node_feautres": export_array(template.node_attr_names),
            "num_edge_features": template.num_edge_features,
            "edge_features": export_array(template.edge_attr_names),
            "num_layers": len(layers),
            "layers": export_array([layer["class"] for layer in layers]),
            "layer_shapes": export_array([layer["shape"] for layer in layers])
        },
        "scaler": {
            "node_scale_min": export_array(template.node_scaler.minims),
            "node_scale_max": export_array(template.node_scaler.maxims),
            "edge_scale_min": export_array(template.edge_scaler.minims),
            "edge_scale_max": export_array(template.edge_scaler.maxims)
        }
    }
    keyvalue = lambda kv : f"{kv[0]} = {kv[1]}"
    export_params = lambda d : "\n".join( map(keyvalue,d.items()) )

    yaml = "\n\n".join( f"[{key}]\n{export_params(params)}" for key,params in cfg.items() )
    print(yaml)
    
    weight_csv = np.concatenate([ layer["weight"].flatten() for layer in layers ])
    bias_csv = np.concatenate([ layer["bias"].flatten() for layer in layers ])
    
    os.makedirs(outdir)
    with open(f"{outdir}/model.cfg","w") as f: f.write(yaml) 
    np.savetxt(f"{outdir}/weights.txt", weight_csv)
    np.savetxt(f"{outdir}/bias.txt",bias_csv)
