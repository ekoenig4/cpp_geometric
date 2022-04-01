import numpy as np
import os

def export_nodes(nodes,outdir=None,tag=None):
    if not os.path.isdir(outdir): os.makedirs(outdir)
    if tag is None: tag = ""
    else: tag = tag+"_"
    
    np.savetxt(f'{outdir}/{tag}node_shape.txt',np.array([ x.shape for x in nodes]))
    np.savetxt(f'{outdir}/{tag}node_x.txt',np.concatenate([x for x in nodes]))
    
def export_edges(edges,outdir=None,tag=None,index=None):
    if not os.path.isdir(outdir): os.makedirs(outdir)
    if tag is None: tag = ""
    else: tag = tag+"_"
    
    np.savetxt(f'{outdir}/{tag}edge_shape.txt',np.array([ edge_attr.shape for edge_attr in edges]))
    np.savetxt(f'{outdir}/{tag}edge_attr.txt',np.concatenate([edge_attr for edge_attr in edges]))
    if index is not None:
        np.savetxt(f'{outdir}/{tag}edge_index.txt',np.concatenate([edge_index.T for edge_index in index]))
        
def export_graphs(graphs,outdir=None,tag=None):
    export_nodes([graph.x for graph in graphs],outdir,tag)
    export_edges([graph.edge_attr for graph in graphs],outdir,tag,index=[graph.edge_index for graph in graphs])

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
    else: # use empty weights and biases for layers without them
        return  {
            "class": type(layer).__name__,
            "shape": np.array([0,0]),
            "weight": np.array([]),
            "bias": np.array([])
        }
        
def export_model(model,template,outdir=None):
    print(f"Exporting model to: {outdir}")
    
    layers = [
        export_layer(layer)
        for _, layer in model.named_children()
    ]
    
    cfg = {
        "model": {
            "num_layers": len(layers),
            "layers": export_array([layer["class"] for layer in layers]),
            "layer_shapes": export_array([layer["shape"] for layer in layers])
        },
        "features":{
            "num_node_features": len(template.node_attr_names),
            "node_features": export_array(template.node_attr_names),
            
            "num_node_mask":len(model.hparams["node_attr_names"]),
            "node_mask":export_array(model.hparams["node_attr_names"]),
            
            "num_edge_features": len(template.edge_attr_names),
            "edge_features": export_array(template.edge_attr_names),
            
            "num_edge_mask":len(model.hparams["edge_attr_names"]),
            "edge_mask":export_array(model.hparams["edge_attr_names"]),
            
            "scale":model.hparams['scale'],
        },
        "scaler": {
            "node_scale_min": export_array(template.node_scaler.minims),
            "node_scale_max": export_array(template.node_scaler.maxims),
            "node_scale_mean":export_array(template.node_scaler.means),
            "node_scale_std": export_array(template.node_scaler.stds),
            "edge_scale_min": export_array(template.edge_scaler.minims),
            "edge_scale_max": export_array(template.edge_scaler.maxims),
            "edge_scale_mean":export_array(template.edge_scaler.means),
            "edge_scale_std": export_array(template.edge_scaler.stds)
        }
    }
    keyvalue = lambda kv : f"{kv[0]} = {kv[1]}"
    export_params = lambda d : "\n".join( map(keyvalue,d.items()) )

    yaml = "\n\n".join( f"[{key}]\n{export_params(params)}" for key,params in cfg.items() )
    print(yaml)
    
    weight_csv = np.concatenate([ layer["weight"].flatten() for layer in layers ])
    bias_csv = np.concatenate([ layer["bias"].flatten() for layer in layers ])
    
    if not os.path.isdir(outdir): os.makedirs(outdir)
    with open(f"{outdir}/model.cfg","w") as f: f.write(yaml) 
    np.savetxt(f"{outdir}/weights.txt", weight_csv)
    np.savetxt(f"{outdir}/bias.txt",bias_csv)