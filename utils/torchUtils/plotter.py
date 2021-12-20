import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

from ..selectUtils import *
from ..utils import *
from ..plotUtils import *
from .gnn import graph_pred

# Default Coloring


def default_coloring(g, node_attr=None):
    """Default coloring which colors nodes based on a node attr

    Args:
        g (torch_geometric.data.Data): Pytroch garph data
        node_attr (int, optional): Index of node attr to color nodes with. Defaults to None.

    Returns:
        dict: Dictionary attrs for coloring in networkx
    """
    node_color = [g.nodes[n]['x'][node_attr]
                  for n in g.nodes] if node_attr is not None else 'tab:purple'
    width = [g.get_edge_data(ni, nj)['edge_y'] for ni, nj in g.edges]
    return dict(node_color=node_color, width=np.array(width))


# Paired Coloring


def paired_coloring(g):
    """Paired coloring which colors the highest scoring node pairs

    Args:
        g (torch_geometric.data.Data): Pytorch graph data

    Returns:
        dict: Dictionary attrs for coloring in networkx
    """
    node_pairs = {n: -1 for n in g.nodes}
    node_color = {n: 'tab:purple' for n in g.nodes}
    pair_score = {n: 0 for n in g.nodes}
    edge_color = {e: 'black' for e, _ in enumerate(g.edges)}
    width = {e: 0 for e, _ in enumerate(g.edges)}

    scores = {(e, (ni, nj)): g.get_edge_data(ni, nj)[
        'edge_y'] for e, (ni, nj) in enumerate(g.edges)}
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for e, (ni, nj) in map(lambda kv: kv[0], sorted_scores):
        paired = g.get_edge_data(ni, nj)['edge_y']
        width[e] = paired

        if pair_score[ni] < paired and pair_score[nj] < paired:
            node_pairs[ni] = e
            node_pairs[nj] = e
            pair_score[ni] = paired
            pair_score[nj] = paired

    npairs = 0
    for e, (ni, nj) in map(lambda kv: kv[0], sorted_scores):
        if npairs == 3: break
        if node_pairs[ni] == -1 or node_pairs[nj] == -1:
            continue
        if node_pairs[ni] == node_pairs[nj]:
            npairs += 1
            node_color[ni] = 'tab:blue'
            node_color[nj] = 'tab:blue'
            edge_color[e] = 'tab:orange'

    node_color = list(node_color.values())
    edge_color = list(edge_color.values())
    width = np.array(list(width.values()))
    width = width/np.std(width)
    return dict(node_color=node_color, edge_color=edge_color, width=np.array(width))


def display_graph(g, pos='xy', sizing=1, coloring='paired', show_detector=False, figax=None):
    """Create a graph display

    Args:
        g (torch_geometric.data.Data): Pytorch graph data
        pos (str, optional): What 2D coorinates to use for positioning
        sizing (str, int, optional): What node attr to use as sizing, y uses target, int uses x at that index
        coloring (str, optional): type of coloring to use, paired or default. Defaults to 'paired'
        show_detector (bool, optional): If xye position, draws the barrel or endcap of the detector
    """
    if figax is None:
        figax = plt.subplots()
    fig, ax = figax
    plt.sca(ax)

    posmap = dict(
        e=lambda attr: attr[2],
        p=lambda attr: attr[3],
        x=lambda attr: (np.cos(2*np.pi*attr[3])+1)/2,
        y=lambda attr: (np.sin(2*np.pi*attr[3])+1)/2,
    )

    colorings = dict(paired=paired_coloring, default=default_coloring)

    g = nx.Graph(to_networkx(g, node_attrs=['x', 'y'], edge_attrs=[
                    'edge_attr', 'edge_y'], remove_self_loops=True))

    node_pos = np.array([[posmap[p](g.nodes[n]['x'])
                        for p in pos] for n in g.nodes])

    get_size = dict(
        y=lambda node: node['y'],
        x=lambda node, sizing=sizing: node['x'][sizing],
    )

    if type(sizing) is int:
        sizing = 'x'

    node_size = np.array([get_size[sizing](g.nodes[n]) for n in g])
    node_size = node_size/np.std(node_size)
    node_size = 1000*(node_size-np.min(node_size)) / \
        (np.max(node_size)-np.min(node_size))
    node_size = np.where(node_size > 100, node_size, 100)

    coloring = colorings.get(coloring, lambda g: {'node_color':'tab:purple'})(g)

    nx.draw(g, node_pos, node_size=node_size, **coloring, alpha=0.8)

    plt.gca().set(xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    plt.gca().set_aspect('equal')

    if show_detector and all(r in 'xye' for r in pos):
        if sorted(pos) == ['x', 'y']:
            plt.gca().add_patch(plt.Circle((0.5, 0.5), 0.5, fill=False))
        else:
            plt.plot([[0], [0]], 'k')
            plt.plot([[1], [1]], 'k')
            plt.plot([[0.5], [0.5]], 'k-.')

    return fig, ax


def display_pred(model, g, *args, **kwargs):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    display_graph(g, *args, **kwargs, figax=(fig, axs[0]))
    axs[0].set(title="True")
    g_pred = graph_pred(model, g)
    display_graph(g_pred, *args, **kwargs, figax=(fig, axs[1]))
    axs[1].set(title="Pred")


def plot_auroc(node_metrics, edge_metrics):
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))

    fpr, tpr, auc = node_metrics.get_values()

    graph_simple(fpr, tpr, xlabel="Node False Positive", ylabel="Node True Positive",
                 title=f"AUC: {auc:.3}", marker=None, figax=(fig, axs[0]))

    fpr, tpr, auc = edge_metrics.get_values()

    graph_simple(fpr, tpr, xlabel="Edge False Positive", ylabel="Edge True Positive",
                 title=f"AUC: {auc:.3}", marker=None, figax=(fig, axs[1]))

    return fig, axs
