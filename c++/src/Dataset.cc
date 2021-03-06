#include "Dataset.h"

#include "TorchUtils.h"

using namespace std;
using namespace Eigen;

TorchUtils::Dataset::Dataset(string root)
{
    this->root = root;

    vector<MatrixXf> node_x;
    vector<vector<vector<int>>> edge_index;
    vector<MatrixXf> edge_attr;

    load_nodes(node_x);
    load_edges(edge_index, edge_attr);

    for (unsigned int i = 0; i < node_x.size(); i++)
    {
        Graph g(node_x[i], edge_index[i], edge_attr[i]);
        this->push_back(g);
    }
}

void TorchUtils::Dataset::load_extra(string tag)
{
    vector<MatrixXf> node_x;
    load_extra_nodes(tag, node_x);

    vector<MatrixXf> edge_attr;
    load_extra_edges(tag, edge_attr);

    for (unsigned int i = 0; i < this->size(); i++)
    {
        this->at(i).add_extra(tag, node_x[i], edge_attr[i]);
    }
}

void TorchUtils::Dataset::load_nodes(vector<MatrixXf> &node_x)
{
    vector<vector<int>> node_shape;
    loadtxt(root + "/node_shape.txt", node_shape);

    vector<vector<float>> f_node_x;
    loadtxt(root + "/node_x.txt", f_node_x);

    int current_node = 0;
    for (vector<int> shape : node_shape)
    {
        int n_nodes = shape[0];
        vector<vector<float>> nodes;
        for (int first_node = current_node; current_node - first_node < n_nodes; current_node++)
        {
            nodes.push_back(f_node_x[current_node]);
        }
        node_x.push_back(to_eigen(nodes));
    }
}

void TorchUtils::Dataset::load_extra_nodes(string tag, vector<MatrixXf> &node_x)
{
    vector<vector<int>> node_shape;
    loadtxt(root + "/" + tag + "_node_shape.txt", node_shape);

    vector<vector<float>> f_node_x;
    loadtxt(root + "/" + tag + "_node_x.txt", f_node_x);

    int current_node = 0;
    for (vector<int> shape : node_shape)
    {
        int n_nodes = shape[0];
        vector<vector<float>> nodes;
        for (int first_node = current_node; current_node - first_node < n_nodes; current_node++)
        {
            nodes.push_back(f_node_x[current_node]);
        }
        node_x.push_back(to_eigen(nodes));
    }
}

void TorchUtils::Dataset::load_edges(vector<vector<vector<int>>> &edge_index, vector<MatrixXf> &edge_attr)
{
    vector<vector<int>> edge_shape;
    loadtxt(root + "/edge_shape.txt", edge_shape);

    vector<vector<int>> f_edge_index;
    loadtxt(root + "/edge_index.txt", f_edge_index);

    vector<vector<float>> f_edge_attr;
    loadtxt(root + "/edge_attr.txt", f_edge_attr);

    int current_edge = 0;
    for (vector<int> shape : edge_shape)
    {
        int n_edges = shape[0];
        vector<vector<float>> edges;
        vector<int> edge_i;
        vector<int> edge_j;
        for (int first_edge = current_edge; current_edge - first_edge < n_edges; current_edge++)
        {
            edges.push_back(f_edge_attr[current_edge]);
            edge_i.push_back(f_edge_index[current_edge][0]);
            edge_j.push_back(f_edge_index[current_edge][1]);
        }
        edge_index.push_back({edge_i, edge_j});
        edge_attr.push_back(to_eigen(edges));
    }
}

void TorchUtils::Dataset::load_extra_edges(string tag, vector<MatrixXf> &edge_attr)
{
    vector<vector<int>> edge_shape;
    loadtxt(root + "/" + tag + "_edge_shape.txt", edge_shape);

    vector<vector<float>> f_edge_attr;
    loadtxt(root + "/" + tag + "_edge_attr.txt", f_edge_attr);

    int current_edge = 0;
    for (vector<int> shape : edge_shape)
    {
        int n_edges = shape[0];
        vector<vector<float>> edges;
        for (int first_edge = current_edge; current_edge - first_edge < n_edges; current_edge++)
        {
            edges.push_back(f_edge_attr[current_edge]);
        }
        edge_attr.push_back(to_eigen(edges));
    }
}

