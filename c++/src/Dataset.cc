#include "Dataset.h"

// #include "TorchUtils.h"
#include <cstring> 

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


void TorchUtils::Dataset::load_edges(vector<vector<vector<int>>> &edge_index,vector<MatrixXf> &edge_attr)
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

template <typename T>
void TorchUtils::Dataset::loadtxt(string fname, vector<vector<T>> &out)
{
    ifstream file(fname);
    if (!file.is_open())
        throw runtime_error("Could not open file: "+fname);

    string line, word;
    string delim = " ";

    while(getline(file,line))
    {
        size_t pos = 0;
        string token;
        vector<T> vec;
        while ((pos = line.find(delim)) != string::npos)
        {
            token = line.substr(0, pos);
            vec.push_back(stof(token));
            line.erase(0, pos + delim.length());
        }
        vec.push_back(stof(line));
        out.push_back(vec);
    }
}
