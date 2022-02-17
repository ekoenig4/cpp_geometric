#include "GCNConv.h"

using namespace std;
using namespace Eigen;

TorchUtils::GCNConv::GCNConv(int n_in_node, int n_in_edge, int n_out) : Linear(2*n_in_node+n_in_edge,n_out)
{
    this->n_in_node = n_in_node;
    this->n_in_edge = n_in_edge;
    this->n_out = n_out;
}

void TorchUtils::GCNConv::apply(Eigen::MatrixXf &x, vector<vector<int>> &edge_index, Eigen::MatrixXf &edge_attr)
{
    return propagate(x, edge_index, edge_attr);
}

Eigen::MatrixXf TorchUtils::GCNConv::message(Eigen::MatrixXf &x, vector<vector<int>> &edge_index, Eigen::MatrixXf &edge_attr)
{
    vector<int> src = edge_index[0];
    vector<int> dest = edge_index[1];
    MatrixXf x_i = x(dest, Eigen::placeholders::all);
    MatrixXf x_j = x(src, Eigen::placeholders::all);

    // Concatenate src node, dest node, and edge features
    // output will have have (n_edges,2*n_node_features+n_edge_features)
    int n_edges = edge_attr.rows();
    int n_node_features = x_i.cols();
    int n_edge_features = edge_attr.cols();
    MatrixXf msg(n_edges, 2 * n_node_features + n_edge_features);
    msg << x_i, x_j - x_i, edge_attr;

    // Apply linear layer to msg as defined by constructor
    Linear::apply(msg);

    return msg;
}

void TorchUtils::GCNConv::aggregate(Eigen::MatrixXf &x, vector<vector<int>> &edge_index, Eigen::MatrixXf &edge_attr, Eigen::MatrixXf &msg)
{
    /**
     * @brief Currently using add aggregate function
     * TODO implement scatter max
     *
     */
    scatter_add(x, edge_index, msg);
}

void TorchUtils::GCNConv::propagate(Eigen::MatrixXf &x, vector<vector<int>> &edge_index, Eigen::MatrixXf &edge_attr)
{
    Eigen::MatrixXf msg = message(x, edge_index, edge_attr);
    aggregate(x, edge_index, edge_attr, msg);
}