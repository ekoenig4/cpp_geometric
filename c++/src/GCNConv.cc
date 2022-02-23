#include "GCNConv.h"

using namespace std;
using namespace Eigen;

void TorchUtils::GCNConv::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    propagate(x, edge_index, edge_attr);
}

MatrixXf TorchUtils::GCNConv::message(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
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
    msg = MatrixXf(n_edges, 2 * n_node_features + n_edge_features);
    msg << x_i, x_j - x_i, edge_attr;

    // Apply linear layer to msg as defined by constructor
    apply(msg);

    return msg;
}

void TorchUtils::GCNConv::aggregate(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr, MatrixXf &msg)
{
    /**
     * @brief Currently using add aggregate function
     * TODO implement scatter max
     *
     */
    scatter_add(x, edge_index, msg);
}

void TorchUtils::GCNConv::propagate(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    MatrixXf msg = message(x, edge_index, edge_attr);
    aggregate(x, edge_index, edge_attr, msg);
}

void TorchUtils::GCNConvMSG::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    propagate(x,edge_index,edge_attr);

    int n_edges = edge_attr.rows();

    vector<int> src = edge_index[0];
    vector<int> dest = edge_index[1];
    MatrixXf x_i = x(dest, Eigen::placeholders::all);
    MatrixXf x_j = x(src, Eigen::placeholders::all);

    edge_attr = MatrixXf(n_edges, 3 * n_out);
    edge_attr << x_i, x_j, msg;
}

void TorchUtils::NodeLinear::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    apply(x);
}

void TorchUtils::EdgeLinear::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    apply(edge_attr);
}

void TorchUtils::GCNRelu::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    apply(x);
    apply(edge_attr);
}

void TorchUtils::GCNLogSoftmax::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    apply(x);
    apply(edge_attr);
}