#ifndef GCNCONV_H
#define GCNCONV_H

#include "TorchUtils.h"

namespace TorchUtils
{
    struct GCNConv : public Linear
    {
        int n_in_node;
        int n_in_edge;
        int n_out;

        GCNConv(int n_in_node, int n_in_edge, int n_out);
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        Eigen::MatrixXf message(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void aggregate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr, Eigen::MatrixXf &msg);
        void propagate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
    };
}

#endif // GCNCONV_H