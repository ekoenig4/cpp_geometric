#ifndef GCNCONV_H
#define GCNCONV_H

#include "TorchUtils.h"

namespace TorchUtils
{

    struct GCNConv : public Linear
    {
        Eigen::MatrixXf msg;

        GCNConv(int n_in_node, int n_in_edge, int n_out) : Linear(2*n_in_node+n_in_edge,n_out) {}
        GCNConv(int n_in, int n_out) : Linear(n_in,n_out) {}
        std::string name() { return "GCNConv"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { Linear::apply(x); }
        Eigen::MatrixXf message(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void aggregate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr, Eigen::MatrixXf &msg);
        void propagate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
    };

    struct GCNConvMSG : public GCNConv
    {
        GCNConvMSG(int n_in_node, int n_in_edge, int n_out) : GCNConv(n_in_node, n_in_edge, n_out) {}
        GCNConvMSG(int n_in, int n_out) : GCNConv(n_in, n_out) {}
        std::string name() { return "GCNConvMSG"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { Linear::apply(x); }
    };

    struct NodeLinear : public Linear
    {
        NodeLinear(int n_in, int n_out) : Linear(n_in,n_out) {}
        std::string name() { return "NodeLinear"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { Linear::apply(x); }
    };

    struct EdgeLinear : public Linear
    {
        EdgeLinear(int n_in, int n_out) : Linear(n_in,n_out) {}
        std::string name() { return "EdgeLinear"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { Linear::apply(x); }
    };

    struct GCNRelu : public Layer
    {
        std::string name() { return "GCNRelu"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { relu(x); }
    };

    struct GCNLogSoftmax : public Layer
    {
        std::string name() { return "GCNLogSoftmax"; }
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x) { log_softmax(x); }
        // void apply(Eigen::MatrixXf &x) {  }
    };
}

#endif // GCNCONV_H