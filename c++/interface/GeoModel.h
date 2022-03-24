#ifndef GEOMODEL_H
#define GEOMODEL_H

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "GCNConv.h"
#include "CfgParser.h"

namespace TorchUtils
{
    class Scaler
    {
    public:
        Scaler(std::vector<float> minims, std::vector<float> maxims, std::vector<float> means, std::vector<float> stdvs);
        void raw(Eigen::MatrixXf &x);
        void normalize(Eigen::MatrixXf &x);
        void standardize(Eigen::MatrixXf &x);
        void scale(Eigen::MatrixXf &x, std::string type);

    private:
        std::vector<float> minims;
        std::vector<float> maxims;
        std::vector<float> means;
        std::vector<float> stdvs;
    };

    class GeoModel
    {
    public:
        std::vector<Layer *> sequence;
        GeoModel(std::string root);
        std::tuple<std::vector<float>, std::vector<float>> evaluate(std::vector<std::vector<float>> &x, std::vector<std::vector<int>> &edge_index, std::vector<std::vector<float>> &edge_attr);
        std::tuple<std::vector<float>, std::vector<float>> evaluate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void scale(Eigen::MatrixXf &x, Eigen::MatrixXf &edge_attr);
        void mask(Eigen::MatrixXf &x, Eigen::MatrixXf &edge_attr);
        void print();

    private:
        std::string root;
        CfgParser cfg;

        std::vector<Eigen::MatrixXf> get_weights(std::vector<std::vector<int>> shapes);
        std::vector<Eigen::MatrixXf> get_biases(std::vector<std::vector<int>> shapes);

        void init_scaler();
        std::string scale_type;
        Scaler *node_scaler;
        Scaler *edge_scaler;

        void init_masks();
        std::vector<int> node_mask;
        std::vector<int> edge_mask;
    };
}

#endif // GEOMODEL_H