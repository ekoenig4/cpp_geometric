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
    class GeoModel
    {
    public:
        std::vector<Layer *> sequence;
        GeoModel(std::string root);
        std::tuple<std::vector<float>, std::vector<float>> evaluate(std::vector<std::vector<float>> &x, std::vector<std::vector<int>> &edge_index, std::vector<std::vector<float>> &edge_attr);
        std::tuple<std::vector<float>, std::vector<float>> evaluate(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void scale(Eigen::MatrixXf &x, Eigen::MatrixXf &edge_attr);
        void print();

    private:
        std::string root;
        CfgParser cfg;

        std::vector<Eigen::MatrixXf> get_weights(std::vector<std::vector<int>> shapes);
        std::vector<Eigen::MatrixXf> get_biases(std::vector<std::vector<int>> shapes);

        std::vector<float> node_scale_min;
        std::vector<float> node_scale_max;
        std::vector<float> edge_scale_min;
        std::vector<float> edge_scale_max;
    };
}

#endif // GEOMODEL_H