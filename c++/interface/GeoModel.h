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
        std::vector<Layer*> sequence;
        GeoModel(std::string root);
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        void print();
    private:
        std::string root;
        CfgParser cfg;

        std::vector<Eigen::MatrixXf> get_weights(std::vector<std::vector<int>> shapes);
        std::vector<Eigen::MatrixXf> get_biases(std::vector<std::vector<int>> shapes);
    };
}

#endif // GEOMODEL_H