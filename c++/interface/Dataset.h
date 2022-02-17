#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include <Eigen/Dense>
#include "TorchUtils.h"

namespace TorchUtils
{

    struct Graph
    {
        Eigen::MatrixXf node_x;
        std::vector<std::vector<int>> edge_index;
        Eigen::MatrixXf edge_attr;
        Graph(Eigen::MatrixXf node_x,std::vector<std::vector<int>> edge_index,Eigen::MatrixXf edge_attr)
        {
            this->node_x = node_x;
            this->edge_index = edge_index;
            this->edge_attr = edge_attr;
        }

        void print()
        {
            print_matrix(node_x, "node_x");
            print_vector(edge_index[0], "edge_src");
            print_vector(edge_index[1], "edge_dest");
            print_matrix(edge_attr, "edge_attr");
        }
    };

    class Dataset : public std::vector<Graph>
    {
    public:
        std::string root;
        Dataset(std::string root);

    private:
        template <typename T>
        void loadtxt(std::string fname, std::vector<std::vector<T>> &out);
        void load_nodes(std::vector<Eigen::MatrixXf> &node_x);
        void load_edges(std::vector<std::vector<std::vector<int>>> &edge_index,std::vector<Eigen::MatrixXf> &edge_attr);
    };
}

#endif // DATASET_H