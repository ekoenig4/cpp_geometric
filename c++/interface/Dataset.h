#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <Eigen/Dense>
#include "TorchUtils.h"

namespace TorchUtils
{

    struct Graph
    {
        Eigen::MatrixXf node_x;
        std::vector<std::vector<int>> edge_index;
        Eigen::MatrixXf edge_attr;

        std::map<std::string,Eigen::MatrixXf> extra_node_x;
        std::map<std::string,Eigen::MatrixXf> extra_edge_attr;

        Graph(Eigen::MatrixXf node_x,std::vector<std::vector<int>> edge_index,Eigen::MatrixXf edge_attr)
        {
            this->node_x = node_x;
            this->edge_index = edge_index;
            this->edge_attr = edge_attr;
        }

        void add_extra(std::string tag, Eigen::MatrixXf node_x, Eigen::MatrixXf edge_attr)
        {
            extra_node_x[tag] = node_x;
            extra_edge_attr[tag] = edge_attr;
        }

        void get_extra(std::string tag, Eigen::MatrixXf &node_x, Eigen::MatrixXf &edge_attr)
        {
            node_x = extra_node_x[tag];
            edge_attr = extra_edge_attr[tag];
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
        void load_extra(std::string tag);

    private:
        void load_nodes(std::vector<Eigen::MatrixXf> &node_x);
        void load_edges(std::vector<std::vector<std::vector<int>>> &edge_index,std::vector<Eigen::MatrixXf> &edge_attr);

        void load_extra_nodes(std::string tag, std::vector<Eigen::MatrixXf> &node_x);
        void load_extra_edges(std::string tag, std::vector<Eigen::MatrixXf> &edge_attr);

    };
}

#endif // DATASET_H