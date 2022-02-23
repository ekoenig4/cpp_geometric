#ifndef TORCHUTILS_H
#define TORCHUTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <Eigen/Dense>

namespace TorchUtils
{
    template <typename T>
    void loadtxt(std::string fname, std::vector<std::vector<T>> &out)
    {
        std::ifstream file(fname);
        if (!file.is_open())
            throw std::runtime_error("Could not open file: " + fname);

        std::string line, word;
        std::string delim = " ";

        while (std::getline(file, line))
        {
            size_t pos = 0;
            std::string token;
            std::vector<T> vec;
            while ((pos = line.find(delim)) != std::string::npos)
            {
                token = line.substr(0, pos);
                vec.push_back(stof(token));
                line.erase(0, pos + delim.length());
            }
            vec.push_back(stof(line));
            out.push_back(vec);
        }
    }

    Eigen::MatrixXf to_eigen(std::vector<std::vector<float>> data);
    void print_shape(const Eigen::MatrixXf mat, std::string name = "array");
    void print_matrix(const Eigen::MatrixXf mat, std::string name = "array");

    template <typename T>
    void print_vector(const std::vector<T> vec, std::string name = "array")
    {
        int m = vec.size();
        std::cout << name << "(" << m << "): {" << std::endl;
        for (int j = 0; j < m; j++)
        {
            std::cout << vec[j] << ",";
        }
        printf("\n}\n");
    }
    /**
     * @brief Compute the chi2 sum for all the elements
     *
     * @param true_mat matrix to reference
     * @param test matrix to test
     */
    void compare_matrix(const Eigen::MatrixXf true_mat, const Eigen::MatrixXf test);
    float matrix_difference(const Eigen::MatrixXf true_mat, const Eigen::MatrixXf test);
    void scatter_add(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &msg);

    struct Layer
    {
        Eigen::MatrixXf weights;
        Eigen::MatrixXf bias;

        virtual std::string name() { return "Layer"; }
        virtual void apply(Eigen::MatrixXf &x) = 0;
        virtual void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr) = 0;
        void set_weights(std::vector<std::vector<float>> weights);
        void set_weights(Eigen::MatrixXf weights);
        void set_bias(std::vector<std::vector<float>> bias);
        void set_bias(Eigen::MatrixXf bias);
        void set_parameters(std::vector<std::vector<float>> weights, std::vector<std::vector<float>> bias);
        void set_parameters(Eigen::MatrixXf weights, Eigen::MatrixXf bias);
        void print_parameters();
        void print_shapes();
    };
    void initialize_layer(Layer &layer);

    struct Linear : public Layer
    {
        int n_in;
        int n_out;

        /**
         * @brief Construct a new Linear object
         *
         * @param n number of input features
         * @param m number of output features
         */
        Linear(int n, int m);
        
        void apply(Eigen::MatrixXf &x);
        void apply(Eigen::MatrixXf &x, std::vector<std::vector<int>> &edge_index, Eigen::MatrixXf &edge_attr);
        std::string name() { return "Linear"; }
    };

    void relu(Eigen::MatrixXf &x);
    void log_softmax(Eigen::MatrixXf &x);
}
#endif // TORHCUTILS_H