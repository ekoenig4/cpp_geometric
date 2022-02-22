#ifndef TORCHUTILS_H
#define TORCHUTILS_H

#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense>

namespace TorchUtils
{
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
        std::string name = "layer";
        Eigen::MatrixXf weights;
        Eigen::MatrixXf bias;

        void apply(Eigen::MatrixXf &x);
        void set_weights(std::vector<std::vector<float>> weights);
        void set_bias(std::vector<std::vector<float>> bias);
        void set_parameters(std::vector<std::vector<float>> weights, std::vector<std::vector<float>> bias);
        void print_parameters();
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
    };

    struct ReLu : public Layer
    {
        void apply(Eigen::MatrixXf &x);
    };
}
#endif // TORHCUTILS_H