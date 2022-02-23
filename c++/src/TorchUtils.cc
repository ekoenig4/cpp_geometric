#include "TorchUtils.h"

#include "math.h"

using namespace std;
using namespace Eigen;

MatrixXf TorchUtils::to_eigen(std::vector<std::vector<float>> data)
{
    MatrixXf eMatrix(data.size(), data[0].size());
    for (unsigned int i = 0; i < data.size(); ++i)
        eMatrix.row(i) = VectorXf::Map(&data[i][0], data[0].size());
    return eMatrix;
}

void TorchUtils::print_shape(const MatrixXf mat, string name)
{
    int n = mat.rows();
    int m = mat.cols();

    cout << name << "(" << n << "," << m << "): " << endl;
}

void TorchUtils::print_matrix(const MatrixXf mat, string name)
{
    int n = mat.rows();
    int m = mat.cols();

    cout << name << "(" << n << "," << m << "): " << endl;
    cout << "--------------------" << endl;
    cout << mat << endl;
    cout << "--------------------" << endl << endl;
}

void TorchUtils::compare_matrix(const MatrixXf true_mat, const MatrixXf test)
{
    float compare = matrix_difference(true_mat, test);
    printf("Matrix Element Comparison to True: %f\n", compare);
}

float TorchUtils::matrix_difference(const MatrixXf true_mat, const MatrixXf test)
{

    int m = true_mat.cols();
    int n = true_mat.rows();

    if (m != test.cols() || n != test.rows())
    {
        printf("Test matrix doesn't have the same size as the true matrix\n");
    }

    float compare = (true_mat - test).squaredNorm();
    return compare;
}

void initialize_weights(MatrixXf &weights)
{
    int rows = weights.rows();
    int cols = weights.cols();

    vector<vector<float>> weight_init;
    for (int i = 0; i < rows; i++)
    {
        vector<float> init;
        for (int j = 0; j < cols; j++)
        {
            float elem = (i * cols + j + 1.0) / (rows * cols);
            init.push_back(elem);
        }
        weight_init.push_back(init);
    }
    weights = TorchUtils::to_eigen(weight_init);
}

void TorchUtils::initialize_layer(Layer &layer)
{
    initialize_weights(layer.weights);
    initialize_weights(layer.bias);
}

void TorchUtils::Layer::set_weights(vector<vector<float>> weights)
{
    set_weights(to_eigen(weights));
}

void TorchUtils::Layer::set_weights(MatrixXf weights)
{
    int m = weights.cols();
    int n = weights.rows();

    int n_out = this->weights.cols();
    int n_in = this->weights.rows();
    if (n != n_in || m != n_out)
    {
        printf("Expected weights(%i,%i), but got weights(%i,%i)\n", n_out, n_in, m, n);
    }

    this->weights = weights;
}

void TorchUtils::Layer::set_bias(vector<vector<float>> bias)
{
    set_bias(to_eigen(bias));
}

void TorchUtils::Layer::set_bias(MatrixXf bias)
{
    int m = bias.cols();
    int n = bias.rows();

    int n_out = this->bias.cols();
    int n_in = this->bias.rows();
    if (n != n_in || m != n_out)
    {
        printf("Expected bias(%i,%i), but got bias(%i,%i)\n", n_out, n_in, m, n);
    }

    this->bias = bias;
}

void TorchUtils::Layer::set_parameters(vector<vector<float>> weights, vector<vector<float>> bias)
{
    set_weights(weights);
    set_bias(bias);
}

void TorchUtils::Layer::set_parameters(MatrixXf weights, MatrixXf bias)
{
    set_weights(weights);
    set_bias(bias);
}

void TorchUtils::Layer::print_parameters()
{
    cout << "Layer: " << name() << endl;
    print_matrix(weights, "--weights");
    print_matrix(bias, "--bias");
}

void TorchUtils::Layer::print_shapes()
{
    cout << "Layer: " << name() << endl;
    print_shape(weights, "--weights");
    print_shape(bias, "--bias");
}

TorchUtils::Linear::Linear(int n, int m) : Layer()
{
    n_in = n;
    n_out = m;
    weights = MatrixXf(m, n);
    bias = MatrixXf(1, m);
}

void TorchUtils::Linear::apply(MatrixXf &x)
{
    // Compute the Linear algebra for the transform
    x = x * weights.transpose() + MatrixXf::Ones(x.rows(), 1) * bias;
}

void TorchUtils::Linear::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    apply(x);
}

void TorchUtils::scatter_add(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &msg)
{
    vector<int> dest = edge_index[1];

    int n_edges = dest.size();
    int n_nodes = x.rows();
    int n_out = msg.cols();
    MatrixXf out = MatrixXf::Zero(n_nodes,n_out);
    for (int i = 0; i < n_edges; i++)
    {
        for (int j = 0; j < n_out; j++)
        {
            out(dest[i], j) += msg(i, j);
        }
    }
    x = out;
}

void TorchUtils::relu(MatrixXf &x)
{
    int rows = x.rows();
    int cols = x.cols();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (x(i,j) < 0)
                x(i, j) = 0;
        }
    }
}

void TorchUtils::log_softmax(MatrixXf &x)
{
    int rows = x.rows();
    int cols = x.cols();
    for (int i = 0; i < rows; i++)
    {
        float norm = 0.0;
        for (int j = 0; j < cols; j++)
        {
            norm += exp(x(i, j));
        }

        for (int j = 0; j < cols; j++)
        {
            x(i, j) = log(exp(x(i, j)) / norm);
        }
    }
}