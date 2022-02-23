#include "GeoModel.h"

using namespace std;
using namespace Eigen;

TorchUtils::GeoModel::GeoModel(string root)
{
    this->root = root;
    cfg.init(root + "/model.cfg");

    vector<string> layers = cfg.readStringListOpt("model", "layers");
    vector<int> f_shapes = cfg.readIntListOpt("model", "layer_shapes");
    vector<vector<int>> shapes;
    for (unsigned int i = 0; i < f_shapes.size(); i += 2)
    {
        vector<int> shape = {f_shapes[i], f_shapes[i + 1]};
        shapes.push_back(shape);
    }

    vector<MatrixXf> weights = get_weights(shapes);
    vector<MatrixXf> biases = get_biases(shapes);

    for (unsigned int i = 0; i < layers.size(); i++)
    {
        Layer *layer;
        string name = layers[i];
        vector<int> shape = shapes[i];
        MatrixXf weight = weights[i];
        MatrixXf bias = biases[i];

        if (name == "GCNConv")
        {
            layer = new GCNConv(shape[1], shape[0]);
        }
        else if (name == "GCNConvMSG")
        {
            layer = new GCNConvMSG(shape[1], shape[0]);
        }
        else if (name == "NodeLinear")
        {
            layer = new NodeLinear(shape[1], shape[0]);
        }
        else if (name == "EdgeLinear")
        {
            layer = new EdgeLinear(shape[1], shape[0]);
        }
        else if (name == "GCNRelu")
        {
            layer = new GCNRelu();
        }
        else if (name == "GCNLogSoftmax")
        {
            layer = new GCNLogSoftmax();
        }

        if (shape[0] * shape[1] > 0)
        {
            // cout << "Setting " << layer->name() << " Parameters" << endl;
            layer->set_parameters(weight, bias);
            // layer->print_parameters();
        }
        sequence.push_back(layer);
    }
}

void TorchUtils::GeoModel::apply(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    for (Layer *layer : sequence)
    {
        layer->apply(x, edge_index, edge_attr);
    }
}

void TorchUtils::GeoModel::print()
{
    for (Layer *layer : sequence)
    {
        layer->print_shapes();
    }
}

vector<vector<float>> reshape_vector(vector<float> weight, int nrows, int ncols)
{
    if (nrows * ncols != (int)weight.size())
    {
        printf("Cannot reshape vector(1,%i) to array(%i,%i)\n", (int)weight.size(), nrows, ncols);
    }
    vector<vector<float>> reshaped(nrows, vector<float>(ncols));
    for (unsigned int i = 0; i < weight.size(); i++)
    {
        int row = i / ncols;
        int col = i % ncols;
        reshaped[row][col] = weight[i];
    }
    return reshaped;
}

vector<MatrixXf> TorchUtils::GeoModel::get_weights(vector<vector<int>> shapes)
{
    vector<MatrixXf> weights;

    vector<vector<float>> f_weights;
    loadtxt(root + "/weights.txt", f_weights);

    int current = 0;
    for (vector<int> shape : shapes)
    {
        int nrows = shape[0];
        int ncols = shape[1];
        vector<float> f_weight;
        for (int first = current; current - first < nrows * ncols; current++)
        {
            f_weight.push_back(f_weights[current][0]);
        }

        MatrixXf weight(nrows, ncols);

        if (nrows * ncols > 0)
        {
            weight = to_eigen( reshape_vector(f_weight,nrows,ncols) );
        }

        weights.push_back(weight);
    }

    return weights;
}

vector<MatrixXf> TorchUtils::GeoModel::get_biases(vector<vector<int>> shapes)
{
    vector<MatrixXf> biases;

    vector<vector<float>> f_biases;
    loadtxt(root + "/bias.txt", f_biases);

    int current = 0;
    for (vector<int> shape : shapes)
    {
        int nrows = shape[0];
        vector<float> f_bias;
        for (int first = current; current - first < nrows; current++)
        {
            f_bias.push_back(f_biases[current][0]);
        }

        MatrixXf bias(1, nrows);

        if (nrows > 0)
        {
            bias = to_eigen(reshape_vector(f_bias, 1, nrows));
        }
        biases.push_back(bias);
    }

    return biases;
}