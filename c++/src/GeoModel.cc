#include "GeoModel.h"

#include <algorithm>

using namespace std;
using namespace Eigen;

TorchUtils::Scaler::Scaler(vector<float> minims, vector<float> maxims, vector<float> means, vector<float> stdvs)
{
    this->minims = minims;
    this->maxims = maxims;
    this->means = means;
    this->stdvs = stdvs;
}

void TorchUtils::Scaler::raw(MatrixXf &x) {}
void TorchUtils::Scaler::normalize(MatrixXf &x)
{
    for (unsigned int i = 0; i < minims.size(); i++)
    {
        for (unsigned int j = 0; j < x.rows(); j++)
        {
            x(j, i) = (x(j, i) - minims[i]) / (maxims[i] - minims[i]);
        }
    }
}
void TorchUtils::Scaler::standardize(MatrixXf &x)
{
    for (unsigned int i = 0; i < minims.size(); i++)
    {
        for (unsigned int j = 0; j < x.rows(); j++)
        {
            x(j, i) = (x(j, i) - means[i]) / stdvs[i];
        }
    }
}
void TorchUtils::Scaler::scale(MatrixXf &x, string type)
{
    if (type == "raw")
        raw(x);
    if (type == "normalize")
        normalize(x);
    if (type == "standardize")
        standardize(x);
}

void TorchUtils::GeoModel::init_scaler()
{
    scale_type = cfg.readStringOpt("features", "scale");

    node_scaler = new Scaler(
        cfg.readFloatListOpt("scaler", "node_scale_min"),
        cfg.readFloatListOpt("scaler", "node_scale_max"),
        cfg.readFloatListOpt("scaler", "node_scale_mean"),
        cfg.readFloatListOpt("scaler", "node_scale_std"));

    edge_scaler = new Scaler(
        cfg.readFloatListOpt("scaler", "edge_scale_min"),
        cfg.readFloatListOpt("scaler", "edge_scale_max"),
        cfg.readFloatListOpt("scaler", "edge_scale_mean"),
        cfg.readFloatListOpt("scaler", "edge_scale_std"));
}

vector<int> get_mask(vector<string> str_features, vector<string> str_mask)
{
    vector<int> mask;

    if (str_features.size() == str_mask.size())
        return mask;

    for (unsigned int i = 0; i < str_features.size(); i++)
    {
        if (std::find(str_mask.begin(), str_mask.end(), str_features[i]) != str_mask.end())
        {
            mask.push_back(i);
        }
    }

    return mask;
}

void TorchUtils::GeoModel::init_masks()
{
    vector<string> str_node_features = cfg.readStringListOpt("features", "node_features");
    vector<string> str_node_mask = cfg.readStringListOpt("features", "node_mask");
    node_mask = get_mask(str_node_features, str_node_mask);

    vector<string> str_edge_features = cfg.readStringListOpt("features", "edge_features");
    vector<string> str_edge_mask = cfg.readStringListOpt("features", "edge_mask");
    edge_mask = get_mask(str_edge_features, str_edge_mask);
}

TorchUtils::GeoModel::GeoModel(string root)
{
    this->root = root;
    cfg.init(root + "/model.cfg");

    init_scaler();
    init_masks();

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

void TorchUtils::GeoModel::scale(MatrixXf &x, MatrixXf &edge_attr)
{
    node_scaler->scale(x, scale_type);
    edge_scaler->scale(edge_attr, scale_type);
}

void _mask(MatrixXf &x, vector<int> mask)
{
    MatrixXf masked_x(x.rows(), mask.size());
    for (unsigned int row = 0; row < x.rows(); row++)
    {
        for (unsigned int col = 0; col < mask.size(); col++)
        {
            masked_x(row,col) = x(row, mask[col]);
        }
    }
    x = masked_x;
}

void TorchUtils::GeoModel::mask(MatrixXf &x, MatrixXf &edge_attr)
{
    if (node_mask.size() > 0)
    {
        // x = x(Eigen::placeholders::all, node_mask);
        _mask(x, node_mask);

    }

    if (edge_mask.size() > 0)
    {
        // edge_attr = edge_attr(Eigen::placeholders::all, edge_mask);
        _mask(edge_attr, edge_mask);
    }
}

tuple<vector<float>,vector<float>> TorchUtils::GeoModel::evaluate(vector<vector<float>> &x, vector<vector<int>> &edge_index, vector<vector<float>> &edge_attr)
{
    MatrixXf m_node_o = to_eigen(x);
    MatrixXf m_edge_o = to_eigen(edge_attr);
    return evaluate(m_node_o,edge_index,m_edge_o);
}
tuple<vector<float>,vector<float>> TorchUtils::GeoModel::evaluate(MatrixXf &x, vector<vector<int>> &edge_index, MatrixXf &edge_attr)
{
    scale(x, edge_attr);
    mask(x, edge_attr);
    apply(x, edge_index, edge_attr);

    vector<float> node_o(x.rows());
    for (unsigned int i = 0; i < x.rows(); i++)
    {
        node_o[i] = exp(x(i, 1));
    }

    vector<float> edge_o(edge_attr.rows());
    for (unsigned int i = 0; i < edge_attr.rows(); i++)
    {
        edge_o[i] = exp(edge_attr(i, 1));
    }
    return make_tuple(node_o, edge_o);
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