
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "cpp_geometric.h"

using namespace std;
using namespace Eigen;

TorchUtils::Dataset dataset("../data_csv");

const vector<vector<float>> test_x = {{0.0512, 0.0819, 0.5242, 0.8290, 0.6909},
                                      {0.0664, 0.0803, 0.4328, 0.1801, 0.1115},
                                      {0.0477, 0.0498, 0.5256, 0.7370, 0.9995},
                                      {0.0423, 0.0461, 0.5412, 0.4574, 1.0000},
                                      {0.0332, 0.0345, 0.6013, 0.2524, 0.7095},
                                      {0.0124, 0.0228, 0.4840, 0.3787, 0.0775},
                                      {0.0137, 0.0150, 0.6543, 0.5511, 0.0064}};

const vector<vector<int>> test_edge_index = {{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5},
                                             {1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 5, 6, 6}};

const vector<vector<float>> test_edge_attr = {{0.2260}, {0.0546}, {0.2210}, {0.2618}, {0.2698}, {0.2068}, {0.2775}, {0.1947}, {0.1674}, {0.1277}, {0.3061}, {0.1665}, {0.2965}, {0.2162}, {0.1656}, {0.1346}, {0.0720}, {0.1220}, {0.1352}, {0.1843}, {0.1928}};

void test_Linear()
{
  vector<vector<float>> vec_x = test_x;
  MatrixXf x = TorchUtils::to_eigen(vec_x);
  vector<vector<float>> vec_edge_attr = test_edge_attr;
  MatrixXf edge_attr = TorchUtils::to_eigen(vec_edge_attr);
  vector<vector<int>> edge_index = test_edge_index;

  printf("Testing TorchUtils::Linear Layer...\n");
  TorchUtils::Linear linear(5, 2);
  linear.print_parameters();

  printf("Setting parameters...\n");
  linear.set_parameters({{0.3498, -0.0110, -0.4418, -0.3503, 0.2580},
                         {-0.1232, 0.2168, 0.3799, -0.3032, -0.4404}},
                        {{-0.0825, 0.2490}});
  linear.print_parameters();

  printf("Applying Linear layer on ...\n");
  TorchUtils::print_matrix(x, "org");

  vector<vector<float>> vec_exp_x = {{-0.4092, -0.0960},
                                     {-0.2857, 0.3190},
                                     {-0.2989, -0.2101},
                                     {-0.2095, -0.1197},
                                     {-0.2423, 0.0918},
                                     {-0.4049, 0.2873},
                                     {-0.5584, 0.3292}};
  MatrixXf exp_x = TorchUtils::to_eigen(vec_exp_x);
  TorchUtils::print_matrix(exp_x, "exp");
  linear.apply(x);
  TorchUtils::print_matrix(x, "new");
  TorchUtils::compare_matrix(exp_x, x);

  printf("Finished Testing TorchUtils::Linear Layer\n");
}

void test_GCNConv()
{
  vector<vector<float>> vec_x = test_x;
  MatrixXf x = TorchUtils::to_eigen(vec_x);
  vector<vector<float>> vec_edge_attr = test_edge_attr;
  MatrixXf edge_attr = TorchUtils::to_eigen(vec_edge_attr);
  vector<vector<int>> edge_index = test_edge_index;

  printf("Testing TorchUtils::GCNConv Layer...\n");
  TorchUtils::GCNConv conv(5, 1, 2);
  printf("Setting parameters...\n");
  conv.set_parameters({{-0.1274, 0.2725, -0.2208, 0.0745, 0.2331, -0.0438, 0.0009, 0.0686,
                        -0.0600, -0.0723, -0.0197},
                       {-0.2133, -0.0064, 0.1091, 0.0283, -0.1003, -0.2419, 0.0016, -0.2718,
                        -0.0960, -0.1954, -0.0102}},
                      {{-0.0080, -0.2720}});
  conv.print_parameters();

  printf("Applying GCNConv layer on ...\n");
  TorchUtils::print_matrix(x, "org");
  vector<vector<float>> vec_exp_x = {{0.0000, 0.0000},
                                     {-0.1290, -0.4445},
                                     {0.4433, -0.3137},
                                     {0.4822, -0.7074},
                                     {0.0793, -1.1414},
                                     {-0.5888, -1.8665},
                                     {-0.9600, -1.6114}};
  MatrixXf exp_x = TorchUtils::to_eigen(vec_exp_x);
  conv.apply(x, edge_index, edge_attr);
  TorchUtils::print_matrix(x, "new");
  TorchUtils::compare_matrix(exp_x, x);

  printf("Finished Testing TorchUtils::GCNConv Layer\n");
}

void test_Eigen()
{  
  std::vector<std::vector<float>> x = test_x;
  MatrixXf mat_x = TorchUtils::to_eigen(x);
  TorchUtils::print_matrix(mat_x, "org");

  std::vector<std::vector<float>> w = {{-0.4911, 0.8553, 0.5390, 0.1936, 0.7807},
                                       {-0.6773, -0.3715, 0.2942, 0.5200, 0.7096}};
  MatrixXf mat_w = TorchUtils::to_eigen(w);
  TorchUtils::print_matrix(mat_w, "weights");

  std::vector<std::vector<float>> b = {{0.2344, 0.3054}};
  MatrixXf mat_b = TorchUtils::to_eigen(b);
  TorchUtils::print_matrix(mat_b, "bias");

  TorchUtils::print_matrix(mat_x * mat_w.transpose() + MatrixXf::Ones(mat_x.rows(), 1) * mat_b, "w*x+b");
}

void test_Slicing()
{
  vector<vector<float>> vec_x = test_x;
  MatrixXf x = TorchUtils::to_eigen(vec_x);
  vector<vector<float>> vec_edge_attr = test_edge_attr;
  MatrixXf edge_attr = TorchUtils::to_eigen(vec_edge_attr);
  vector<vector<int>> edge_index = test_edge_index;

  vector<int> rows = edge_index[0];
  MatrixXf x_j = x(rows, Eigen::placeholders::all);
  TorchUtils::print_matrix(x_j, "x_j");

  TorchUtils::print_matrix(edge_attr * MatrixXf::Ones(1, 5), "edge_attr");
}

void test_ScatterAdd()
{
  vector<vector<float>> vec_x = test_x;
  MatrixXf x = TorchUtils::to_eigen(vec_x);
  vector<vector<float>> vec_edge_attr = test_edge_attr;
  MatrixXf edge_attr = TorchUtils::to_eigen(vec_edge_attr);
  vector<vector<int>> edge_index = test_edge_index;

  vector<int> cols = edge_index[1];

  int n_edges = cols.size();
  int n_node_features = x.cols();
  MatrixXf out = MatrixXf::Zero(x.rows(), x.cols());
  for (int i = 0; i < n_edges; i++)
  {
    for (int j = 0; j < n_node_features; j++)
    {
      out(cols[i], j) += edge_attr(i, 0);
    }
  }
  TorchUtils::print_matrix(out);
}

void test_Dataset()
{
  printf("Testing TorchUtils::Dataset...\n");

  cout << "Loaded in " << dataset.size() << " graphs." << endl;
  TorchUtils::Graph g = dataset.at(0);
  g.print();
  printf("Finished Testing TorchUtils::Dataset\n");
}

void test_Extra_Dataset()
{
  printf("Testing TorchUtils::Dataset::load_extra...\n");

  dataset.load_extra("gcnconvmsg");

  cout << "Loaded in " << dataset.size() << " graphs." << endl;

  MatrixXf node_x, edge_attr;
  dataset[0].get_extra("gcnconvmsg", node_x, edge_attr);

  TorchUtils::print_matrix(node_x, "GCNConvMSG::node_o");
  TorchUtils::print_matrix(edge_attr, "GCNConvMSG::edge_o");

  printf("Finished Testing TorchUtils::Dataset::load_extra\n");
}

void test_Init_Layer()
{
  printf("Testing TorchUtils::Layer Init...\n");
  TorchUtils::GCNConv conv(5, 1, 2);

  printf("Pre Init...\n");
  conv.print_parameters();

  TorchUtils::initialize_layer(conv);
  printf("Post Init...\n");
  conv.print_parameters();

  printf("Finished Testing TorchUtils::Layer Init\n");
}

void test_GCNConvMSG()
{
  char test[] = "TorchUtils::GCNConvMSG";
  printf("Testing %s...\n", test);

  printf("Loading extra dataset...\n");
  dataset.load_extra("gcnconvmsg");

  printf("Initialize GCNConvMSG...\n");
  TorchUtils::GCNConvMSG conv(5, 1, 2);
  TorchUtils::initialize_layer(conv);

  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error = 0;
  float edge_error = 0;
  for (unsigned i = 0; i < dataset.size(); i++)
  {
    TorchUtils::Graph g = dataset[i];
    MatrixXf node_o = g.node_x;
    MatrixXf edge_o = g.edge_attr;
    conv.apply(node_o, g.edge_index, edge_o);

    MatrixXf node_targ, edge_targ;
    g.get_extra("gcnconvmsg", node_targ, edge_targ);

    node_error += TorchUtils::matrix_difference(node_targ,node_o);
    edge_error += TorchUtils::matrix_difference(edge_targ, edge_o);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);

  printf("Finished Testing %s\n", test);
}

void test_GeoModel_conv1()
{
  char test[] = "TorchUtils::GeoModel::Conv1";
  printf("Testing %s...\n", test);

  printf("Loading model...\n");
  TorchUtils::GeoModel model("../gnn_model");
  model.print();

  dataset.load_extra("golden_conv1");

  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error = 0;
  float edge_error = 0;
  for (TorchUtils::Graph g : dataset)
  {
    Eigen::MatrixXf node_targ, edge_targ;
    g.get_extra("golden_conv1",node_targ,edge_targ);

    MatrixXf node_o = g.node_x;
    MatrixXf edge_o = g.edge_attr;
    model.sequence[0]->apply(node_o, g.edge_index, edge_o);

    node_error += TorchUtils::matrix_difference(node_targ,node_o);
    edge_error += TorchUtils::matrix_difference(edge_targ, edge_o);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);
  printf("Finished Testing %s\n", test);

}
void test_GeoModel()
{
  char test[] = "TorchUtils::GeoModel";
  printf("Testing %s...\n", test);

  printf("Loading model...\n");
  TorchUtils::GeoModel model("../gnn_model");
  model.print();

  dataset.load_extra("golden");

  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error = 0;
  float edge_error = 0;
  for (TorchUtils::Graph g : dataset)
  {
    Eigen::MatrixXf node_targ, edge_targ;
    g.get_extra("golden",node_targ,edge_targ);

    MatrixXf node_o = g.node_x;
    MatrixXf edge_o = g.edge_attr;
    model.apply(node_o, g.edge_index, edge_o);

    node_error += TorchUtils::matrix_difference(node_targ,node_o);
    edge_error += TorchUtils::matrix_difference(edge_targ, edge_o);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);
  printf("Finished Testing %s\n", test);

}

void test_GCNRelu()
{
  
  char test[] = "TorchUtils::GCNRelu";
  printf("Testing %s...\n", test);

  TorchUtils::GCNRelu relu;
  
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  for (unsigned i = 0; i < dataset.size(); i++)
  {
    TorchUtils::Graph g = dataset[i];
    MatrixXf node_o = -1*g.node_x;
    MatrixXf edge_o = g.edge_attr;

    TorchUtils::print_matrix(node_o,"Negative Input");
    relu.apply(node_o, g.edge_index, edge_o);
    TorchUtils::print_matrix(node_o,"Relu Output");
    break;
  }
  printf("Finished Testing %s\n", test);
}

void test_GCNLogSoftmax()
{
  
  char test[] = "TorchUtils::GCNLogSoftmax";
  printf("Testing %s...\n", test);

  TorchUtils::GCNLogSoftmax log_softmax;
  
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  for (unsigned i = 0; i < dataset.size(); i++)
  {
    TorchUtils::Graph g = dataset[i];
    MatrixXf node_o = g.node_x;
    MatrixXf edge_o = g.edge_attr;

    TorchUtils::print_matrix(node_o,"Input");
    log_softmax.apply(node_o, g.edge_index, edge_o);
    TorchUtils::print_matrix(node_o,"log_softmax Output");
    break;
  }
  printf("Finished Testing %s\n", test);
}

void test_GeoModel_Scale()
{

  char test[] = "TorchUtils::GeoModel::scale";
  printf("Testing %s...\n", test);

  printf("Loading model...\n");
  TorchUtils::GeoModel model("../gnn_model");
  model.print();

  dataset.load_extra("unscaled");
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error, edge_error;
  node_error = edge_error = 0;
  for (TorchUtils::Graph g : dataset)
  {
    Eigen::MatrixXf node_o, edge_o;
    g.get_extra("unscaled",node_o,edge_o);

    MatrixXf node_targ = g.node_x;
    MatrixXf edge_targ = g.edge_attr;
    model.scale(node_o, edge_o);

    node_error += TorchUtils::matrix_difference(node_targ,node_o);
    edge_error += TorchUtils::matrix_difference(edge_targ, edge_o);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);
  printf("Finished Testing %s\n", test);
}

float vector_difference(vector<float> targ, vector<float> test)
{
  float error = 0;
  for (unsigned int i = 0; i < targ.size(); i++)
  {
    error += (targ[i] - test[i]) * (targ[i] - test[i]);
  }
  return error;
}

void test_GeoModel_Evaluate()
{
  char test[] = "TorchUtils::GeoModel::evaluate";
  printf("Testing %s...\n", test);

  printf("Loading model...\n");
  TorchUtils::GeoModel model("../gnn_model");
  model.print();

  dataset.load_extra("unscaled");
  dataset.load_extra("golden");
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error, edge_error;
  node_error = edge_error = 0;
  for (TorchUtils::Graph g : dataset)
  {
    Eigen::MatrixXf node_o, edge_o;
    g.get_extra("unscaled",node_o,edge_o);

    MatrixXf m_node_targ, m_edge_targ;
    g.get_extra("golden", m_node_targ, m_edge_targ);

    vector<float> node_targ(m_node_targ.rows());
    for (unsigned int i = 0; i < m_node_targ.rows(); i++)
    {
      node_targ[i] = exp(m_node_targ(i, 1));
    }

    vector<float> edge_targ(m_edge_targ.rows());
    for (unsigned int i = 0; i < m_edge_targ.rows(); i++)
    {
      edge_targ[i] = exp(m_edge_targ(i, 1));
    }

      tuple<vector<float>, vector<float>> pred = model.evaluate(node_o, g.edge_index, edge_o);

    vector<float> node_pred = get<0>(pred);
    vector<float> edge_pred = get<1>(pred);

    node_error += vector_difference(node_targ,node_pred);
    edge_error += vector_difference(edge_targ, edge_pred);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);
  printf("Finished Testing %s\n", test);
}


void test_GeoModel_Evaluate_Golden()
{
  char test[] = "TorchUtils::GeoModel::evaluate";
  printf("Testing %s...\n", test);

  printf("Loading model...\n");
  TorchUtils::GeoModel model("../golden_gcn");
  model.print();

  TorchUtils::Dataset dataset("../golden_csv");

  dataset.load_extra("golden");
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error, edge_error;
  node_error = edge_error = 0;
  for (TorchUtils::Graph g : dataset)
  {
    Eigen::MatrixXf node_o, edge_o;
    node_o = g.node_x;
    edge_o = g.edge_attr;

    MatrixXf m_node_targ, m_edge_targ;
    g.get_extra("golden", m_node_targ, m_edge_targ);

    vector<float> node_targ(m_node_targ.rows());
    for (unsigned int i = 0; i < m_node_targ.rows(); i++)
    {
      node_targ[i] = exp(m_node_targ(i, 1));
    }

    vector<float> edge_targ(m_edge_targ.rows());
    for (unsigned int i = 0; i < m_edge_targ.rows(); i++)
    {
      edge_targ[i] = exp(m_edge_targ(i, 1));
    }

      tuple<vector<float>, vector<float>> pred = model.evaluate(node_o, g.edge_index, edge_o);

    vector<float> node_pred = get<0>(pred);
    vector<float> edge_pred = get<1>(pred);

    node_error += vector_difference(node_targ,node_pred);
    edge_error += vector_difference(edge_targ, edge_pred);
  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);
  printf("Finished Testing %s\n", test);
}

int main()
{
  // test_Linear();
  // test_GCNConv();
  // test_Eigen();
  // test_Slicing();
  // test_ScatterAdd();
  // test_Dataset();
  // test_Extra_Dataset();
  // test_Init_Layer();
  // test_GCNConvMSG();
  // test_GCNLogSoftmax();
  // test_GeoModel();
  // test_GeoModel_Scale();
  // test_GeoModel_Evaluate();
}
