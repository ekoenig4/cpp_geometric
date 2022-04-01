
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include "cpp_geometric.h"

using namespace std;
using namespace Eigen;
using namespace TorchUtils;

Dataset dataset("/uscms_data/d3/ekoenig/8BAnalysis/studies/sixbStudies/jupyter/eightb/pairing_methods/graph_net/cpg_test_data/golden_gcn_no-btag");
GeoModel model("/uscms_data/d3/ekoenig/8BAnalysis/studies/sixbStudies/jupyter/eightb/pairing_methods/graph_net/cpg_models/golden_gcn_no-btag");


float vector_difference(vector<float> targ, vector<float> test)
{
  float error = 0;
  for (unsigned int i = 0; i < targ.size(); i++)
  {
    error += (targ[i] - test[i]) * (targ[i] - test[i]);
  }
  return error;
}

void test_Scale()
{
  char test[] = "TorchUtils::GeoModel::Scale";
  printf("Testing %s...\n", test);

  dataset.load_extra("prep");
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error, edge_error;
  node_error = edge_error = 0;
  for (Graph g : dataset)
  {
    Eigen::MatrixXf node_pred, edge_pred;
    node_pred = g.node_x;
    edge_pred = g.edge_attr;

    model.scale(node_pred, edge_pred);
    model.mask(node_pred, edge_pred);

    MatrixXf node_targ, edge_targ;
    g.get_extra("prep", node_targ, edge_targ);

    node_error += matrix_difference(node_targ,node_pred);
    edge_error += matrix_difference(edge_targ,edge_pred);
    break;

  }

  printf("--- Node Error: %f\n", node_error);
  printf("--- Edge Error: %f\n", edge_error);


  printf("Finished Testing %s\n", test);

}

void test_Golden()
{
  char test[] = "TorchUtils::GeoModel";
  printf("Testing %s...\n", test);

  dataset.load_extra("golden");
  printf("Processesing %i Graphs...\n", (int)dataset.size());
  float node_error, edge_error;
  node_error = edge_error = 0;
  for (Graph g : dataset)
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
  test_Scale();
  test_Golden();
}
