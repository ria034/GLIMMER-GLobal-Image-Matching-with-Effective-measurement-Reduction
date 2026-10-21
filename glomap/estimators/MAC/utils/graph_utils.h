#pragma once
#include <vector>
#include <utility>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace glomap {

struct Edge {
    int i;
    int j;
    double weight;
};

Eigen::SparseMatrix<double> WeightGraphLapFromEdgeList(
    const std::vector<Edge>& edges, int num_nodes);

Eigen::SparseMatrix<double> WeightGraphLapFromEdges(
    const std::vector<std::pair<int, int>>& edges, 
    const std::vector<double>& weights, 
    int num_nodes);

} // namespace glomap