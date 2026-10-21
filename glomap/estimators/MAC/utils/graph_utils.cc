#include "graph_utils.h"

namespace glomap {

Eigen::SparseMatrix<double> WeightGraphLapFromEdgeList(
    const std::vector<Edge>& edges, int num_nodes) 
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(edges.size() * 4);

    for (const auto& edge : edges) {
        triplets.emplace_back(edge.i, edge.i, edge.weight);
        triplets.emplace_back(edge.j, edge.j, edge.weight);
        triplets.emplace_back(edge.i, edge.j, -edge.weight);
        triplets.emplace_back(edge.j, edge.i, -edge.weight);
    }

    Eigen::SparseMatrix<double> laplacian(num_nodes, num_nodes);
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}

Eigen::SparseMatrix<double> WeightGraphLapFromEdges(
    const std::vector<std::pair<int, int>>& edges, 
    const std::vector<double>& weights, 
    int num_nodes) 
{
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(edges.size() * 4);

    for (size_t k = 0; k < edges.size(); ++k) {
        int u = edges[k].first;
        int v = edges[k].second;
        double w = weights[k];

        triplets.emplace_back(u, u, w);
        triplets.emplace_back(v, v, w);
        triplets.emplace_back(u, v, -w);
        triplets.emplace_back(v, u, -w);
    }

    Eigen::SparseMatrix<double> laplacian(num_nodes, num_nodes);
    laplacian.setFromTriplets(triplets.begin(), triplets.end());
    return laplacian;
}

} // namespace glomap