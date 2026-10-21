#pragma once
#include <vector>
#include <utility>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "../utils/graph_utils.h"

namespace glomap {

class MacSparsifier {
public:
    MacSparsifier(const std::vector<Edge>& fixed_edges,
                  const std::vector<Edge>& candidate_edges,
                  int num_nodes,
                  double min_selection_weight_tol = 1e-10);

    Eigen::SparseMatrix<double> BuildLaplacian(const Eigen::VectorXd& x) const;
    double EvaluateObjective(const Eigen::VectorXd& x);
    std::pair<double, Eigen::VectorXd> ComputeProblem(const Eigen::VectorXd& x);
    Eigen::VectorXd Solve(int k, const Eigen::VectorXd& x_init, int max_iters = 10, bool verbose = false);

private:
    int num_nodes_;
    Eigen::SparseMatrix<double> L_fixed_;
    std::vector<double> candidate_weights_;
    std::vector<std::pair<int, int>> candidate_edge_list_;
    double min_selection_weight_tol_;
};

} // namespace glomap