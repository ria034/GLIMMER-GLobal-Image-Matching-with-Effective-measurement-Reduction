#pragma once
#include <Eigen/Dense>
#include "rounding.h"

namespace glomap {

inline Eigen::VectorXd SolveSubsetBoxLP(const Eigen::VectorXd& g, int k, const std::vector<double>& weights = {}) {
    return RoundNearest(g, k, weights);
}

} // namespace glomap