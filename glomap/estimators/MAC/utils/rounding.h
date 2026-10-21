#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

namespace glomap {

inline Eigen::VectorXd RoundNearest(const Eigen::VectorXd& w, int k, const std::vector<double>& weights = {}) {
    int n = w.size();
    Eigen::VectorXd rounded = Eigen::VectorXd::Zero(n);
    if (k <= 0 || n == 0) return rounded;
    if (k >= n) return Eigen::VectorXd::Ones(n);

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int i, int j) {
        if (std::abs(w(i) - w(j)) > 1e-7) {
            return w(i) < w(j);
        }
        if (!weights.empty()) {
            return weights[i] < weights[j];
        }
        return i < j;
    });

    for (int i = n - k; i < n; ++i) {
        rounded(idx[i]) = 1.0;
    }
    return rounded;
}

} // namespace glomap