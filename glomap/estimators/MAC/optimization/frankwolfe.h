#pragma once
#include <Eigen/Dense>
#include <functional>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace glomap {

inline double NaiveStepSize(int iteration) {
    return 2.0 / (static_cast<double>(iteration) + 2.0);
}

template <typename ObjectiveFunc, typename SolveLpFunc>
std::pair<Eigen::VectorXd, double> FrankWolfe(
    const Eigen::VectorXd& initial,
    ObjectiveFunc problem_fn,
    SolveLpFunc solve_lp_fn,
    int max_iters = 50,
    double relative_duality_gap_tol = 1e-4,
    double grad_norm_tol = 1e-8,
    bool verbose = false) 
{
    Eigen::VectorXd x = initial;
    double u = std::numeric_limits<double>::infinity();

    for (int i = 0; i < max_iters; ++i) {
        auto [f, gradf] = problem_fn(x);
        Eigen::VectorXd s = solve_lp_fn(gradf);

        double duality_gap = gradf.dot(s - x);
        u = std::min(u, f + duality_gap);

        if (gradf.norm() < grad_norm_tol) return {x, u};
        if (std::abs(f) > 1e-12 && (u - f) < relative_duality_gap_tol * std::abs(f)) return {x, u};

        double alpha = NaiveStepSize(i);
        x = x + alpha * (s - x);
    }
    return {x, u};
}

} // namespace glomap