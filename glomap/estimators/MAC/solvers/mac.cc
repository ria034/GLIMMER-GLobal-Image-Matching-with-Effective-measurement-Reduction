#include "mac.h"
#include <glomap/math/tree.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <cmath>

namespace glomap {

inline uint64_t GetMacEdgeKey(image_t u, image_t v) {
    image_t min_node = std::min(u, v);
    image_t max_node = std::max(u, v);
    return (static_cast<uint64_t>(min_node) << 32) | static_cast<uint64_t>(max_node);
}

void RunMacViewGraphSparsification(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    double retention_ratio) 
{
    // 1. Filter and map registered images to dense contiguous indices [0, N-1]
    std::unordered_map<image_t, int> image_to_idx;
    std::vector<image_t> idx_to_image;
    
    for (const auto& [image_id, image] : images) {
        if (!image.IsRegistered()) continue;
        image_to_idx[image_id] = static_cast<int>(idx_to_image.size());
        idx_to_image.push_back(image_id);
    }

    int num_nodes = static_cast<int>(idx_to_image.size());
    if (num_nodes <= 1) return;

    // 2. Build Laplacian Matrix using Inlier Numbers as Conductance Weights
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(num_nodes, num_nodes);
    
    struct EdgeInfo {
        int u_idx;
        int v_idx;
        double eff_resistance;
        image_t u_orig;
        image_t v_orig;
        image_pair_t pair_id;
    };

    struct RawEdge {
        int u_idx;
        int v_idx;
        double inlier_weight;
        image_t u_orig;
        image_t v_orig;
        image_pair_t pair_id;
    };
    std::vector<RawEdge> valid_edges;

    for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
        if (!image_pair.is_valid) continue;

        image_t u = image_pair.image_id1;
        image_t v = image_pair.image_id2;

        if (images.at(u).IsRegistered() == false || images.at(v).IsRegistered() == false) continue;
        if (image_to_idx.find(u) == image_to_idx.end() || image_to_idx.find(v) == image_to_idx.end()) continue;

        int u_idx = image_to_idx[u];
        int v_idx = image_to_idx[v];
        
        double inlier_weight = static_cast<double>(image_pair.inliers.size());
        if (inlier_weight <= 0.0) inlier_weight = 1.0; 

        L(u_idx, u_idx) += inlier_weight;
        L(v_idx, v_idx) += inlier_weight;
        L(u_idx, v_idx) -= inlier_weight;
        L(v_idx, u_idx) -= inlier_weight;

        valid_edges.push_back({u_idx, v_idx, inlier_weight, u, v, pair_id});
    }

    if (valid_edges.empty()) return;

    // 3. Compute Moore-Penrose Pseudo-Inverse of Laplacian (L†) for Effective Resistance
    Eigen::BDCSVD<Eigen::MatrixXd> svd(L, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd inv_singular_values = Eigen::VectorXd::Zero(singular_values.size());
    
    double tol = 1e-6 * std::max(num_nodes, 1) * singular_values(0);
    for (int i = 0; i < singular_values.size(); ++i) {
        if (singular_values(i) > tol) {
            inv_singular_values(i) = 1.0 / singular_values(i);
        }
    }
    Eigen::MatrixXd L_pinv = svd.matrixV() * inv_singular_values.asDiagonal() * svd.matrixU().transpose();

    // 4. Build Protected Edge Set (Odometric Chain + Maximum Spanning Tree)
    std::unordered_set<uint64_t> protected_edge_keys;

    // A. Odometric Backbone
    std::vector<image_t> sorted_registered_images = idx_to_image;
    std::sort(sorted_registered_images.begin(), sorted_registered_images.end());

    for (size_t i = 0; i < sorted_registered_images.size() - 1; ++i) {
        image_t u = sorted_registered_images[i];
        image_t v = sorted_registered_images[i + 1];
        image_pair_t pair_id = colmap::ImagePairToPairId(u, v);
        if (view_graph.image_pairs.find(pair_id) != view_graph.image_pairs.end() &&
            view_graph.image_pairs[pair_id].is_valid) {
            protected_edge_keys.insert(GetMacEdgeKey(u, v));
        }
    }

    // B. Maximum Spanning Tree Backbone using INLIER_NUM
    std::unordered_map<image_t, image_t> parents;
    MaximumSpanningTree(view_graph, images, parents, WeightType::INLIER_NUM);
    for (const auto& [child, parent] : parents) {
        if (child != parent) {
            protected_edge_keys.insert(GetMacEdgeKey(child, parent));
        }
    }

    // 5. Separate and Score Candidate Edges
    std::vector<EdgeInfo> candidate_edges;

    for (const auto& edge_info : valid_edges) {
        double eff_resistance = L_pinv(edge_info.u_idx, edge_info.u_idx) +
                                L_pinv(edge_info.v_idx, edge_info.v_idx) -
                                2.0 * L_pinv(edge_info.u_idx, edge_info.v_idx);

        eff_resistance = std::max(0.0, eff_resistance);
        uint64_t key = GetMacEdgeKey(edge_info.u_orig, edge_info.v_orig);

        if (protected_edge_keys.find(key) == protected_edge_keys.end()) {
            candidate_edges.push_back({edge_info.u_idx, edge_info.v_idx, eff_resistance, edge_info.u_orig, edge_info.v_orig, edge_info.pair_id});
        }
    }

    if (candidate_edges.empty()) return;

    // 6. Greedy Selection based on Effective Resistance
    int k_budget = static_cast<int>(candidate_edges.size() * retention_ratio);
    k_budget = std::clamp(k_budget, 0, static_cast<int>(candidate_edges.size()));

    std::sort(candidate_edges.begin(), candidate_edges.end(), [](const auto& a, const auto& b) {
        return a.eff_resistance > b.eff_resistance;
    });

    // 7. Apply Pruning Mask
    for (size_t i = 0; i < candidate_edges.size(); ++i) {
        if (i >= static_cast<size_t>(k_budget)) {
            view_graph.image_pairs[candidate_edges[i].pair_id].is_valid = false;
        }
    }
}

} // namespace glomap