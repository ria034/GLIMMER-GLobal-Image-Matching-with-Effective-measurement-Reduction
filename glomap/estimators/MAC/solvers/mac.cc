#include "glomap/estimators/MAC/solvers/mac.h"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ios>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace glomap {
namespace {

// ===========================================================================
// Basic building blocks
// ===========================================================================
struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit UnionFind(int n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int Find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool Union(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return false;
    if (rank[a] < rank[b]) std::swap(a, b);
    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];
    return true;
  }
};

struct MacEdge {
  image_pair_t pair_id;
  int u;          // local node index
  int v;          // local node index
  double weight;  // inlier count
};

struct MacGraph {
  int num_nodes = 0;
  std::vector<image_t> node_to_image;  // local index -> image id
  std::vector<MacEdge> edges;
  int num_outside = 0;  // valid edges outside the largest component
};

// The graph is built after glomap's relative pose estimation, so `inliers`
// holds glomap's own inlier count. The match count is only a fallback.
double EdgeWeight(const ImagePair& pair) {
  if (!pair.inliers.empty()) return static_cast<double>(pair.inliers.size());
  if (pair.matches.rows() > 0) return static_cast<double>(pair.matches.rows());
  return 1.0;
}

inline uint64_t EdgeKey(int a, int b) {
  if (a > b) std::swap(a, b);
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(b));
}

inline void AddToLaplacian(Eigen::MatrixXd& L, int u, int v, double w) {
  L(u, u) += w;
  L(v, v) += w;
  L(u, v) -= w;
  L(v, u) -= w;
}

// lambda_2 of L and (optionally) its Fiedler vector. Dense O(N^3).
double ComputeFiedler(const Eigen::MatrixXd& L, Eigen::VectorXd* fiedler) {
  if (L.rows() < 2) return 0.0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L);
  if (es.info() != Eigen::Success) {
    LOG(WARNING) << "[MAC] Eigen decomposition failed";
    if (fiedler) *fiedler = Eigen::VectorXd::Zero(L.rows());
    return 0.0;
  }
  if (fiedler) *fiedler = es.eigenvectors().col(1);
  return std::max(0.0, es.eigenvalues()(1));
}

// Indices of the k largest entries (ties broken by index -> deterministic).
std::vector<int> TopK(const Eigen::VectorXd& values, int k) {
  std::vector<int> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int a, int b) { return values(a) > values(b); });
  idx.resize(std::min<size_t>(k, idx.size()));
  return idx;
}

Eigen::MatrixXd WeightedLaplacian(const Eigen::MatrixXd& L_base,
                                  const std::vector<MacEdge>& candidates,
                                  const Eigen::VectorXd& w) {
  Eigen::MatrixXd L = L_base;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (w(i) <= 0.0) continue;
    AddToLaplacian(
        L, candidates[i].u, candidates[i].v, w(i) * candidates[i].weight);
  }
  return L;
}

// Order-independent FNV-1a fingerprint of a set of pair ids.
uint64_t Fingerprint(std::vector<image_pair_t> ids) {
  std::sort(ids.begin(), ids.end());
  uint64_t h = 1469598103934665603ULL;
  for (image_pair_t id : ids) {
    const uint64_t v = static_cast<uint64_t>(id);
    for (int b = 0; b < 8; ++b) {
      h ^= (v >> (8 * b)) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

// ===========================================================================
// Step A: graph
// Valid edges of the view graph, restricted to the largest connected
// component (lambda_2 is 0 on a disconnected graph).
// ===========================================================================
MacGraph BuildGraph(const ViewGraph& view_graph,
                    const std::unordered_map<image_t, Image>& images) {
  // Sorted pair ids -> the same graph always gets the same node indices.
  std::vector<image_pair_t> pair_ids;
  for (const auto& [pair_id, pair] : view_graph.image_pairs) {
    if (!pair.is_valid || pair.image_id1 == pair.image_id2) continue;
    if (images.find(pair.image_id1) == images.end() ||
        images.find(pair.image_id2) == images.end()) {
      continue;
    }
    pair_ids.push_back(pair_id);
  }
  std::sort(pair_ids.begin(), pair_ids.end());

  std::unordered_map<image_t, int> image_to_idx;
  std::vector<image_t> idx_to_image;
  auto get_idx = [&](image_t id) {
    auto it = image_to_idx.find(id);
    if (it != image_to_idx.end()) return it->second;
    const int idx = static_cast<int>(idx_to_image.size());
    image_to_idx.emplace(id, idx);
    idx_to_image.push_back(id);
    return idx;
  };

  std::vector<MacEdge> all_edges;
  for (image_pair_t pair_id : pair_ids) {
    const ImagePair& pair = view_graph.image_pairs.at(pair_id);
    all_edges.push_back({pair_id,
                         get_idx(pair.image_id1),
                         get_idx(pair.image_id2),
                         EdgeWeight(pair)});
  }

  MacGraph graph;
  const int n_all = static_cast<int>(idx_to_image.size());
  if (n_all == 0) return graph;

  // Largest connected component (smallest root wins ties).
  UnionFind uf(n_all);
  for (const auto& e : all_edges) uf.Union(e.u, e.v);
  std::vector<int> comp_size(n_all, 0);
  for (int i = 0; i < n_all; ++i) ++comp_size[uf.Find(i)];
  const int best_root = static_cast<int>(
      std::max_element(comp_size.begin(), comp_size.end()) -
      comp_size.begin());

  std::vector<int> to_local(n_all, -1);
  for (int i = 0; i < n_all; ++i) {
    if (uf.Find(i) == best_root) {
      to_local[i] = static_cast<int>(graph.node_to_image.size());
      graph.node_to_image.push_back(idx_to_image[i]);
    }
  }
  graph.num_nodes = static_cast<int>(graph.node_to_image.size());

  for (const auto& e : all_edges) {
    const int lu = to_local[e.u];
    const int lv = to_local[e.v];
    if (lu < 0 || lv < 0) {
      ++graph.num_outside;
      continue;
    }
    graph.edges.push_back({e.pair_id, lu, lv, e.weight});
  }
  return graph;
}

// ===========================================================================
// Step B: backbone
// Protected edges that are never pruned. Depends only on the graph, never on
// the budget, so it is identical for every retention ratio.
//
// HEURISTIC HOOK: to try another backbone (e.g. a tree ordered by effective
// resistance, or a minimum degree per image), change this function only.
// ===========================================================================
std::vector<char> BuildBackbone(const MacGraph& graph,
                                const MacOptions& options,
                                MacStats& stats) {
  const int n = graph.num_nodes;
  const auto& edges = graph.edges;
  std::vector<char> is_backbone(edges.size(), 0);

  // B1. Maximum spanning tree (Kruskal, most inliers first). Always spans the
  //     component: exactly n - 1 edges.
  std::vector<int> order(edges.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return edges[a].weight > edges[b].weight;
  });
  UnionFind uf(n);
  for (int idx : order) {
    if (uf.Union(edges[idx].u, edges[idx].v)) {
      is_backbone[idx] = 1;
      ++stats.num_tree_edges;
    }
  }

  // B2. Optional odometry chain: consecutive image ids, if that pair exists.
  if (options.use_odometry_backbone) {
    std::unordered_map<uint64_t, int> key_to_edge;
    for (int i = 0; i < static_cast<int>(edges.size()); ++i) {
      key_to_edge[EdgeKey(edges[i].u, edges[i].v)] = i;
    }
    std::vector<int> nodes(n);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
      return graph.node_to_image[a] < graph.node_to_image[b];
    });
    for (int i = 0; i + 1 < n; ++i) {
      auto it = key_to_edge.find(EdgeKey(nodes[i], nodes[i + 1]));
      if (it == key_to_edge.end() || is_backbone[it->second]) continue;
      is_backbone[it->second] = 1;
      ++stats.num_odometry_edges;
    }
  }
  return is_backbone;
}

// ===========================================================================
// Step C: MAC
// maximize lambda_2(L_base + sum_i w_i * L_i)
//   s.t. w in [0,1]^m, sum(w) = K
// Frank-Wolfe on the relaxation, then keep the K largest weights.
//
// HEURISTIC HOOK: to try another selection rule (e.g. effective resistance),
// write a function with this signature and call it in Step C below.
// ===========================================================================
std::vector<int> SelectEdgesMac(const Eigen::MatrixXd& L_base,
                                const std::vector<MacEdge>& candidates,
                                int K,
                                const MacOptions& options,
                                MacStats& stats) {
  const int m = static_cast<int>(candidates.size());
  if (K <= 0) return {};
  if (K >= m) {
    std::vector<int> all(m);
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  Eigen::VectorXd w = Eigen::VectorXd::Constant(m, static_cast<double>(K) / m);
  Eigen::VectorXd grad(m);
  Eigen::VectorXd fiedler;

  int it = 0;
  for (; it < options.max_iterations; ++it) {
    const double lambda2 =
        ComputeFiedler(WeightedLaplacian(L_base, candidates, w), &fiedler);

    // Gradient of lambda_2 w.r.t. w_i: weight_i * (f_u - f_v)^2
    for (int i = 0; i < m; ++i) {
      const double d = fiedler(candidates[i].u) - fiedler(candidates[i].v);
      grad(i) = candidates[i].weight * d * d;
    }

    // Best vertex of the feasible set: all K units on the largest gradients.
    Eigen::VectorXd s = Eigen::VectorXd::Zero(m);
    for (int idx : TopK(grad, K)) s(idx) = 1.0;

    // Stop when the duality gap is small.
    const double gap = grad.dot(s - w);
    if (gap <= options.gap_tolerance * std::max(1.0, lambda2)) break;

    w += (2.0 / (it + 2.0)) * (s - w);
  }

  stats.fw_iterations = it;
  stats.lambda2_relaxed =
      ComputeFiedler(WeightedLaplacian(L_base, candidates, w), nullptr);
  return TopK(w, K);  // rounding
}

}  // namespace

// ===========================================================================
// Entry point: graph -> backbone -> MAC -> apply
// ===========================================================================
MacStats RunMacViewGraphSparsification(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const MacOptions& options) {
  MacStats stats;

  // ---- Step A: graph ----
  const MacGraph graph = BuildGraph(view_graph, images);
  const int n = graph.num_nodes;
  stats.num_nodes = n;
  stats.num_valid_edges = static_cast<int>(graph.edges.size());
  stats.num_outside = graph.num_outside;
  if (n < 3 || graph.edges.empty()) {
    LOG(INFO) << "[MAC] Graph too small for sparsification, skipping";
    return stats;
  }
  if (n > 5000) {
    LOG(WARNING) << "[MAC] " << n << " nodes: dense eigen-decomposition will "
                 << "be slow.";
  }

  // ---- Step B: backbone ----
  const std::vector<char> is_backbone = BuildBackbone(graph, options, stats);

  std::vector<image_pair_t> graph_ids, backbone_ids;
  Eigen::MatrixXd L_base = Eigen::MatrixXd::Zero(n, n);
  std::vector<MacEdge> candidates;
  for (size_t i = 0; i < graph.edges.size(); ++i) {
    const MacEdge& e = graph.edges[i];
    graph_ids.push_back(e.pair_id);
    if (is_backbone[i]) {
      AddToLaplacian(L_base, e.u, e.v, e.weight);
      backbone_ids.push_back(e.pair_id);
    } else {
      candidates.push_back(e);
    }
  }
  stats.num_backbone_edges = static_cast<int>(backbone_ids.size());
  stats.num_candidates = static_cast<int>(candidates.size());
  stats.graph_hash = Fingerprint(graph_ids);
  stats.backbone_hash = Fingerprint(backbone_ids);
  stats.lambda2_backbone = ComputeFiedler(L_base, nullptr);

  // ---- Step C: MAC ----
  const double ratio = std::clamp(options.retention_ratio, 0.0, 1.0);
  const int K = std::clamp(
      static_cast<int>(std::lround(ratio * candidates.size())),
      0,
      static_cast<int>(candidates.size()));

  const std::vector<int> selected =
      SelectEdgesMac(L_base, candidates, K, options, stats);

  // ---- Apply: switch off every candidate that was not selected ----
  std::vector<char> keep(candidates.size(), 0);
  Eigen::MatrixXd L_final = L_base;
  for (int idx : selected) {
    keep[idx] = 1;
    AddToLaplacian(
        L_final, candidates[idx].u, candidates[idx].v, candidates[idx].weight);
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!keep[i]) {
      view_graph.image_pairs.at(candidates[i].pair_id).is_valid = false;
      ++stats.num_pruned;
    }
  }
  stats.num_selected = static_cast<int>(selected.size());
  stats.lambda2_final = ComputeFiedler(L_final, nullptr);
  if (K == 0 || K == stats.num_candidates) {
    stats.lambda2_relaxed = stats.lambda2_final;  // no optimization needed
  }

  // ---- Log ----
  const int kept = stats.num_backbone_edges + stats.num_selected;
  LOG(INFO)
      << "\n[MAC] ========================================"
      << "\n[MAC] Graph"
      << "\n[MAC]   Images:               " << stats.num_nodes
      << "\n[MAC]   Valid edges:          " << stats.num_valid_edges
      << "\n[MAC]   Outside largest CC:   " << stats.num_outside
      << " (untouched)"
      << "\n[MAC]   Fingerprint:          " << std::hex << stats.graph_hash
      << std::dec
      << "\n[MAC] Backbone"
      << "\n[MAC]   Spanning tree:        " << stats.num_tree_edges
      << "\n[MAC]   Odometry (extra):     " << stats.num_odometry_edges
      << "\n[MAC]   Total:                " << stats.num_backbone_edges
      << "\n[MAC]   Fingerprint:          " << std::hex << stats.backbone_hash
      << std::dec
      << "\n[MAC] MAC (retention ratio " << ratio << ")"
      << "\n[MAC]   Candidates kept:      " << stats.num_selected << " / "
      << stats.num_candidates
      << "\n[MAC]   Frank-Wolfe iters:    " << stats.fw_iterations
      << "\n[MAC]   lambda2 backbone:     " << stats.lambda2_backbone
      << "\n[MAC]   lambda2 relaxed:      " << stats.lambda2_relaxed
      << "\n[MAC]   lambda2 final:        " << stats.lambda2_final
      << "\n[MAC] Result"
      << "\n[MAC]   Edges kept:           " << kept << " / "
      << stats.num_valid_edges << " ("
      << 100.0 * kept / std::max(1, stats.num_valid_edges) << "%)"
      << "\n[MAC]   Edges pruned:         " << stats.num_pruned
      << "\n[MAC] ========================================";

  return stats;
}

}  // namespace glomap