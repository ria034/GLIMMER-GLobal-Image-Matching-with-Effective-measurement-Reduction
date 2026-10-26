#pragma once

#include "glomap/scene/image.h"
#include "glomap/scene/view_graph.h"

#include <cstdint>
#include <unordered_map>

namespace glomap {

struct MacOptions {
  // Fraction of the candidate (non-backbone) edges to keep, in [0, 1].
  // Set from the command line with --mac_retention_ratio.
  double retention_ratio = 0.5;

  // Add a chain of consecutive image ids to the backbone (only meaningful
  // for sequential capture, e.g. video).
  bool use_odometry_backbone = true;

  // Frank-Wolfe settings for MAC.
  int max_iterations = 20;
  double gap_tolerance = 1e-6;  // relative duality-gap stopping criterion
};

struct MacStats {
  // Graph (input)
  int num_nodes = 0;          // images in the largest connected component
  int num_valid_edges = 0;    // valid edges in that component
  int num_outside = 0;        // valid edges outside it (left untouched)
  uint64_t graph_hash = 0;    // fingerprint of the valid edge set

  // Backbone
  int num_tree_edges = 0;      // maximum spanning tree
  int num_odometry_edges = 0;  // odometry edges not already in the tree
  int num_backbone_edges = 0;
  uint64_t backbone_hash = 0;  // fingerprint of the backbone edge set

  // MAC
  int num_candidates = 0;
  int num_selected = 0;
  int num_pruned = 0;
  int fw_iterations = 0;
  double lambda2_backbone = 0.0;
  double lambda2_relaxed = 0.0;
  double lambda2_final = 0.0;
};

// 1. Takes the valid edges of the view graph (largest connected component).
// 2. Builds the backbone: maximum spanning tree (+ optional odometry chain).
// 3. Runs MAC: keeps K = retention_ratio x #candidates extra edges that
//    maximize the algebraic connectivity (lambda_2) of backbone + kept edges.
// Unselected candidate edges get is_valid = false.
MacStats RunMacViewGraphSparsification(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const MacOptions& options = MacOptions());

}  // namespace glomap