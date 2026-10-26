#include "glomap/controllers/global_mapper.h"

#include "glomap/controllers/option_manager.h"
#include "glomap/estimators/MAC/solvers/mac.h"
#include "glomap/io/colmap_io.h"
#include "glomap/io/pose_io.h"

#include <colmap/util/file.h>
#include <colmap/util/misc.h>
#include <colmap/util/timer.h>

namespace glomap {

// -------------------------------------
// Mappers starting from COLMAP database
// -------------------------------------
int RunMapper(int argc, char** argv) {
  std::string database_path;
  std::string output_path;

  std::string image_path = "";
  std::string constraint_type = "ONLY_POINTS";
  std::string output_format = "bin";

  // MAC options
  bool mac_enable = true;
  double mac_retention_ratio = 0.5;
  bool mac_use_odometry = true;
  int mac_max_iterations = 20;

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("constraint_type",
                           &constraint_type,
                           "{ONLY_POINTS, ONLY_CAMERAS, "
                           "POINTS_AND_CAMERAS_BALANCED, POINTS_AND_CAMERAS}");
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddDefaultOption("mac_enable", &mac_enable);
  options.AddDefaultOption("mac_retention_ratio", &mac_retention_ratio);
  options.AddDefaultOption("mac_use_odometry", &mac_use_odometry);
  options.AddDefaultOption("mac_max_iterations", &mac_max_iterations);
  options.AddGlobalMapperFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsFile(database_path)) {
    LOG(ERROR) << "`database_path` is not a file";
    return EXIT_FAILURE;
  }

  if (constraint_type == "ONLY_POINTS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_POINTS;
  } else if (constraint_type == "ONLY_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::ONLY_CAMERAS;
  } else if (constraint_type == "POINTS_AND_CAMERAS_BALANCED") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS_BALANCED;
  } else if (constraint_type == "POINTS_AND_CAMERAS") {
    options.mapper->opt_gp.constraint_type =
        GlobalPositionerOptions::POINTS_AND_CAMERAS;
  } else {
    LOG(ERROR) << "Invalid constraint type";
    return EXIT_FAILURE;
  }

  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  if (mac_retention_ratio < 0.0 || mac_retention_ratio > 1.0) {
    LOG(ERROR) << "`mac_retention_ratio` must be in [0, 1]";
    return EXIT_FAILURE;
  }

  // Load the database
  ViewGraph view_graph;
  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;

  auto database = colmap::Database::Open(database_path);
  ConvertDatabaseToGlomap(*database, view_graph, rigs, cameras, frames, images);

  if (view_graph.image_pairs.empty()) {
    LOG(ERROR) << "Can't continue without image pairs";
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Loaded database";

  colmap::Timer run_timer;
  run_timer.Start();

  if (!mac_enable) {
    // Plain glomap (baseline).
    GlobalMapper global_mapper(*options.mapper);
    global_mapper.Solve(
        *database, view_graph, rigs, cameras, frames, images, tracks);
  } else {
    // ------------------------------------------------------------------
    // Step 1: build the graph. Relative poses + glomap's inlier filter on
    //         ALL pairs. Identical for every budget.
    // ------------------------------------------------------------------
    GlobalMapperOptions build_graph = *options.mapper;
    build_graph.skip_rotation_averaging = true;
    build_graph.skip_track_establishment = true;
    build_graph.skip_global_positioning = true;
    build_graph.skip_bundle_adjustment = true;
    build_graph.skip_retriangulation = true;
    build_graph.skip_pruning = true;
    LOG(INFO) << "Step 1: building the graph (relative poses + inlier filter)";
    {
      GlobalMapper mapper(build_graph);
      mapper.Solve(
          *database, view_graph, rigs, cameras, frames, images, tracks);
    }

    // ------------------------------------------------------------------
    // Steps 2-3: backbone + MAC (see mac.cc). Only MAC uses the budget.
    // ------------------------------------------------------------------
    MacOptions mac_options;
    mac_options.retention_ratio = mac_retention_ratio;
    mac_options.use_odometry_backbone = mac_use_odometry;
    mac_options.max_iterations = mac_max_iterations;

    LOG(INFO) << "Steps 2-3: backbone + MAC";
    colmap::Timer mac_timer;
    mac_timer.Start();
    RunMacViewGraphSparsification(view_graph, images, mac_options);
    mac_timer.Pause();
    LOG(INFO) << "MAC done in " << mac_timer.ElapsedSeconds() << " seconds";

    // ------------------------------------------------------------------
    // Step 4: rest of glomap on the sparsified graph.
    // ------------------------------------------------------------------
    GlobalMapperOptions rest = *options.mapper;
    rest.skip_preprocessing = true;
    rest.skip_view_graph_calibration = true;
    rest.skip_relative_pose_estimation = true;
    LOG(INFO) << "Step 4: rotation averaging onwards (sparsified graph)";
    GlobalMapper mapper(rest);
    mapper.Solve(*database, view_graph, rigs, cameras, frames, images, tracks);
  }

  run_timer.Pause();
  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(output_path,
                            rigs,
                            cameras,
                            frames,
                            images,
                            tracks,
                            output_format,
                            image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

// -------------------------------------
// Mappers starting from COLMAP reconstruction
// -------------------------------------
int RunMapperResume(int argc, char** argv) {
  std::string input_path;
  std::string output_path;
  std::string image_path = "";
  std::string output_format = "bin";

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("output_format", &output_format, "{bin, txt}");
  options.AddGlobalMapperResumeFullOptions();

  options.Parse(argc, argv);

  if (!colmap::ExistsDir(input_path)) {
    LOG(ERROR) << "`input_path` is not a directory";
    return EXIT_FAILURE;
  }

  if (output_format != "bin" && output_format != "txt") {
    LOG(ERROR) << "Invalid output format";
    return EXIT_FAILURE;
  }

  // Load the reconstruction
  ViewGraph view_graph;                        // dummy variable
  std::shared_ptr<colmap::Database> database;  // dummy variable

  std::unordered_map<rig_t, Rig> rigs;
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<frame_t, Frame> frames;
  std::unordered_map<image_t, Image> images;
  std::unordered_map<track_t, Track> tracks;
  colmap::Reconstruction reconstruction;
  reconstruction.Read(input_path);
  ConvertColmapToGlomap(reconstruction, rigs, cameras, frames, images, tracks);

  // MAC is not run here: in resume mode the view graph is an empty dummy.

  GlobalMapper global_mapper(*options.mapper);

  colmap::Timer run_timer;
  run_timer.Start();
  global_mapper.Solve(
      *database, view_graph, rigs, cameras, frames, images, tracks);
  run_timer.Pause();

  LOG(INFO) << "Reconstruction done in " << run_timer.ElapsedSeconds()
            << " seconds";

  WriteGlomapReconstruction(output_path,
                            rigs,
                            cameras,
                            frames,
                            images,
                            tracks,
                            output_format,
                            image_path);
  LOG(INFO) << "Export to COLMAP reconstruction done";

  return EXIT_SUCCESS;
}

}  // namespace glomap