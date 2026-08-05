// glim_kitti : standalone (non-ROS) driver that feeds KITTI odometry velodyne
// scans into GLIM v1.0.0's C++ API.
//
// GLIM v1.0.0's plain-CMake build installs ONLY shared libraries (libglim.so +
// the dlopen-able module .so files) -- it contains no add_executable() and no
// main(). glim_rosnode / glim_rosbag / offline_viewer live in the separate
// koide3/glim_ros1 (or glim_ros2) package. This file replaces that frontend
// with a file-based one, mirroring GlimROS::insert_frame() / ::wait() from
// glim_ros1/src/glim_ros/glim_ros.cpp.
//
// Usage:
//   glim_kitti <config_dir> <kitti_sequence_dir> <dump_dir>
// e.g.
//   glim_kitti /config /data/sequences/04 /output/dump

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <cmath>
#include <cstdlib>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <glim/util/config.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/raw_points.hpp>
#include <glim/util/time_keeper.hpp>
#include <glim/preprocess/cloud_preprocessor.hpp>
#include <glim/odometry/odometry_estimation_base.hpp>
#include <glim/odometry/async_odometry_estimation.hpp>
#include <glim/mapping/sub_mapping_base.hpp>
#include <glim/mapping/async_sub_mapping.hpp>
#include <glim/mapping/global_mapping_base.hpp>
#include <glim/mapping/async_global_mapping.hpp>

namespace {

// Per-point timestamp policy.
//   AZIMUTH : derive t from the yaw angle of each point (correct for KITTI)
//   ORDER   : leave RawPoints::times empty so that glim::TimeKeeper synthesizes
//             pseudo timestamps from the point *index* order. This is WRONG for
//             KITTI: the .bin files are ring-major (all 360 deg of laser 0, then
//             laser 1, ...), verified by counting 64 azimuth wraps and a
//             monotonically decreasing elevation over the point index.
//   ZERO    : all per-point times = 0 (treat the scan as a global shutter);
//             equivalent to "global_shutter_lidar": true in config_sensors.json.
enum class TimeMode { AZIMUTH, ORDER, ZERO };

// KITTI velodyne_points/data/*.bin : float32 x,y,z,intensity per point
glim::RawPoints::Ptr read_kitti_bin(const std::string& path, double stamp, TimeMode time_mode, double scan_duration) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    return nullptr;
  }
  const std::streamsize bytes = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  const size_t num_floats = static_cast<size_t>(bytes) / sizeof(float);
  std::vector<float> buffer(num_floats);
  ifs.read(reinterpret_cast<char*>(buffer.data()), bytes);

  const size_t num_points = num_floats / 4;
  auto raw = std::make_shared<glim::RawPoints>();
  raw->stamp = stamp;
  raw->points.resize(num_points);
  raw->intensities.resize(num_points);
  for (size_t i = 0; i < num_points; i++) {
    raw->points[i] << buffer[4 * i + 0], buffer[4 * i + 1], buffer[4 * i + 2], 1.0;
    raw->intensities[i] = buffer[4 * i + 3];
  }
  switch (time_mode) {
    case TimeMode::ORDER:
      // leave raw->times empty -> glim::TimeKeeper::replace_points_stamp() fills in
      // pseudo timestamps from the point index order.
      break;
    case TimeMode::ZERO:
      raw->times.assign(num_points, 0.0);
      break;
    case TimeMode::AZIMUTH:
      // KISS-ICP's KITTI convention: yaw = -atan2(y, x) in [-pi, pi] mapped to
      // [0, 1) over one revolution, then scaled by the scan duration.
      raw->times.resize(num_points);
      for (size_t i = 0; i < num_points; i++) {
        const double yaw = -std::atan2(raw->points[i].y(), raw->points[i].x());
        raw->times[i] = scan_duration * 0.5 * (yaw / M_PI + 1.0);
      }
      break;
  }
  return raw;
}

std::vector<double> read_times(const std::string& path, size_t num_scans) {
  std::vector<double> stamps;
  std::ifstream ifs(path);
  if (ifs) {
    double t;
    while (ifs >> t) {
      stamps.push_back(t);
    }
  }
  if (stamps.size() != num_scans) {
    spdlog::warn("times.txt has {} entries but {} scans found; falling back to 10 Hz", stamps.size(), num_scans);
    stamps.resize(num_scans);
    for (size_t i = 0; i < num_scans; i++) {
      stamps[i] = 0.1 * static_cast<double>(i);
    }
  }
  return stamps;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: glim_kitti <config_dir> <kitti_sequence_dir> <dump_dir> [max_scans] [stamp_offset]" << std::endl;
    return 1;
  }

  const std::string config_path = argv[1];
  const std::string seq_dir = argv[2];
  const std::string dump_path = argv[3];
  const int max_scans = (argc > 4) ? std::stoi(argv[4]) : -1;
  // GLIM timestamps are absolute seconds. KITTI times.txt starts at 0.0; a
  // positive offset keeps every stamp strictly > 0 which is what the rest of
  // the pipeline expects.
  const double stamp_offset = (argc > 5) ? std::stod(argv[5]) : 1000.0;

  // GLIM_KITTI_TIME_MODE = azimuth (default) | order | zero
  TimeMode time_mode = TimeMode::AZIMUTH;
  std::string time_mode_name = "azimuth";
  if (const char* env = std::getenv("GLIM_KITTI_TIME_MODE")) {
    time_mode_name = env;
    if (time_mode_name == "order") {
      time_mode = TimeMode::ORDER;
    } else if (time_mode_name == "zero") {
      time_mode = TimeMode::ZERO;
    } else if (time_mode_name != "azimuth") {
      std::cerr << "unknown GLIM_KITTI_TIME_MODE=" << time_mode_name << " (azimuth|order|zero)" << std::endl;
      return 1;
    }
  }

  auto logger = spdlog::stdout_color_mt("glim_kitti");
  spdlog::set_default_logger(logger);
  spdlog::set_pattern("[%H:%M:%S.%e] [%n] [%^%l%$] %v");
  glim::set_default_logger(logger);

  // ---- config -------------------------------------------------------------
  spdlog::info("config_path: {}", config_path);
  glim::GlobalConfig::instance(config_path);

  glim::TimeKeeper time_keeper;
  glim::CloudPreprocessor preprocessor;

  // ---- odometry -----------------------------------------------------------
  glim::Config config_odometry(glim::GlobalConfig::get_config_path("config_odometry"));
  const std::string odom_so = config_odometry.param<std::string>("odometry_estimation", "so_name", "libodometry_estimation_ct.so");
  spdlog::info("load {}", odom_so);
  auto odom = glim::OdometryEstimationBase::load_module(odom_so);
  if (!odom) {
    spdlog::critical("failed to load odometry estimation module {}", odom_so);
    return 1;
  }
  spdlog::info("odometry requires_imu={}", odom->requires_imu());
  if (odom->requires_imu()) {
    spdlog::critical("this odometry module needs IMU data, which KITTI velodyne-only has none of.");
    spdlog::critical("set \"config_odometry\": \"config_odometry_ct.json\" in config.json");
    return 1;
  }
  auto odometry_estimation = std::make_shared<glim::AsyncOdometryEstimation>(odom, odom->requires_imu());

  // ---- sub mapping --------------------------------------------------------
  std::shared_ptr<glim::AsyncSubMapping> sub_mapping;
  {
    const std::string so = glim::Config(glim::GlobalConfig::get_config_path("config_sub_mapping")).param<std::string>("sub_mapping", "so_name", "libsub_mapping.so");
    spdlog::info("load {}", so);
    auto sub = glim::SubMappingBase::load_module(so);
    if (!sub) {
      spdlog::critical("failed to load sub mapping module {}", so);
      return 1;
    }
    sub_mapping = std::make_shared<glim::AsyncSubMapping>(sub);
  }

  // ---- global mapping -----------------------------------------------------
  std::shared_ptr<glim::AsyncGlobalMapping> global_mapping;
  {
    const std::string so =
      glim::Config(glim::GlobalConfig::get_config_path("config_global_mapping")).param<std::string>("global_mapping", "so_name", "libglobal_mapping.so");
    spdlog::info("load {}", so);
    auto global = glim::GlobalMappingBase::load_module(so);
    if (!global) {
      spdlog::critical("failed to load global mapping module {}", so);
      return 1;
    }
    global_mapping = std::make_shared<glim::AsyncGlobalMapping>(global);
  }

  // ---- extension modules (viewers) ----------------------------------------
  // GLIM's viewers are ordinary extension modules: loading one is all it takes,
  // because it subscribes to glim's callback slots itself and renders from its
  // own thread. Load nothing and no OpenGL context is ever created, which is
  // what keeps the default run headless.
  //
  // Sources, in order of precedence:
  //   GLIM_KITTI_VIEWER=1                 -> force libstandard_viewer.so
  //   config_ros.json glim_ros/extension_modules -> upstream's own convention
  std::vector<std::shared_ptr<glim::ExtensionModule>> extensions;
  {
    std::vector<std::string> module_names;
    const char* viewer_env = std::getenv("GLIM_KITTI_VIEWER");
    if (viewer_env && std::string(viewer_env) != "0") {
      module_names.push_back("libstandard_viewer.so");
    } else {
      module_names = glim::Config(glim::GlobalConfig::get_config_path("config_ros"))
                       .param<std::vector<std::string>>("glim_ros", "extension_modules", std::vector<std::string>{});
    }
    for (const auto& name : module_names) {
      spdlog::info("load extension module {}", name);
      auto ext = glim::ExtensionModule::load_module(name);
      if (!ext) {
        // Not fatal: a missing viewer should not throw away a mapping run.
        spdlog::warn("failed to load extension module {} (continuing headless)", name);
        continue;
      }
      extensions.push_back(ext);
    }
    if (!extensions.empty()) {
      spdlog::info("{} extension module(s) active -- close the viewer window to abort early", extensions.size());
    }
  }

  // ---- enumerate scans ----------------------------------------------------
  const std::filesystem::path velodyne_dir = std::filesystem::path(seq_dir) / "velodyne";
  std::vector<std::string> scan_files;
  for (const auto& e : std::filesystem::directory_iterator(velodyne_dir)) {
    if (e.path().extension() == ".bin") {
      scan_files.push_back(e.path().string());
    }
  }
  std::sort(scan_files.begin(), scan_files.end());
  if (scan_files.empty()) {
    spdlog::critical("no .bin scans found in {}", velodyne_dir.string());
    return 1;
  }
  const auto stamps = read_times((std::filesystem::path(seq_dir) / "times.txt").string(), scan_files.size());
  const size_t num_scans = (max_scans > 0) ? std::min<size_t>(max_scans, scan_files.size()) : scan_files.size();
  // median inter-scan interval == one Velodyne revolution
  const double scan_duration = (stamps.size() > 1) ? (stamps.back() - stamps.front()) / (stamps.size() - 1) : 0.1;
  spdlog::info("{} scans in {} (feeding {})", scan_files.size(), velodyne_dir.string(), num_scans);
  spdlog::info("per-point time mode = {}, scan_duration = {:.6f} s", time_mode_name, scan_duration);

  // ---- background result draining thread (mirrors GlimROS::loop) -----------
  std::atomic_bool kill_switch{false};
  std::thread drain([&] {
    while (!kill_switch) {
      std::vector<glim::EstimationFrame::ConstPtr> results, marginalized;
      odometry_estimation->get_results(results, marginalized);
      if (results.empty() && marginalized.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      for (const auto& f : marginalized) {
        sub_mapping->insert_frame(f);
      }
      for (const auto& submap : sub_mapping->get_results()) {
        global_mapping->insert_submap(submap);
      }
    }
  });

  // ---- feed ---------------------------------------------------------------
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_scans; i++) {
    auto raw = read_kitti_bin(scan_files[i], stamps[i] + stamp_offset, time_mode, scan_duration);
    if (!raw) {
      spdlog::warn("failed to read {}", scan_files[i]);
      continue;
    }

    time_keeper.process(raw);
    auto preprocessed = preprocessor.preprocess(raw);
    if (!preprocessed || preprocessed->size() < 100) {
      spdlog::warn("skipping frame {} with too few points", i);
      continue;
    }

    while (odometry_estimation->workload() > 10) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    odometry_estimation->insert_frame(preprocessed);

    // A viewer that has been closed reports !ok(); stop feeding so the user can
    // abort a long run from the GUI. needs_wait() lets a module that has fallen
    // behind (the viewer, when rendering a big map) throttle the feed.
    bool aborted = false;
    for (const auto& ext : extensions) {
      while (ext->needs_wait()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (!ext->ok()) {
        aborted = true;
      }
    }
    if (aborted) {
      spdlog::warn("viewer closed at scan {}/{} -- stopping feed and saving what we have", i, num_scans);
      break;
    }

    if (i % 25 == 0) {
      spdlog::info("fed scan {}/{} stamp={:.6f} raw_pts={} pre_pts={}", i, num_scans, raw->stamp, raw->size(), preprocessed->size());
    }
  }
  spdlog::info("all scans fed, draining pipeline");

  // ---- flush (mirrors GlimROS::wait) --------------------------------------
  odometry_estimation->join();
  {
    std::vector<glim::EstimationFrame::ConstPtr> results, marginalized;
    odometry_estimation->get_results(results, marginalized);
    for (const auto& f : marginalized) {
      sub_mapping->insert_frame(f);
    }
  }
  sub_mapping->join();
  for (const auto& submap : sub_mapping->get_results()) {
    global_mapping->insert_submap(submap);
  }
  global_mapping->join();

  kill_switch = true;
  drain.join();

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double wall = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
  spdlog::info("mapping done in {:.3f} s wall ({:.2f} scans/s)", wall, num_scans / wall);

  std::filesystem::create_directories(dump_path);
  spdlog::info("saving dump to {}", dump_path);
  global_mapping->save(dump_path);
  spdlog::info("saved");

  // With a viewer up, exiting immediately would slam the window shut the instant
  // mapping finished, which is useless for actually looking at the result. Hold
  // it open until the user closes it. Headless runs skip this entirely, so CI
  // behaviour is unchanged.
  if (!extensions.empty()) {
    spdlog::info("mapping finished -- close the viewer window to exit");
    while (std::all_of(extensions.begin(), extensions.end(), [](const auto& ext) { return ext->ok(); })) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  return 0;
}
