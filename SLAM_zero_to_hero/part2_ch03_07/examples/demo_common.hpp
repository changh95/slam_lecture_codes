/**
 * Shared helpers for the part2_ch03_07 demos:
 *  - KITTI velodyne scan loading and ground-truth pose handling
 *  - one CLI + data-loading path shared by all four pairwise demos
 *  - optional live streaming to a rerun viewer on the host
 *
 * Every demo in this chapter runs on KITTI odometry data. That buys a real
 * ground truth: KITTI ships per-frame poses, so the relative transform between
 * two scans is known and each method can be scored against it instead of only
 * reporting its own fitness score. KITTI poses are given in the left-camera
 * frame, so they are mapped into the velodyne frame with the calib Tr matrix
 * before use (see posesToLidarFrame).
 *
 * The rerun parts are built when CMake finds the rerun C++ SDK and defines
 * HAVE_RERUN; without it RegistrationViz compiles to a no-op, so every demo
 * builds and runs unchanged.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <Eigen/Dense>

#ifdef HAVE_RERUN
#include <array>
#include <cerrno>
#include <cstdlib>
#include <optional>

#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <rerun.hpp>
#endif

#include <memory>

#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/ann/kdtree_omp.hpp>
#include <small_gicp/factors/gicp_factor.hpp>
#include <small_gicp/util/normal_estimation_omp.hpp>
#include <small_gicp/util/lie.hpp>
#include <small_gicp/registration/registration.hpp>
#include <small_gicp/registration/reduction_omp.hpp>
#include <small_gicp/registration/registration_result.hpp>

// These public headers pull only Eigen and PCL - fast_gicp instantiates the
// templates inside libfast_vgicp_cuda.so - so nothing here needs the CUDA
// toolkit's include path.
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#include <fast_gicp/ndt/ndt_cuda.hpp>

namespace demo {

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/** The KITTI scan pair bundled with this chapter (sequence 04, frames 0 and 1) */
inline const char* kSampleSourceScan = "sample_sequences/000000.bin";
inline const char* kSampleTargetScan = "sample_sequences/000001.bin";

/**
 * Load a KITTI velodyne scan (.bin): raw float32 records of x, y, z, intensity
 */
inline CloudT::Ptr loadKittiBin(const std::string& file) {
    std::ifstream input(file, std::ios::binary);
    if (!input) {
        return nullptr;
    }

    CloudT::Ptr cloud(new CloudT);
    float record[4];
    while (input.read(reinterpret_cast<char*>(record), sizeof(record))) {
        cloud->push_back(PointT(record[0], record[1], record[2]));
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Voxel-downsample a cloud (leaf size in meters)
 */
inline CloudT::Ptr voxelDownsample(const CloudT& cloud, float leaf) {
    CloudT::Ptr out(new CloudT);
    pcl::VoxelGrid<PointT> voxel;
    voxel.setInputCloud(cloud.makeShared());
    voxel.setLeafSize(leaf, leaf, leaf);
    voxel.filter(*out);
    return out;
}

/**
 * Load a point cloud from a KITTI .bin or a .pcd file; nullptr on failure
 */
inline CloudT::Ptr loadCloud(const std::string& file) {
    if (fs::path(file).extension() == ".bin") {
        CloudT::Ptr cloud = loadKittiBin(file);
        return (cloud && !cloud->empty()) ? cloud : nullptr;
    }
    CloudT::Ptr cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1 || cloud->empty()) {
        return nullptr;
    }
    return cloud;
}

/**
 * Resolve a file in this chapter's data/ directory
 *
 * The demos are run both from the project root (./build/gicp_demo) and from
 * inside build/ (the Docker WORKDIR), and data/ is mounted at /data in the
 * container, so try each location. Returns an empty string when not found.
 */
inline std::string findDataFile(const std::string& filename) {
    const std::vector<std::string> candidates = {
        "data/" + filename,
        "../data/" + filename,
        "../../data/" + filename,
        "/data/" + filename,
    };

    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return path;
        }
    }
    return {};
}

/**
 * Build a rigid transform from a translation and XYZ rotation (radians)
 */
inline Eigen::Matrix4f makeTransform(float tx, float ty, float tz,
                                     float rx, float ry, float rz) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
    transform.translation() << tx, ty, tz;
    return transform.matrix();
}

// ============================================================
// KITTI odometry dataset handling
// ============================================================

/**
 * Files that make up one KITTI odometry sequence
 *
 * A sequence directory looks like
 *   sequences/04/{velodyne/,calib.txt,times.txt}
 *   poses/04.txt
 * with the ground-truth poses living outside the sequence folder. The scan pair
 * bundled under data/sample_sequences/ is flatter - scans, calib.txt and
 * poses.txt all sit in one directory - so both layouts are accepted.
 */
struct KittiSequence {
    std::string velodyne_dir;
    std::string calib_file;   // empty when unavailable
    std::string poses_file;   // empty when unavailable (sequences 11-21 have none)
    std::string name;
};

/**
 * Work out the sequence layout from a sequence dir, a velodyne dir, or the
 * directory a scan file sits in
 */
inline KittiSequence resolveKittiSequence(const std::string& path) {
    KittiSequence seq;

    fs::path root(path);
    // Tolerate a trailing slash, which would otherwise make filename() empty
    if (root.filename().empty()) {
        root = root.parent_path();
    }

    // Given a sequence directory: descend into velodyne/
    const fs::path velodyne = root / "velodyne";
    if (fs::is_directory(velodyne)) {
        seq.velodyne_dir = velodyne.string();
        seq.name = root.filename().string();
    } else {
        // Given the velodyne directory (or any directory holding the scans)
        seq.velodyne_dir = root.string();
        root = (root.filename() == "velodyne") ? root.parent_path() : root;
        seq.name = root.filename().string();
    }

    // calib.txt sits in the sequence directory; for the bundled pair it sits
    // next to the scans, which is the same directory
    const fs::path calib = root / "calib.txt";
    if (fs::exists(calib)) seq.calib_file = calib.string();

    // dataset/sequences/NN -> dataset/poses/NN.txt
    const fs::path poses = root.parent_path().parent_path() / "poses" / (seq.name + ".txt");
    if (fs::exists(poses)) {
        seq.poses_file = poses.string();
    } else if (fs::exists(root / "poses.txt")) {
        // Bundled pair: the two ground-truth lines ship alongside the scans
        seq.poses_file = (root / "poses.txt").string();
    }

    return seq;
}

/**
 * Read KITTI ground-truth poses: one 3x4 row-major matrix per line
 */
inline std::vector<Eigen::Matrix4f> loadKittiPoses(const std::string& filename) {
    std::vector<Eigen::Matrix4f> poses;

    std::ifstream file(filename);
    if (!file.is_open()) {
        return poses;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        bool ok = true;

        for (int i = 0; i < 3 && ok; ++i) {
            for (int j = 0; j < 4 && ok; ++j) {
                ok = static_cast<bool>(ss >> pose(i, j));
            }
        }

        if (ok) {
            poses.push_back(pose);
        }
    }

    return poses;
}

/**
 * Read the velodyne-to-camera transform (Tr) from a KITTI calib.txt;
 * identity when the file has no Tr line
 */
inline Eigen::Matrix4f loadVelodyneToCamera(const std::string& filename) {
    Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();

    std::ifstream file(filename);
    if (!file.is_open()) {
        return Tr;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("Tr:", 0) != 0) continue;

        std::istringstream ss(line.substr(3));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                ss >> Tr(i, j);
            }
        }
        break;
    }

    return Tr;
}

/**
 * Express camera-frame ground-truth poses in the velodyne frame
 *
 * KITTI poses are T_cam0_cami. Registration runs on velodyne scans, so the
 * comparable quantity is T_velo0_veloi = Tr^-1 * T_cam0_cami * Tr.
 */
inline std::vector<Eigen::Matrix4f> posesToLidarFrame(
    const std::vector<Eigen::Matrix4f>& cam_poses, const Eigen::Matrix4f& Tr) {
    const Eigen::Matrix4f Tr_inv = Tr.inverse();

    std::vector<Eigen::Matrix4f> out;
    out.reserve(cam_poses.size());
    for (const auto& pose : cam_poses) {
        out.push_back(Tr_inv * pose * Tr);
    }
    return out;
}

/**
 * Ground-truth poses of a sequence, in the velodyne frame;
 * empty when the sequence ships no poses or no calibration
 */
inline std::vector<Eigen::Matrix4f> loadKittiLidarPoses(const KittiSequence& seq) {
    if (seq.poses_file.empty() || seq.calib_file.empty()) {
        return {};
    }
    return posesToLidarFrame(loadKittiPoses(seq.poses_file),
                             loadVelodyneToCamera(seq.calib_file));
}

/**
 * Frame index of a KITTI scan, taken from its zero-padded filename;
 * -1 when the stem is not a number
 */
inline int kittiFrameIndex(const std::string& scan_file) {
    const std::string stem = fs::path(scan_file).stem().string();
    try {
        return std::stoi(stem);
    } catch (const std::exception&) {
        return -1;
    }
}

/**
 * Path of a KITTI scan by frame index: six zero-padded digits plus .bin
 */
inline std::string kittiScanPath(const std::string& velodyne_dir, int frame) {
    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << frame << ".bin";
    return (fs::path(velodyne_dir) / name.str()).string();
}

/**
 * Rotation and translation error of an estimate against ground truth
 */
struct PoseError {
    double rotation_deg;
    double translation_m;
};

inline PoseError poseError(const Eigen::Matrix4f& estimated,
                          const Eigen::Matrix4f& ground_truth) {
    const Eigen::Matrix4f delta = ground_truth.inverse() * estimated;

    // Angle of the residual rotation.
    //
    // Do NOT use the textbook acos((trace - 1) / 2) here. Neither input is
    // exactly orthonormal - the KITTI ground truth is rebased through Tr and
    // round-tripped via text, and the estimate is a long product of float
    // matrices - so the residual trace lands slightly above 3, the argument
    // clamps to 1, and acos returns exactly 0. That silently reports "no
    // rotation error" precisely in the sub-degree range that matters. Going
    // through a normalized quaternion absorbs the non-orthonormality, and the
    // atan2 form stays accurate for both tiny and large angles.
    Eigen::Quaterniond q(delta.block<3, 3>(0, 0).cast<double>());
    q.normalize();
    const double angle = 2.0 * std::atan2(q.vec().norm(), std::abs(q.w()));

    return {angle * 180.0 / M_PI, delta.block<3, 1>(0, 3).norm()};
}

/**
 * One method's error curve, collected step by step during align()
 *
 * Filled in by attachIterationLogging() and handed to
 * RegistrationViz::logErrorCurves() once every method has run, so that all of
 * them can be drawn on one pair of graphs.
 */
struct ErrorTrace {
    std::string method;
    uint8_t r = 255, g = 255, b = 255;
    std::vector<PoseError> steps;

    /// Wall-clock offset of each step from the start of align(), parallel to
    /// `steps`.
    ///
    /// Recorded because a step is not a unit of time, and the two axes rank the
    /// methods differently. On the bundled pair PCL GICP converges in 10 steps
    /// and the CUDA VGICP in 8, so against iteration count they look like much
    /// the same method - while a PCL GICP step costs about 32 ms and a CUDA step
    /// about 0.5 ms. The step axis answers "how many iterations", which is a
    /// question about the optimizer; the elapsed axis answers "how long", which
    /// is the one a pipeline actually pays. Both go into the viewer, and the
    /// timeline picker switches between them - see logErrorCurves.
    std::vector<double> elapsed_ms;

    /// Clock origin, set immediately before align() so that preprocessing -
    /// which the backends divide up very differently - stays out of the curve.
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    /// Seconds from t0 to now
    double sinceStart() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    }
};

/// One recorded optimization step: where the method was, and when it got there
struct TracedStep {
    Eigen::Isometry3d pose;
    double elapsed_ms;
};

/**
 * Recover the rigid transform that maps `source` onto `transformed`
 *
 * PCL's per-iteration visualization callback hands over the source cloud as it
 * currently stands, not the transform behind it. The two clouds hold the same
 * points in the same order, though, so one SVD fit recovers that transform
 * exactly - and exactly is the word: the fit is over a rigid motion of identical
 * points, so it is not an approximation. A strided subsample is enough to pin it
 * down and keeps the cost off the registration timings being reported.
 */
inline Eigen::Matrix4f recoverTransform(const CloudT& source, const CloudT& transformed) {
    Eigen::Matrix4f estimate = Eigen::Matrix4f::Identity();
    if (source.size() != transformed.size() || source.size() < 3) {
        return estimate;
    }

    constexpr std::size_t kMaxSamples = 2000;
    const std::size_t stride = std::max<std::size_t>(1, source.size() / kMaxSamples);

    CloudT from, to;
    from.reserve(source.size() / stride + 1);
    to.reserve(source.size() / stride + 1);
    for (std::size_t i = 0; i < source.size(); i += stride) {
        from.push_back(source[i]);
        to.push_back(transformed[i]);
    }

    pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
    svd.estimateRigidTransformation(from, to, estimate);
    return estimate;
}

/**
 * One KITTI scan pair to register, with the known relative pose
 */
struct KittiPair {
    CloudT::Ptr source;
    CloudT::Ptr target;

    /// Maps source-frame points into the target frame - what align() should find
    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();
    bool has_ground_truth = false;

    std::string source_file;
    std::string target_file;
    std::string sequence;
    int source_frame = -1;
    int target_frame = -1;

    /// Where the scans came from, so a demo can reach for further frames
    KittiSequence sequence_files;
    /// Ground-truth poses of the whole sequence; empty when unavailable
    std::vector<Eigen::Matrix4f> lidar_poses;
};

/**
 * Load two KITTI scans and the ground-truth transform between them
 *
 * Pass the scan paths, or leave them empty to use the pair bundled with this
 * chapter. The sequence's calib.txt and poses file are located relative to the
 * scans, so nothing but the scan paths has to be given on the command line.
 * Returns a pair with a null source when a scan could not be read.
 */
inline KittiPair loadKittiPair(const std::string& source_path = {},
                              const std::string& target_path = {}) {
    KittiPair pair;

    pair.source_file = source_path.empty() ? findDataFile(kSampleSourceScan) : source_path;
    pair.target_file = target_path.empty() ? findDataFile(kSampleTargetScan) : target_path;

    if (pair.source_file.empty() || pair.target_file.empty()) {
        std::cerr << "Error: no KITTI scans given and the bundled pair under "
                     "data/sample_sequences/ was not found.\n";
        return pair;
    }

    pair.source = loadCloud(pair.source_file);
    if (!pair.source) {
        std::cerr << "Error: could not read source scan: " << pair.source_file << "\n";
        return pair;
    }
    pair.target = loadCloud(pair.target_file);
    if (!pair.target) {
        std::cerr << "Error: could not read target scan: " << pair.target_file << "\n";
        pair.source.reset();
        return pair;
    }

    // Ground truth: the two scans have to come from the same sequence, and that
    // sequence has to be one of 00-10 (11-21 ship no poses)
    pair.sequence_files =
        resolveKittiSequence(fs::path(pair.source_file).parent_path().string());
    pair.sequence = pair.sequence_files.name;
    pair.source_frame = kittiFrameIndex(pair.source_file);
    pair.target_frame = kittiFrameIndex(pair.target_file);

    const bool same_sequence = fs::path(pair.source_file).parent_path() ==
                               fs::path(pair.target_file).parent_path();

    if (same_sequence && pair.source_frame >= 0 && pair.target_frame >= 0) {
        pair.lidar_poses = loadKittiLidarPoses(pair.sequence_files);

        const int max_frame = std::max(pair.source_frame, pair.target_frame);
        if (static_cast<int>(pair.lidar_poses.size()) > max_frame) {
            // Points of the source frame, expressed in the target frame
            pair.ground_truth = pair.lidar_poses[pair.target_frame].inverse() *
                                pair.lidar_poses[pair.source_frame];
            pair.has_ground_truth = true;
        }
    }

    return pair;
}

/**
 * Report what was loaded, and warn when there is no ground truth to score against
 */
inline void printKittiPair(const KittiPair& pair) {
    std::cout << "KITTI scans" << (pair.sequence.empty() ? "" : " (sequence " + pair.sequence + ")")
              << ":\n";
    std::cout << "  Source: " << pair.source_file << "  (" << pair.source->size()
              << " points, frame " << pair.source_frame << ")\n";
    std::cout << "  Target: " << pair.target_file << "  (" << pair.target->size()
              << " points, frame " << pair.target_frame << ")\n";

    if (pair.has_ground_truth) {
        const Eigen::Vector3f t = pair.ground_truth.block<3, 1>(0, 3);
        const PoseError motion = poseError(pair.ground_truth, Eigen::Matrix4f::Identity());
        std::cout << "  Ground truth (source -> target): t = [" << std::fixed
                  << std::setprecision(3) << t.x() << ", " << t.y() << ", " << t.z()
                  << "] m, |t| = " << motion.translation_m << " m, rotation = "
                  << std::setprecision(3) << motion.rotation_deg << " deg\n";
    } else {
        std::cout << "  Ground truth: NOT AVAILABLE - the calib.txt / poses file for this\n"
                     "                sequence was not found, so the errors below are\n"
                     "                measured against the identity and are meaningless.\n"
                     "                Point the demo at a KITTI sequence 00-10 to get real\n"
                     "                numbers (see README).\n";
    }
}

/**
 * Print an estimate's error against ground truth, or its distance from the
 * identity when no ground truth is available
 */
inline void printErrorRow(const std::string& label, const Eigen::Matrix4f& estimated,
                         const KittiPair& pair, double time_ms) {
    const PoseError err = poseError(estimated, pair.ground_truth);
    std::cout << "  " << label << ": translation error " << std::fixed << std::setprecision(4)
              << err.translation_m << " m, rotation error " << err.rotation_deg << " deg, "
              << std::setprecision(1) << time_ms << " ms\n";
}

#ifdef HAVE_RERUN

/// True when a TCP server accepts connections at the URL's host:port.
/// connect_grpc() itself never fails on an absent viewer - it just retries
/// in the background - so probe reachability first and fall back to running
/// without streaming.
inline bool viewerReachable(const std::string& url) {
    // Expect rerun+http://HOST:PORT/... - fall back to trying when unparsable.
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return true;
    std::string hostport = url.substr(scheme + 3);
    const auto slash = hostport.find('/');
    if (slash != std::string::npos) hostport = hostport.substr(0, slash);
    const auto colon = hostport.rfind(':');
    if (colon == std::string::npos) return true;
    const std::string host = hostport.substr(0, colon);
    const std::string port = hostport.substr(colon + 1);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) return false;

    bool ok = false;
    for (addrinfo* ai = res; ai && !ok; ai = ai->ai_next) {
        const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            ok = true;
        } else if (errno == EINPROGRESS) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(fd, &wfds);
            timeval tv{2, 0};  // 2 s connect timeout
            if (select(fd + 1, nullptr, &wfds, nullptr, &tv) == 1) {
                int err = 0;
                socklen_t len = sizeof(err);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
                ok = (err == 0);
            }
        }
        close(fd);
    }
    freeaddrinfo(res);
    return ok;
}

/// Viewer address: RERUN_URL env var, or the default local viewer
inline std::string viewerUrl() {
    const char* env_url = std::getenv("RERUN_URL");
    return env_url ? env_url : "rerun+http://127.0.0.1:9876/proxy";
}

inline std::vector<rerun::Vec3D> toVecs(const CloudT& cloud) {
    std::vector<rerun::Vec3D> pts;
    pts.reserve(cloud.size());
    for (const auto& p : cloud) {
        pts.push_back({p.x, p.y, p.z});
    }
    return pts;
}

/// Blue (low) -> yellow (high) ramp over roughly ground..building height
inline std::vector<rerun::Color> heightColors(const CloudT& cloud) {
    std::vector<rerun::Color> colors;
    colors.reserve(cloud.size());
    for (const auto& p : cloud) {
        const float s = std::clamp((p.z + 2.5f) / 7.5f, 0.0f, 1.0f);
        colors.push_back(rerun::Color(static_cast<uint8_t>(40 + s * 215),
                                      static_cast<uint8_t>(90 + s * 130),
                                      static_cast<uint8_t>(210 - s * 170)));
    }
    return colors;
}

/**
 * Streams pairwise-registration inputs and results to a rerun viewer:
 * one static entity per cloud so the viewer overlays target, initial
 * source, and each method's aligned source. logIteration() additionally
 * puts every optimization step on an "iteration" timeline, so the
 * convergence can be scrubbed frame by frame.
 */
class RegistrationViz {
public:
    /// The stream is only constructed once a viewer has answered. A
    /// RecordingStream with nowhere to send its data keeps it in a buffered sink
    /// and warns that it is dropping it when the process exits, which reads like
    /// an error in a demo that is deliberately running without a viewer.
    explicit RegistrationViz(const std::string& app_id) {
        const std::string url = viewerUrl();
        if (!viewerReachable(url)) {
            std::cerr << "Note: no rerun viewer reachable at " << url
                      << " - running without live streaming.\n"
                         "      Start one on the host first (rerun &); with Docker "
                         "add --network=host.\n";
            return;
        }

        rec_.emplace(app_id);
        if (!rec_->connect_grpc(url).is_ok()) {
            rec_.reset();
            return;
        }
        connected_ = true;
        std::cout << "Streaming to rerun viewer at " << url << std::endl;
        rec_->log_static("registration", rerun::ViewCoordinates::RIGHT_HAND_Z_UP);
    }

    bool active() const { return connected_; }

    /// Log a cloud under registration/<name> with a fixed color
    void logCloud(const std::string& name, const CloudT& cloud,
                  uint8_t r, uint8_t g, uint8_t b) {
        if (!connected_ || cloud.empty()) return;
        rec_->log_static("registration/" + name,
                        rerun::Points3D(toVecs(cloud))
                            .with_colors({rerun::Color(r, g, b)})
                            .with_radii({rerun::Radius::ui_points(1.5f)}));
    }

    /// Log a cloud colored by height, so a street scene stays readable
    void logCloudByHeight(const std::string& name, const CloudT& cloud) {
        if (!connected_ || cloud.empty()) return;
        rec_->log_static("registration/" + name,
                        rerun::Points3D(toVecs(cloud))
                            .with_colors(heightColors(cloud))
                            .with_radii({rerun::Radius::ui_points(1.0f)}));
    }

    /// Log one optimization step of a method on the "iteration" timeline
    void logIteration(const std::string& method, int iteration, const CloudT& cloud,
                      uint8_t r, uint8_t g, uint8_t b) {
        if (!connected_ || cloud.empty()) return;
        rec_->set_time_sequence("iteration", iteration);
        rec_->log("registration/steps/" + method,
                 rerun::Points3D(toVecs(cloud))
                     .with_colors({rerun::Color(r, g, b)})
                     .with_radii({rerun::Radius::ui_points(1.5f)}));
    }

    /// Plot every method's error curve, all methods on one pair of graphs
    ///
    /// The methods run one after another, so their curves are collected during
    /// align() and sent here afterwards. They go into a single entity per metric,
    /// carrying one scalar per method at each step: that is what puts them in the
    /// same graph with a shared axis, which is the whole point - a curve per view
    /// would leave the reader comparing across separate y-scales.
    ///
    /// A method that converged early holds its final value for the remaining
    /// steps, since that is where it actually ended up.
    void logErrorCurves(const std::vector<ErrorTrace>& traces) {
        if (!connected_ || traces.empty()) return;

        std::vector<std::string> names;
        std::vector<rerun::Color> colors;
        std::size_t longest = 0;
        for (const auto& t : traces) {
            names.push_back(t.method);
            colors.push_back(rerun::Color(t.r, t.g, t.b));
            longest = std::max(longest, t.steps.size());
        }
        if (longest == 0) return;

        // One entity per plot carrying every method as a series, NOT an entity
        // per method. Splitting them would put each method in its own graph in
        // the viewer's default layout - which defeats the whole point, since the
        // comparison needs a shared axis - and only looks right if the reader
        // happens to load a blueprint that regroups them.
        for (const char* plot : {"translation_error", "rotation_error"}) {
            rec_->log_static(plot, rerun::SeriesLines()
                                       .with_names(names)
                                       .with_colors(colors)
                                       .with_widths({1.5f}));
        }

        // Against iteration count. A method that converged early holds its final
        // value for the remaining steps, since that is where it ended up.
        rec_->reset_time();
        for (std::size_t step = 0; step < longest; ++step) {
            std::vector<double> translation, rotation;
            translation.reserve(traces.size());
            rotation.reserve(traces.size());
            for (const auto& t : traces) {
                const PoseError e = t.steps.empty()
                                        ? PoseError{0.0, 0.0}
                                        : t.steps[std::min(step, t.steps.size() - 1)];
                translation.push_back(e.translation_m);
                rotation.push_back(e.rotation_deg);
            }
            rec_->set_time_sequence("iteration", static_cast<int64_t>(step));
            rec_->log("translation_error", rerun::Scalars(translation));
            rec_->log("rotation_error", rerun::Scalars(rotation));
        }

        // Against wall-clock time. Step k of each method lands at a different
        // millisecond, so a shared entity needs a shared set of time points:
        // merge every method's timestamps and sample all of them at each one.
        std::vector<double> times;
        for (const auto& t : traces) {
            times.insert(times.end(), t.elapsed_ms.begin(), t.elapsed_ms.end());
        }
        std::sort(times.begin(), times.end());
        times.erase(std::unique(times.begin(), times.end()), times.end());

        // reset_time() so these rows carry only "elapsed" and do not also land on
        // the iteration timeline at whatever index it was left at.
        rec_->reset_time();
        for (double ms : times) {
            std::vector<double> translation, rotation;
            translation.reserve(traces.size());
            rotation.reserve(traces.size());
            for (const auto& t : traces) {
                const PoseError e = interpolateAt(t, ms);
                translation.push_back(e.translation_m);
                rotation.push_back(e.rotation_deg);
            }
            rec_->set_time_duration_secs("elapsed", ms / 1000.0);
            rec_->log("translation_error", rerun::Scalars(translation));
            rec_->log("rotation_error", rerun::Scalars(rotation));
        }
    }

    /// Log the source cloud moved by an estimated transform
    void logAligned(const std::string& name, const CloudT& source,
                    const Eigen::Matrix4f& transform,
                    uint8_t r, uint8_t g, uint8_t b) {
        if (!connected_) return;
        CloudT aligned;
        pcl::transformPointCloud(source, aligned, transform);
        logCloud(name, aligned, r, g, b);
    }

    /// Log FPFH correspondences as line segments between the two clouds
    void logCorrespondences(const std::string& name, const CloudT& source,
                            const CloudT& target,
                            const std::vector<std::pair<int, int>>& pairs,
                            uint8_t r, uint8_t g, uint8_t b) {
        if (!connected_ || pairs.empty()) return;

        std::vector<rerun::components::LineStrip3D> strips;
        strips.reserve(pairs.size());
        for (const auto& [si, ti] : pairs) {
            const auto& sp = source[si];
            const auto& tp = target[ti];
            strips.push_back(rerun::components::LineStrip3D(
                std::vector<rerun::Vec3D>{{sp.x, sp.y, sp.z}, {tp.x, tp.y, tp.z}}));
        }
        rec_->log_static("registration/" + name,
                        rerun::LineStrips3D(strips)
                            .with_colors({rerun::Color(r, g, b)})
                            .with_radii({rerun::Radius::ui_points(0.5f)}));
    }

private:
    /// A trace's error at an arbitrary elapsed time, linearly interpolated
    /// between its own samples
    ///
    /// Interpolating rather than holding the previous value is what keeps the
    /// drawn curve honest: every extra point lands exactly on a segment the
    /// method's own samples already define, so sampling all methods at a merged
    /// set of times draws the same polyline as plotting each one alone. Holding
    /// instead would introduce staircases that no method actually traced.
    ///
    /// Outside a trace's range the endpoints are clamped, which is also right:
    /// before it started it was at its initial guess, and once it converged and
    /// stopped it stayed where it finished.
    static PoseError interpolateAt(const ErrorTrace& trace, double ms) {
        const std::size_t n = std::min(trace.steps.size(), trace.elapsed_ms.size());
        if (n == 0) return {0.0, 0.0};
        if (ms <= trace.elapsed_ms.front()) return trace.steps.front();
        if (ms >= trace.elapsed_ms[n - 1]) return trace.steps[n - 1];

        const auto it = std::lower_bound(trace.elapsed_ms.begin(),
                                         trace.elapsed_ms.begin() + n, ms);
        const std::size_t hi = static_cast<std::size_t>(it - trace.elapsed_ms.begin());
        const std::size_t lo = hi - 1;
        const double span = trace.elapsed_ms[hi] - trace.elapsed_ms[lo];
        const double f = span > 0.0 ? (ms - trace.elapsed_ms[lo]) / span : 0.0;
        return {trace.steps[lo].rotation_deg +
                    f * (trace.steps[hi].rotation_deg - trace.steps[lo].rotation_deg),
                trace.steps[lo].translation_m +
                    f * (trace.steps[hi].translation_m - trace.steps[lo].translation_m)};
    }

    std::optional<rerun::RecordingStream> rec_;
    bool connected_ = false;
};

#else  // !HAVE_RERUN

class RegistrationViz {
public:
    explicit RegistrationViz(const std::string&) {}
    bool active() const { return false; }
    void logCloud(const std::string&, const CloudT&, uint8_t, uint8_t, uint8_t) {}
    void logCloudByHeight(const std::string&, const CloudT&) {}
    void logIteration(const std::string&, int, const CloudT&, uint8_t, uint8_t,
                      uint8_t) {}
    void logErrorCurves(const std::vector<ErrorTrace>&) {}
    void logAligned(const std::string&, const CloudT&, const Eigen::Matrix4f&,
                    uint8_t, uint8_t, uint8_t) {}
    void logCorrespondences(const std::string&, const CloudT&, const CloudT&,
                            const std::vector<std::pair<int, int>>&, uint8_t, uint8_t,
                            uint8_t) {}
};

#endif  // HAVE_RERUN

/**
 * Stream every optimization iteration of a PCL registration object: PCL
 * invokes the visualization callback once per iteration with the current
 * intermediate source cloud, which lands on the viewer's "iteration"
 * timeline. Call before align(); no-op when the viz is not connected.
 *
 * Pass `source`, `ground_truth` and a `trace` as well and each step is
 * additionally scored against the true transform. The scores accumulate in the
 * trace rather than going straight to the viewer, so several methods can later be
 * drawn on one graph by RegistrationViz::logErrorCurves(). That turns "it
 * converged" into something you can read off: how fast each method approaches the
 * answer, whether it is still improving when the iteration limit stops it, and
 * whether it converged to the wrong place.
 * `source` must be the very cloud handed to setInputSource().
 */
template <typename Registration>
void attachIterationLogging(Registration& reg, RegistrationViz* viz,
                            const std::string& method,
                            uint8_t r, uint8_t g, uint8_t b,
                            const CloudT::ConstPtr& source = nullptr,
                            const Eigen::Matrix4f* ground_truth = nullptr,
                            ErrorTrace* trace = nullptr,
                            const Eigen::Matrix4f* initial_guess = nullptr) {
    if (!viz || !viz->active()) return;

    // The callbacks outlive this scope, so both matrices are copied in rather
    // than captured by pointer
    const Eigen::Matrix4f gt = ground_truth ? *ground_truth : Eigen::Matrix4f::Identity();
    const Eigen::Matrix4f guess =
        initial_guess ? *initial_guess : Eigen::Matrix4f::Identity();

    const bool score = source && ground_truth && trace;
    if (score) {
        trace->method = method;
        trace->r = r;
        trace->g = g;
        trace->b = b;
        // Step 0 is where the method starts, which is the initial guess. It has
        // to be seeded here rather than read off the first callback: PCL fires
        // one callback before the first update whose cloud does not carry the
        // guess, so taking it at face value would claim every run started from
        // the identity - harmless when the guess IS the identity, and wildly
        // wrong when it is not (a refinement seeded by global registration would
        // appear to start from the full displacement it was handed the answer to).
        trace->steps.assign(1, poseError(guess, gt));
        // Step 0 is the starting point, so it sits at t = 0 by definition. The
        // clock itself is restarted by runPcl() just before align(), which is
        // what makes the elapsed axis measure the solve rather than the setup.
        trace->elapsed_ms.assign(1, 0.0);
        trace->t0 = std::chrono::steady_clock::now();
    }

    std::function<void(const CloudT&, const pcl::Indices&, const CloudT&,
                       const pcl::Indices&)>
        callback = [viz, method, r, g, b, source, gt, score, trace, callbacks = 0](
                       const CloudT& intermediate, const pcl::Indices&,
                       const CloudT&, const pcl::Indices&) mutable {
            // Drop that same pre-update callback here, so 3D step k and curve
            // point k both mean "after iteration k"
            if (callbacks++ == 0) return;

            viz->logIteration(method, callbacks - 1, intermediate, r, g, b);
            if (score) {
                // Stamp before the SVD fit below, so the cost of recovering the
                // transform for the trace is not charged to the method's curve
                const double elapsed = trace->sinceStart();
                trace->steps.push_back(
                    poseError(recoverTransform(*source, intermediate), gt));
                trace->elapsed_ms.push_back(elapsed);
            }
        };
    reg.registerVisualizationCallback(callback);
}

// ===========================================================================
// Backend-neutral results
// ===========================================================================

/**
 * One registration run, from any backend
 *
 * Preprocessing and alignment are timed separately on purpose. PCL builds its
 * correspondence structures inside align(), while small_gicp and fast_gicp build
 * KdTrees, covariances and voxel maps when the clouds are handed over. Reporting
 * only align() would therefore flatter them: the optimizer loop really is far
 * faster, but a pipeline pays for the setup too. Two numbers keep both facts
 * visible.
 *
 * The split is not perfectly clean for one backend: NDTCuda builds its voxel
 * maps at the top of its computeTransformation(), so that cost lands in
 * align_ms rather than preprocess_ms. total_ms is the number to trust when
 * comparing across backends.
 */
struct RunResult {
    std::string method;
    bool converged = false;
    double fitness = std::numeric_limits<double>::quiet_NaN();
    PoseError error{0.0, 0.0};
    double preprocess_ms = 0.0;
    double align_ms = 0.0;
    double total_ms = 0.0;
    int iterations = -1;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
};

inline void printRunResults(const std::vector<RunResult>& results) {
    std::cout << std::string(104, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Method" << std::right
              << std::setw(8) << "Conv"
              << std::setw(7) << "Iters"
              << std::setw(14) << "Trans Err (m)"
              << std::setw(14) << "Rot Err (deg)"
              << std::setw(13) << "Prep (ms)"
              << std::setw(13) << "Align (ms)"
              << std::setw(13) << "Total (ms)" << std::endl;
    std::cout << std::string(104, '-') << std::endl;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(22) << r.method << std::right
                  << std::setw(8) << (r.converged ? "YES" : "NO");
        if (r.iterations >= 0) std::cout << std::setw(7) << r.iterations;
        else                   std::cout << std::setw(7) << "-";
        std::cout << std::fixed
                  << std::setw(14) << std::setprecision(4) << r.error.translation_m
                  << std::setw(14) << std::setprecision(4) << r.error.rotation_deg
                  << std::setw(13) << std::setprecision(1) << r.preprocess_ms
                  << std::setw(13) << std::setprecision(1) << r.align_ms
                  << std::setw(13) << std::setprecision(1) << r.total_ms << std::endl;
    }
    std::cout << std::string(104, '-') << std::endl;
}

/// Only some PCL registration classes report how many iterations they took:
/// NormalDistributionsTransform has getFinalNumIteration(), while ICP and GICP
/// keep nr_iterations_ protected. Detect it rather than print a number that is
/// not there - the count matters for NDT, where "converged after 1 iteration"
/// is the signature of a silent failure.
template <typename T, typename = void>
struct HasFinalNumIteration : std::false_type {};
template <typename T>
struct HasFinalNumIteration<
    T, std::void_t<decltype(std::declval<const T&>().getFinalNumIteration())>>
    : std::true_type {};

/**
 * Run a PCL registration and time setup separately from the solve
 *
 * PCL's split looks lopsided next to the other backends, and that is the point:
 * setInputTarget() only builds a search tree, while GICP's per-point covariances
 * and NDT's voxel grid are computed inside align(). So nearly all of PCL's cost
 * lands in align_ms, whereas small_gicp and fast_gicp have already paid part of
 * theirs by the time align() is called. Compare on total_ms.
 *
 * Call attachIterationLogging() on `reg` beforehand to also collect a curve.
 */
template <typename Reg>
RunResult runPcl(const std::string& method, Reg& reg,
                 const CloudT::Ptr& source, const CloudT::Ptr& target,
                 const Eigen::Matrix4f& ground_truth,
                 const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                 ErrorTrace* trace = nullptr) {
    using clock = std::chrono::high_resolution_clock;
    RunResult result;
    result.method = method;

    const auto t0 = clock::now();
    reg.setInputSource(source);
    reg.setInputTarget(target);
    const auto t1 = clock::now();

    // Restart the trace clock here, so its elapsed axis starts where align()
    // does. attachIterationLogging() runs before the clouds are even handed
    // over, so without this the curve would carry PCL's setup as well.
    if (trace) trace->t0 = std::chrono::steady_clock::now();

    CloudT aligned;
    reg.align(aligned, initial_guess);
    const auto t2 = clock::now();

    result.converged = reg.hasConverged();
    result.transform = reg.getFinalTransformation();
    result.error = poseError(result.transform, ground_truth);
    if constexpr (HasFinalNumIteration<Reg>::value) {
        result.iterations = reg.getFinalNumIteration();
    }
    result.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.align_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.total_ms = result.preprocess_ms + result.align_ms;
    // Deliberately after the timers: getFitnessScore() runs its own full
    // nearest-neighbour pass and would otherwise be charged to alignment
    result.fitness = reg.getFitnessScore();
    return result;
}

/// Score a recorded pose sequence against ground truth, ready for logErrorCurves()
inline ErrorTrace traceFromPoses(const std::string& method,
                                 uint8_t r, uint8_t g, uint8_t b,
                                 const std::vector<TracedStep>& recorded,
                                 const Eigen::Matrix4f& ground_truth) {
    ErrorTrace trace;
    trace.method = method;
    trace.r = r;
    trace.g = g;
    trace.b = b;
    trace.steps.reserve(recorded.size());
    trace.elapsed_ms.reserve(recorded.size());
    for (const auto& step : recorded) {
        trace.steps.push_back(poseError(step.pose.matrix().cast<float>(), ground_truth));
        trace.elapsed_ms.push_back(step.elapsed_ms);
    }
    return trace;
}

// ===========================================================================
// fast_gicp: per-iteration tracing
// ===========================================================================

/**
 * A fast_gicp registration class that records the pose at every outer iteration
 *
 * Do NOT reach for registerVisualizationCallback() here. fast_gicp's classes do
 * derive from pcl::Registration, so the call compiles and even returns true -
 * but `update_visualizer_` appears nowhere in fast_gicp, so it is never invoked
 * during optimization. PCL 1.14 fires the callback once at registration time,
 * and attachIterationLogging() deliberately drops exactly that call, so the
 * trace would come back holding only the seeded initial guess: one point on the
 * graph, no compile error, no warning. Silent data loss.
 *
 * linearize() is the hook that does work. It is protected virtual, every leaf
 * class overrides it, and step_gn() and step_lm() each call it exactly once per
 * OUTER iteration with the currently accepted pose. So recording there yields
 * the pose before each update - the initial guess, then the result of iteration
 * 1, of iteration 2, and so on - and appending align()'s final transformation
 * completes the sequence. Step k then means "after iteration k", which is what
 * the PCL demos plot, so the curves are directly comparable.
 *
 * step_optimize()/step_gn()/step_lm() are protected but NOT virtual, so
 * linearize() really is the only available seam.
 *
 * Caveat: evaluateCost() is public and also calls linearize(). Set the sink
 * immediately before align() and do not call evaluateCost() while it is set, or
 * the trace picks up samples that are not optimization steps.
 */
template <typename Base>
class Traced : public Base {
public:
    /// Start recording. Call immediately before align() - the elapsed clock
    /// starts here, so setInputSource/setInputTarget stay out of the curve.
    void traceInto(std::vector<TracedStep>* sink) {
        sink_ = sink;
        t0_ = std::chrono::steady_clock::now();
    }

protected:
    double linearize(const Eigen::Isometry3d& trans,
                     Eigen::Matrix<double, 6, 6>* H = nullptr,
                     Eigen::Matrix<double, 6, 1>* b = nullptr) override {
        if (sink_) {
            sink_->push_back({trans, std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - t0_)
                                         .count()});
        }
        return Base::linearize(trans, H, b);
    }

private:
    std::vector<TracedStep>* sink_ = nullptr;
    std::chrono::steady_clock::time_point t0_{};
};

/**
 * Run any fast_gicp registration and time setup separately from the solve
 *
 * `poses`, when given, receives the per-iteration trajectory (see Traced).
 */
template <typename Reg>
RunResult runFastGicp(const std::string& method, Reg& reg,
                      const CloudT::Ptr& source, const CloudT::Ptr& target,
                      const Eigen::Matrix4f& ground_truth,
                      const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                      std::vector<TracedStep>* poses = nullptr) {
    using clock = std::chrono::high_resolution_clock;
    RunResult result;
    result.method = method;

    const auto t0 = clock::now();
    reg.setInputTarget(target);
    reg.setInputSource(source);
    const auto t1 = clock::now();

    if (poses) {
        poses->clear();
        reg.traceInto(poses);
    }

    CloudT aligned;
    reg.align(aligned, initial_guess);
    const auto t2 = clock::now();

    if (poses) {
        reg.traceInto(nullptr);
        // linearize() hands over pre-update poses, so the last accepted pose is
        // still missing
        // `reg` is a template parameter, so cast<> is a dependent member template
        // and needs the `template` keyword to parse as one. The last accepted
        // pose lands at the moment align() returned.
        poses->push_back(
            {Eigen::Isometry3d(reg.getFinalTransformation().template cast<double>()),
             std::chrono::duration<double, std::milli>(t2 - t1).count()});
    }

    result.converged = reg.hasConverged();
    result.transform = reg.getFinalTransformation();
    result.error = poseError(result.transform, ground_truth);
    result.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.align_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.total_ms = result.preprocess_ms + result.align_ms;
    if (poses) result.iterations = static_cast<int>(poses->size()) - 1;
    return result;
}

// ===========================================================================
// small_gicp: per-iteration tracing
// ===========================================================================

/**
 * small_gicp's Gauss-Newton optimizer, with the accepted pose recorded per step
 *
 * small_gicp has no callback of any kind in its registration path - the only
 * introspection is a `verbose` flag that prints scalars, including only the
 * NORMS of the update, so a trajectory cannot be reconstructed from it. What it
 * does have is a duck-typed Optimizer template parameter, which is the seam
 * used here: this mirrors small_gicp::GaussNewtonOptimizer and additionally
 * appends each post-update pose to an external vector.
 *
 * Recording after the update (rather than before, as the fast_gicp hook must)
 * means step k is "after iteration k" directly, matching the PCL demos.
 *
 * The sink is a raw pointer to storage the caller owns because Registration's
 * align() is const, so the optimizer member is const inside optimize() and
 * cannot hold a vector it appends to.
 *
 * Note the alternative that looks equivalent and is not: calling align()
 * repeatedly with max_iterations=1. That is bit-identical for Gauss-Newton, but
 * NOT for small_gicp's default Levenberg-Marquardt optimizer, whose damping
 * `lambda` is a function-local reset on every call - so the annealing is thrown
 * away and the reported convergence rate becomes an artifact. It also rebuilds
 * the per-point factor vector once per recorded step.
 */
struct RecordingGaussNewton {
    template <typename TargetPointCloud, typename SourcePointCloud, typename TargetTree,
              typename CorrespondenceRejector, typename TerminationCriteria,
              typename Reduction, typename Factor, typename GeneralFactor>
    small_gicp::RegistrationResult optimize(
        const TargetPointCloud& target, const SourcePointCloud& source,
        const TargetTree& target_tree, const CorrespondenceRejector& rejector,
        const TerminationCriteria& criteria, Reduction& reduction,
        const Eigen::Isometry3d& init_T, std::vector<Factor>& factors,
        GeneralFactor& general_factor) const {
        small_gicp::RegistrationResult result(init_T);

        // Clock starts inside optimize(), which is where the solve begins - the
        // KdTrees and covariances were already built before align() was called.
        const auto t0 = std::chrono::steady_clock::now();
        const auto elapsed = [&t0] {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                .count();
        };

        if (sink) {
            sink->clear();
            sink->push_back({init_T, 0.0});  // step 0 is where the method starts
        }

        for (int i = 0; i < max_iterations && !result.converged; i++) {
            auto [H, b, e] = reduction.linearize(target, source, target_tree, rejector,
                                                 result.T_target_source, factors);
            general_factor.update_linearized_system(target, source, target_tree,
                                                    result.T_target_source, &H, &b, &e);

            const Eigen::Matrix<double, 6, 1> delta =
                (H + lambda * Eigen::Matrix<double, 6, 6>::Identity()).ldlt().solve(-b);

            result.converged = criteria.converged(delta);
            result.T_target_source = result.T_target_source * small_gicp::se3_exp(delta);
            result.iterations = i;
            result.H = H;
            result.b = b;
            result.error = e;

            if (sink) sink->push_back({result.T_target_source, elapsed()});
        }

        result.num_inliers = std::count_if(factors.begin(), factors.end(),
                                           [](const auto& f) { return f.inlier(); });
        return result;
    }

    int max_iterations = 50;
    double lambda = 1e-6;
    std::vector<TracedStep>* sink = nullptr;
};

/// Copy a PCL cloud into small_gicp's own container
inline std::shared_ptr<small_gicp::PointCloud> toSmallGicp(const CloudT& cloud) {
    std::vector<Eigen::Vector4f> points;
    points.reserve(cloud.size());
    for (const auto& p : cloud) points.emplace_back(p.x, p.y, p.z, 1.0f);
    return std::make_shared<small_gicp::PointCloud>(points);
}

struct SmallGicpConfig {
    int num_threads = 4;
    int num_neighbors = 20;
    double max_correspondence_distance = 2.0;
    int max_iterations = 50;
};

/**
 * small_gicp GICP through the native Registration<> template
 *
 * RegistrationPCL, the drop-in that would let the PCL code stay untouched, is
 * deliberately avoided: it hardcodes Registration<GICPFactor,
 * ParallelReductionOMP> with no way to inject an optimizer, so the per-iteration
 * curves cannot be recovered through it.
 */
inline RunResult runSmallGicpGICP(const std::string& method,
                                  const CloudT& source, const CloudT& target,
                                  const Eigen::Matrix4f& ground_truth,
                                  const Eigen::Matrix4f& initial_guess,
                                  const SmallGicpConfig& cfg,
                                  std::vector<TracedStep>* poses = nullptr) {
    using clock = std::chrono::high_resolution_clock;
    RunResult result;
    result.method = method;

    // Preprocessing: the copy in, both KdTrees and both covariance sets. This is
    // the work PCL does inside align() instead.
    const auto t0 = clock::now();
    auto target_pc = toSmallGicp(target);
    auto source_pc = toSmallGicp(source);
    auto target_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
        target_pc, small_gicp::KdTreeBuilderOMP(cfg.num_threads));
    auto source_tree = std::make_shared<small_gicp::KdTree<small_gicp::PointCloud>>(
        source_pc, small_gicp::KdTreeBuilderOMP(cfg.num_threads));
    small_gicp::estimate_covariances_omp(*target_pc, *target_tree, cfg.num_neighbors,
                                         cfg.num_threads);
    small_gicp::estimate_covariances_omp(*source_pc, *source_tree, cfg.num_neighbors,
                                         cfg.num_threads);
    const auto t1 = clock::now();

    small_gicp::Registration<small_gicp::GICPFactor, small_gicp::ParallelReductionOMP,
                             small_gicp::NullFactor, small_gicp::DistanceRejector,
                             RecordingGaussNewton>
        registration;
    registration.rejector.max_dist_sq =
        cfg.max_correspondence_distance * cfg.max_correspondence_distance;
    registration.reduction.num_threads = cfg.num_threads;
    registration.optimizer.max_iterations = cfg.max_iterations;
    registration.optimizer.sink = poses;

    const auto sg_result = registration.align(
        *target_pc, *source_pc, *target_tree,
        Eigen::Isometry3d(initial_guess.cast<double>()));
    const auto t2 = clock::now();

    result.converged = sg_result.converged;
    result.transform = sg_result.T_target_source.matrix().cast<float>();
    result.error = poseError(result.transform, ground_truth);
    result.fitness = sg_result.error;
    // Upstream assigns `iterations = i` inside the loop, so it is a 0-based index
    // of the last step taken, not a count
    result.iterations = poses ? static_cast<int>(poses->size()) - 1
                              : static_cast<int>(sg_result.iterations) + 1;
    result.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.align_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    result.total_ms = result.preprocess_ms + result.align_ms;
    return result;
}

}  // namespace demo
