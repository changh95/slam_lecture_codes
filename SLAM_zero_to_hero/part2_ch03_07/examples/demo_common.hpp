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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

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
 */
template <typename Registration>
void attachIterationLogging(Registration& reg, RegistrationViz* viz,
                            const std::string& method,
                            uint8_t r, uint8_t g, uint8_t b) {
    if (!viz || !viz->active()) return;
    std::function<void(const CloudT&, const pcl::Indices&, const CloudT&,
                       const pcl::Indices&)>
        callback = [viz, method, r, g, b, iteration = 0](
                       const CloudT& intermediate, const pcl::Indices&,
                       const CloudT&, const pcl::Indices&) mutable {
            viz->logIteration(method, iteration++, intermediate, r, g, b);
        };
    reg.registerVisualizationCallback(callback);
}

}  // namespace demo
