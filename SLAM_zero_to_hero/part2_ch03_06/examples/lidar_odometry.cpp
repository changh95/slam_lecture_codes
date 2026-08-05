/**
 * @file lidar_odometry.cpp
 * @brief LiDAR Odometry using ICP for sequential scan registration
 *
 * This example demonstrates:
 * - LidarOdometry class for sequential scan-to-scan matching
 * - Range cropping and voxel grid downsampling for efficient ICP
 * - Constant-velocity motion prediction as the ICP initial guess
 * - Global pose accumulation and trajectory export
 * - Evaluation against KITTI ground-truth poses
 *
 * Point to a KITTI odometry sequence directory and the demo finds the velodyne
 * scans, the calibration, and the ground-truth poses by itself:
 *
 *   ./lidar_odometry ~/data/kitti_vo_slam/extracted/dataset/sequences/04
 *
 * KITTI ground truth is given in the left-camera frame, so it is mapped into
 * the velodyne frame with the calib Tr matrix before it is compared against the
 * estimated trajectory.
 *
 * Usage: ./lidar_odometry <kitti_sequence_dir> [--max-frames N]
 *        ./lidar_odometry /path/to/velodyne/    [--max-frames N]
 *        ./lidar_odometry --generate
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include "demo_common.hpp"

namespace fs = std::filesystem;

using PointT = demo::PointT;
using PointCloudT = demo::CloudT;

/**
 * @class LidarOdometry
 * @brief Sequential scan registration for LiDAR odometry using ICP
 */
class LidarOdometry
{
public:
    /**
     * @brief Constructor with configurable parameters
     */
    LidarOdometry(double voxel_size = 0.2,
                   int max_iterations = 30,
                   double max_correspondence_distance = 2.0,
                   double transformation_epsilon = 1e-6)
        : voxel_size_(voxel_size)
        , max_iterations_(max_iterations)
        , max_correspondence_distance_(max_correspondence_distance)
        , transformation_epsilon_(transformation_epsilon)
        , frame_count_(0)
    {
        // Initialize global pose to identity
        global_pose_ = Eigen::Matrix4f::Identity();

        // Previous frame-to-frame motion, reused as the next initial guess
        last_motion_ = Eigen::Matrix4f::Identity();

        // Configure ICP
        icp_.setMaximumIterations(max_iterations_);
        icp_.setTransformationEpsilon(transformation_epsilon_);
        icp_.setMaxCorrespondenceDistance(max_correspondence_distance_);
        icp_.setEuclideanFitnessEpsilon(1e-6);

        // Configure voxel grid filter
        voxel_filter_.setLeafSize(voxel_size_, voxel_size_, voxel_size_);

        std::cout << "LidarOdometry initialized with:\n";
        std::cout << "  Voxel size: " << voxel_size_ << " m\n";
        std::cout << "  Max iterations: " << max_iterations_ << "\n";
        std::cout << "  Max correspondence distance: " << max_correspondence_distance_ << " m\n";
        std::cout << "  Transformation epsilon: " << transformation_epsilon_ << "\n";
        std::cout << "  Range crop: " << min_range_ << " - " << max_range_ << " m\n";
        std::cout << "  Motion prediction: " << (predict_motion_ ? "constant velocity" : "off")
                  << "\n";
    }

    /** @brief Keep only points within [min_range, max_range] of the sensor */
    void setRangeCrop(double min_range, double max_range)
    {
        min_range_ = min_range;
        max_range_ = max_range;
    }

    /** @brief Use the previous frame-to-frame motion as the ICP initial guess */
    void setPredictMotion(bool enable)
    {
        predict_motion_ = enable;
    }

    /** @brief Print one progress line every n frames (0 disables) */
    void setProgressInterval(int n)
    {
        progress_interval_ = n;
    }

    /**
     * @brief Process a new point cloud
     * @param cloud Input point cloud
     * @return true if registration succeeded
     *
     * On failure the pose is still advanced by the predicted motion so that the
     * trajectory keeps one entry per input scan, which the ground-truth
     * comparison relies on.
     */
    bool processCloud(const PointCloudT::Ptr& cloud)
    {
        if (cloud->empty())
        {
            std::cerr << "Warning: Empty cloud received\n";
            return false;
        }

        // Preprocess: crop by range, then downsample.
        // Returns close to the sensor are the ego vehicle itself and move with
        // it, so they bias scan-to-scan matching towards zero motion.
        PointCloudT::Ptr cropped = cropByRange(cloud);

        PointCloudT::Ptr filtered(new PointCloudT);
        voxel_filter_.setInputCloud(cropped);
        voxel_filter_.filter(*filtered);

        if (filtered->size() < 100)
        {
            std::cerr << "Warning: Too few points after filtering (" << filtered->size() << ")\n";
            return false;
        }

        frame_count_++;

        // First frame: just store it
        if (prev_cloud_ == nullptr)
        {
            prev_cloud_ = filtered;
            trajectory_.push_back(global_pose_);
            std::cout << "Frame 1: Initialized (" << filtered->size()
                      << " points after preprocessing)\n";
            return true;
        }

        // ICP alignment: current (source) -> previous (target)
        icp_.setInputSource(filtered);
        icp_.setInputTarget(prev_cloud_);

        // A scan-to-scan ICP started from identity has to close the whole
        // inter-frame motion, which at 10 Hz and highway speed is ~2 m. Seeding
        // it with the previous motion leaves only the acceleration as residual.
        const Eigen::Matrix4f guess = predict_motion_ ? last_motion_
                                                      : Eigen::Matrix4f::Identity();

        PointCloudT::Ptr aligned(new PointCloudT);
        icp_.align(*aligned, guess);

        if (!icp_.hasConverged())
        {
            std::cerr << "Warning: ICP did not converge for frame " << frame_count_
                      << " - coasting on predicted motion\n";
            ++failed_frames_;

            global_pose_ = global_pose_ * last_motion_;
            trajectory_.push_back(global_pose_);
            prev_cloud_ = filtered;
            return false;
        }

        const double fitness = icp_.getFitnessScore();

        if (fitness > fitness_threshold_)
        {
            std::cerr << "Warning: High fitness score (" << fitness << ") for frame "
                      << frame_count_ << "\n";
            ++high_fitness_frames_;
        }

        // ICP maps source points into the target frame, and the source is the
        // current scan while the target is the previous one, so the result is
        // already T_previous_current - exactly what pose accumulation needs.
        const Eigen::Matrix4f relative_transform = icp_.getFinalTransformation();

        // Update global pose: T_global_new = T_global_old * T_relative
        global_pose_ = global_pose_ * relative_transform;
        last_motion_ = relative_transform;

        // Store trajectory point
        trajectory_.push_back(global_pose_);

        // Update previous cloud
        prev_cloud_ = filtered;

        if (progress_interval_ > 0 && frame_count_ % progress_interval_ == 0)
        {
            const Eigen::Vector3f position = global_pose_.block<3, 1>(0, 3);
            std::cout << "Frame " << std::setw(5) << frame_count_
                      << ": pts=" << std::setw(6) << filtered->size()
                      << " fitness=" << std::fixed << std::setprecision(4) << fitness
                      << " pos=(" << std::setprecision(2) << std::setw(8) << position.x()
                      << "," << std::setw(8) << position.y()
                      << "," << std::setw(7) << position.z() << ")\n";
        }

        return true;
    }

    Eigen::Matrix4f getPose() const { return global_pose_; }
    std::vector<Eigen::Matrix4f> getTrajectory() const { return trajectory_; }
    Eigen::Vector3f getPosition() const { return global_pose_.block<3, 1>(0, 3); }
    int getFrameCount() const { return frame_count_; }
    int getFailedFrames() const { return failed_frames_; }
    int getHighFitnessFrames() const { return high_fitness_frames_; }

    /**
     * @brief Save trajectory to file (KITTI format)
     */
    bool saveTrajectory(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filename << "\n";
            return false;
        }

        for (const auto& pose : trajectory_)
        {
            // KITTI format: 12 values (3x4 matrix row-major)
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    file << std::fixed << std::setprecision(9) << pose(i, j);
                    if (i < 2 || j < 3)
                        file << " ";
                }
            }
            file << "\n";
        }

        std::cout << "Trajectory saved to: " << filename << " (" << trajectory_.size() << " poses)\n";
        return true;
    }

    /**
     * @brief Save trajectory to TUM format (for evo tools)
     * @param timestamps Per-frame times in seconds; frame indices are used when empty
     */
    bool saveTrajectoryTUM(const std::string& filename,
                           const std::vector<double>& timestamps = {}) const
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file for writing: " << filename << "\n";
            return false;
        }

        for (size_t i = 0; i < trajectory_.size(); ++i)
        {
            const auto& pose = trajectory_[i];

            // Extract rotation matrix and convert to quaternion
            const Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
            const Eigen::Quaternionf q(R);

            const double stamp = (i < timestamps.size()) ? timestamps[i]
                                                         : static_cast<double>(i);

            // TUM format: timestamp tx ty tz qx qy qz qw
            file << std::fixed << std::setprecision(6) << stamp
                 << " " << pose(0, 3) << " " << pose(1, 3) << " " << pose(2, 3)
                 << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }

        std::cout << "Trajectory (TUM format) saved to: " << filename << "\n";
        return true;
    }

    /**
     * @brief Reset odometry
     */
    void reset()
    {
        global_pose_ = Eigen::Matrix4f::Identity();
        last_motion_ = Eigen::Matrix4f::Identity();
        prev_cloud_ = nullptr;
        trajectory_.clear();
        frame_count_ = 0;
        failed_frames_ = 0;
        high_fitness_frames_ = 0;
    }

private:
    /**
     * @brief Drop points closer than min_range_ or further than max_range_
     */
    PointCloudT::Ptr cropByRange(const PointCloudT::Ptr& cloud) const
    {
        PointCloudT::Ptr out(new PointCloudT);
        out->points.reserve(cloud->size());

        const double min_sq = min_range_ * min_range_;
        const double max_sq = max_range_ * max_range_;

        for (const auto& p : cloud->points)
        {
            const double range_sq = static_cast<double>(p.x) * p.x +
                                    static_cast<double>(p.y) * p.y +
                                    static_cast<double>(p.z) * p.z;
            if (range_sq >= min_sq && range_sq <= max_sq)
            {
                out->points.push_back(p);
            }
        }

        out->width = out->points.size();
        out->height = 1;
        out->is_dense = true;
        return out;
    }

    // ICP components
    pcl::IterativeClosestPoint<PointT, PointT> icp_;
    pcl::VoxelGrid<PointT> voxel_filter_;

    // State
    PointCloudT::Ptr prev_cloud_;
    Eigen::Matrix4f global_pose_;
    Eigen::Matrix4f last_motion_;
    std::vector<Eigen::Matrix4f> trajectory_;

    // Parameters
    double voxel_size_;
    int max_iterations_;
    double max_correspondence_distance_;
    double transformation_epsilon_;
    double fitness_threshold_ = 1.0;
    double min_range_ = 2.5;
    double max_range_ = 80.0;
    bool predict_motion_ = true;
    int progress_interval_ = 10;

    int frame_count_;
    int failed_frames_ = 0;
    int high_fitness_frames_ = 0;
};

// ============================================================================
// KITTI odometry dataset handling
// ============================================================================

/**
 * @brief Files that make up one KITTI odometry sequence
 *
 * A sequence directory looks like
 *   sequences/04/{velodyne/,calib.txt,times.txt}
 *   poses/04.txt
 * with the ground-truth poses living outside the sequence folder.
 */
struct KittiSequence
{
    std::string velodyne_dir;
    std::string calib_file;   // empty when unavailable
    std::string poses_file;   // empty when unavailable (sequences 11-21 have none)
    std::string times_file;   // empty when unavailable
    std::string name;
};

/**
 * @brief Work out the sequence layout from either a sequence dir or a velodyne dir
 */
KittiSequence resolveKittiSequence(const std::string& path)
{
    KittiSequence seq;

    fs::path root(path);
    // Tolerate a trailing slash, which would otherwise make filename() empty
    if (root.filename().empty())
    {
        root = root.parent_path();
    }

    // Given a sequence directory: descend into velodyne/
    fs::path velodyne = root / "velodyne";
    if (fs::is_directory(velodyne))
    {
        seq.velodyne_dir = velodyne.string();
        seq.name = root.filename().string();
    }
    else
    {
        // Given the velodyne directory itself
        seq.velodyne_dir = root.string();
        root = (root.filename() == "velodyne") ? root.parent_path() : root;
        seq.name = root.filename().string();
    }

    const fs::path calib = root / "calib.txt";
    if (fs::exists(calib)) seq.calib_file = calib.string();

    const fs::path times = root / "times.txt";
    if (fs::exists(times)) seq.times_file = times.string();

    // dataset/sequences/NN -> dataset/poses/NN.txt
    const fs::path poses = root.parent_path().parent_path() / "poses" / (seq.name + ".txt");
    if (fs::exists(poses)) seq.poses_file = poses.string();

    return seq;
}

/**
 * @brief Read KITTI ground-truth poses: one 3x4 row-major matrix per line
 */
std::vector<Eigen::Matrix4f> loadKittiPoses(const std::string& filename)
{
    std::vector<Eigen::Matrix4f> poses;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        return poses;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty()) continue;

        std::istringstream ss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        bool ok = true;

        for (int i = 0; i < 3 && ok; ++i)
        {
            for (int j = 0; j < 4 && ok; ++j)
            {
                ok = static_cast<bool>(ss >> pose(i, j));
            }
        }

        if (ok)
        {
            poses.push_back(pose);
        }
    }

    return poses;
}

/**
 * @brief Read the velodyne-to-camera transform (Tr) from a KITTI calib.txt
 * @return identity when the file has no Tr line
 */
Eigen::Matrix4f loadVelodyneToCamera(const std::string& filename)
{
    Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();

    std::ifstream file(filename);
    if (!file.is_open())
    {
        return Tr;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.rfind("Tr:", 0) != 0) continue;

        std::istringstream ss(line.substr(3));
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                ss >> Tr(i, j);
            }
        }
        break;
    }

    return Tr;
}

/**
 * @brief Read per-frame timestamps (seconds) from a KITTI times.txt
 */
std::vector<double> loadKittiTimes(const std::string& filename)
{
    std::vector<double> times;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        return times;
    }

    double t;
    while (file >> t)
    {
        times.push_back(t);
    }
    return times;
}

/**
 * @brief Express camera-frame ground-truth poses in the velodyne frame
 *
 * KITTI poses are T_cam0_cami. The odometry runs on velodyne scans, so the
 * comparable quantity is T_velo0_veloi = Tr^-1 * T_cam0_cami * Tr.
 */
std::vector<Eigen::Matrix4f> posesToLidarFrame(const std::vector<Eigen::Matrix4f>& cam_poses,
                                               const Eigen::Matrix4f& Tr)
{
    const Eigen::Matrix4f Tr_inv = Tr.inverse();

    std::vector<Eigen::Matrix4f> out;
    out.reserve(cam_poses.size());
    for (const auto& pose : cam_poses)
    {
        out.push_back(Tr_inv * pose * Tr);
    }
    return out;
}

/**
 * @brief Total length of a trajectory
 */
double pathLength(const std::vector<Eigen::Matrix4f>& trajectory)
{
    double length = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i)
    {
        length += (trajectory[i].block<3, 1>(0, 3) -
                   trajectory[i - 1].block<3, 1>(0, 3)).norm();
    }
    return length;
}

/**
 * @brief Compare the estimated trajectory against ground truth
 *
 * Both trajectories start at the identity, so absolute errors can be read off
 * directly without a Umeyama alignment step.
 */
void evaluateTrajectory(const std::vector<Eigen::Matrix4f>& estimated,
                        const std::vector<Eigen::Matrix4f>& ground_truth)
{
    const size_t n = std::min(estimated.size(), ground_truth.size());
    if (n < 2)
    {
        std::cout << "\nNot enough overlapping poses to evaluate.\n";
        return;
    }

    double sum_sq = 0.0;
    double max_err = 0.0;
    double sum_sq_rot = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        const double err = (estimated[i].block<3, 1>(0, 3) -
                            ground_truth[i].block<3, 1>(0, 3)).norm();
        sum_sq += err * err;
        max_err = std::max(max_err, err);

        const double rot = demo::poseError(estimated[i], ground_truth[i]).rotation_deg;
        sum_sq_rot += rot * rot;
    }

    const double rmse = std::sqrt(sum_sq / static_cast<double>(n));
    const double rmse_rot = std::sqrt(sum_sq_rot / static_cast<double>(n));

    const Eigen::Vector3f final_est = estimated[n - 1].block<3, 1>(0, 3);
    const Eigen::Vector3f final_gt = ground_truth[n - 1].block<3, 1>(0, 3);
    const double final_err = (final_est - final_gt).norm();
    const double final_rot = demo::poseError(estimated[n - 1], ground_truth[n - 1]).rotation_deg;

    // Split the final error along and across the ground-truth heading: a scale
    // error in the estimated motion shows up along track, while accumulated
    // heading error shows up across it.
    const Eigen::Vector3f heading = ground_truth[n - 1].block<3, 1>(0, 0);
    const Eigen::Vector3f delta = final_est - final_gt;
    const double along_track = delta.dot(heading);
    const double cross_track = (delta - along_track * heading).norm();

    const std::vector<Eigen::Matrix4f> gt_prefix(ground_truth.begin(),
                                                 ground_truth.begin() + n);
    const double gt_length = pathLength(gt_prefix);

    std::cout << "\n=== Ground Truth Comparison ===\n";
    std::cout << "Poses compared: " << n << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Ground-truth path length: " << gt_length << " m\n";
    std::cout << "ATE (translation RMSE):   " << rmse << " m\n";
    std::cout << "Rotation RMSE:            " << rmse_rot << " deg\n";
    std::cout << "Max position error:       " << max_err << " m\n";
    std::cout << "Final position error:     " << final_err << " m";
    if (gt_length > 0.0)
    {
        std::cout << "  (" << std::setprecision(2) << 100.0 * final_err / gt_length
                  << " % of path length)";
    }
    std::cout << "\n" << std::setprecision(3);
    std::cout << "  along track:            " << along_track << " m\n";
    std::cout << "  across track:           " << cross_track << " m\n";
    std::cout << "Final rotation error:     " << final_rot << " deg\n";

    std::cout << "\nNote: scan-to-scan ICP has no loop closure and no local map, so the\n";
    std::cout << "error accumulates monotonically. Typically the along-track error stays\n";
    std::cout << "small - the per-frame translation is accurate - while a fraction of a\n";
    std::cout << "degree of yaw error per frame integrates into a much larger\n";
    std::cout << "across-track offset. Heading, not distance, is what drifts.\n";
    std::cout << "part2_ch03_07 compares GICP / NDT / TEASER++ on the same data.\n";
}

/**
 * @brief Save ground-truth poses in KITTI format, for side-by-side plots
 */
bool saveReferenceTrajectory(const std::vector<Eigen::Matrix4f>& poses,
                             const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        return false;
    }

    for (const auto& pose : poses)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                file << std::fixed << std::setprecision(9) << pose(i, j);
                if (i < 2 || j < 3) file << " ";
            }
        }
        file << "\n";
    }

    std::cout << "Ground truth saved to: " << filename << " (" << poses.size() << " poses)\n";
    return true;
}

// ============================================================================
// Synthetic fallback
// ============================================================================

/**
 * @brief Generate synthetic LiDAR scans for testing
 *
 * Two parallel walls and a flat floor are a degenerate target for ICP: sliding
 * along the corridor leaves the scan unchanged, so forward motion is
 * unobservable and the estimate under-shoots. Pillars along the walls break
 * that symmetry and give the demo a trajectory worth checking.
 */
std::vector<PointCloudT::Ptr> generateSyntheticScans(int num_scans,
                                                     std::vector<Eigen::Matrix4f>& gt_poses)
{
    std::vector<PointCloudT::Ptr> scans;
    gt_poses.clear();

    // Simulate a corridor environment
    auto generateCorridorScan = [](float sensor_x, float sensor_y, float sensor_yaw) {
        PointCloudT::Ptr scan(new PointCloudT);

        // Express a world point in the sensor frame
        auto addPoint = [&](float world_x, float world_y, float z, float noise) {
            const float dx = world_x - sensor_x;
            const float dy = world_y - sensor_y;

            PointT p;
            p.x = dx * cos(-sensor_yaw) - dy * sin(-sensor_yaw);
            p.y = dx * sin(-sensor_yaw) + dy * cos(-sensor_yaw);
            p.z = z;
            p.x += noise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
            p.y += noise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
            scan->points.push_back(p);
        };

        // Corridor walls at y = +/- 2m, length from x = -10 to x = 20
        const int points_per_wall = 200;

        for (int i = 0; i < points_per_wall; ++i)
        {
            const float x = -10.0f + 30.0f * static_cast<float>(i) / points_per_wall;

            for (float wall_side : {2.0f, -2.0f})
            {
                addPoint(x, wall_side,
                         static_cast<float>(rand()) / RAND_MAX * 2.0f, 0.02f);
            }
        }

        // Pillars every 5 m, alternating sides, to make along-corridor motion observable
        for (int k = 0; k <= 6; ++k)
        {
            const float pillar_x = -10.0f + 5.0f * static_cast<float>(k);
            const float pillar_y = (k % 2 == 0) ? 1.6f : -1.6f;

            for (int i = 0; i < 40; ++i)
            {
                const float angle = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
                addPoint(pillar_x + 0.15f * cos(angle),
                         pillar_y + 0.15f * sin(angle),
                         static_cast<float>(rand()) / RAND_MAX * 2.5f, 0.01f);
            }
        }

        // Floor points, sampled around the sensor as a real scanner would.
        // A flat floor cannot constrain in-plane motion at all: the patch looks
        // the same wherever the sensor is, so these points only ever pull the
        // estimate towards zero motion. They are kept because that bias is real
        // - it is why the KITTI path below crops the close range - but the
        // count is deliberately low so the pillars dominate the solution.
        for (int i = 0; i < 100; ++i)
        {
            const float floor_x = sensor_x + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 8.0f;
            const float floor_y = sensor_y + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
            addPoint(floor_x, floor_y,
                     0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f), 0.0f);
        }

        scan->width = scan->points.size();
        scan->height = 1;
        scan->is_dense = true;

        return scan;
    };

    // Generate scans along a trajectory
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
    const float dx = 0.5f;     // Move 0.5 m per scan
    const float dyaw = 0.02f;  // Small rotation per scan

    for (int i = 0; i < num_scans; ++i)
    {
        scans.push_back(generateCorridorScan(x, y, yaw));

        // The scans are built in the sensor frame, so the pose of scan i
        // relative to scan 0 is the sensor motion since the start
        gt_poses.push_back(demo::makeTransform(x, y, 0.0f, 0.0f, 0.0f, yaw));

        // Update pose (simulate forward motion with slight curve)
        x += dx * cos(yaw);
        y += dx * sin(yaw);
        yaw += dyaw;
    }

    // Referenced to the first scan, which the odometry takes as the origin
    const Eigen::Matrix4f first_inv = gt_poses.front().inverse();
    for (auto& pose : gt_poses)
    {
        pose = first_inv * pose;
    }

    return scans;
}

/**
 * @brief List the .bin / .pcd scans in a directory, sorted by filename
 */
std::vector<std::string> getPointCloudFiles(const std::string& directory)
{
    std::vector<std::string> files;

    for (const auto& entry : fs::directory_iterator(directory))
    {
        const std::string ext = entry.path().extension().string();

        // Support .pcd and .bin (KITTI format)
        if (ext == ".pcd" || ext == ".bin")
        {
            files.push_back(entry.path().string());
        }
    }

    // KITTI scans are zero-padded, so lexicographic order is frame order
    std::sort(files.begin(), files.end());

    return files;
}

void printUsage(const char* program)
{
    std::cout << "Usage: " << program << " <kitti_sequence_dir> [--max-frames N]\n";
    std::cout << "       " << program << " /path/to/velodyne/    [--max-frames N]\n";
    std::cout << "       " << program << " --generate\n";
    std::cout << "\nOptions:\n";
    std::cout << "  <kitti_sequence_dir>  KITTI odometry sequence, e.g.\n";
    std::cout << "                        .../dataset/sequences/04\n";
    std::cout << "                        velodyne/, calib.txt and ../../poses/NN.txt\n";
    std::cout << "                        are picked up automatically\n";
    std::cout << "  --max-frames N        Stop after N scans (default: all)\n";
    std::cout << "  --voxel S             Voxel leaf size in meters (default: 0.2).\n";
    std::cout << "                        Drives both speed and drift: on sequence 04\n";
    std::cout << "                        0.2 m drifts ~1 %, 0.5 m drifts ~4 %\n";
    std::cout << "  --no-prediction       Start each ICP from identity instead of\n";
    std::cout << "                        the previous frame-to-frame motion\n";
    std::cout << "  --generate, -g        Run on synthetic corridor scans\n";
}

int main(int argc, char** argv)
{
    std::cout << "=== LiDAR Odometry using ICP ===\n\n";

    bool generate_mode = false;
    bool predict_motion = true;
    int max_frames = -1;
    // Voxel size dominates the accuracy of scan-to-scan ICP on KITTI: on
    // sequence 04 the final-position drift is ~1 % of path length at 0.2 m,
    // ~2.5 % at 0.3 m and ~4 % at 0.5 m. Larger leaves are faster but throw
    // away the vertical structure that constrains forward motion.
    double voxel_size = 0.2;
    std::string input_dir;

    // Parse arguments
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "--generate" || arg == "-g")
        {
            generate_mode = true;
        }
        else if (arg == "--no-prediction")
        {
            predict_motion = false;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if ((arg == "--max-frames" || arg == "-n") && i + 1 < argc)
        {
            max_frames = std::atoi(argv[++i]);
        }
        else if (arg == "--voxel" && i + 1 < argc)
        {
            voxel_size = std::atof(argv[++i]);
        }
        else if (arg[0] != '-')
        {
            input_dir = arg;
        }
    }

    if (!generate_mode && input_dir.empty())
    {
        printUsage(argv[0]);
        return 0;
    }

    // Create odometry instance
    LidarOdometry odometry(
        voxel_size,  // voxel size (m)
        30,          // max iterations
        2.0,         // max correspondence distance (m)
        1e-6         // transformation epsilon
    );
    odometry.setPredictMotion(predict_motion);

    std::vector<Eigen::Matrix4f> gt_poses;
    std::vector<double> timestamps;

    if (generate_mode)
    {
        std::cout << "\n--- Generating Synthetic LiDAR Scans ---\n";

        // The synthetic corridor is a few meters wide, so the KITTI-scale range
        // crop would discard all of it
        odometry.setRangeCrop(0.0, 1000.0);
        odometry.setProgressInterval(1);

        const std::vector<PointCloudT::Ptr> scans = generateSyntheticScans(30, gt_poses);

        std::cout << "Generated " << scans.size() << " synthetic scans\n";
        std::cout << "\n--- Processing Scans ---\n";

        for (const auto& scan : scans)
        {
            odometry.processCloud(scan);
        }
    }
    else
    {
        const KittiSequence seq = resolveKittiSequence(input_dir);

        if (!fs::is_directory(seq.velodyne_dir))
        {
            std::cerr << "Error: Not a directory: " << seq.velodyne_dir << "\n";
            return -1;
        }

        std::vector<std::string> files = getPointCloudFiles(seq.velodyne_dir);

        if (files.empty())
        {
            std::cerr << "Error: No .bin or .pcd scans found in " << seq.velodyne_dir << "\n";
            return -1;
        }

        std::cout << "--- KITTI Sequence " << seq.name << " ---\n";
        std::cout << "Scans:  " << files.size() << " in " << seq.velodyne_dir << "\n";

        if (max_frames > 0 && static_cast<size_t>(max_frames) < files.size())
        {
            files.resize(max_frames);
            std::cout << "Limited to first " << files.size() << " scans (--max-frames)\n";
        }

        // Ground truth, mapped from the camera frame into the velodyne frame
        if (!seq.poses_file.empty())
        {
            const std::vector<Eigen::Matrix4f> cam_poses = loadKittiPoses(seq.poses_file);

            Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();
            if (!seq.calib_file.empty())
            {
                Tr = loadVelodyneToCamera(seq.calib_file);
                std::cout << "Calib:  " << seq.calib_file << " (Tr loaded)\n";
            }
            else
            {
                std::cout << "Calib:  not found - comparing in the camera frame\n";
            }

            gt_poses = posesToLidarFrame(cam_poses, Tr);
            std::cout << "Poses:  " << gt_poses.size() << " from " << seq.poses_file << "\n";
        }
        else
        {
            std::cout << "Poses:  none (sequences 11-21 have no public ground truth)\n";
        }

        if (!seq.times_file.empty())
        {
            timestamps = loadKittiTimes(seq.times_file);
        }

        std::cout << "\n--- Processing Scans ---\n";

        for (const auto& file : files)
        {
            PointCloudT::Ptr cloud = demo::loadCloud(file);

            if (!cloud || cloud->empty())
            {
                std::cerr << "Warning: Could not load " << file << "\n";
                continue;
            }

            odometry.processCloud(cloud);
        }
    }

    // Print summary
    std::cout << "\n=== Odometry Summary ===\n";
    std::cout << "Total frames processed: " << odometry.getFrameCount() << "\n";
    std::cout << "ICP failures (coasted): " << odometry.getFailedFrames() << "\n";
    std::cout << "High-fitness warnings:  " << odometry.getHighFitnessFrames() << "\n";

    const Eigen::Vector3f final_pos = odometry.getPosition();
    std::cout << "Final position: ("
              << std::fixed << std::setprecision(3)
              << final_pos.x() << ", "
              << final_pos.y() << ", "
              << final_pos.z() << ") m\n";

    const std::vector<Eigen::Matrix4f> trajectory = odometry.getTrajectory();
    std::cout << "Estimated path length: " << std::fixed << std::setprecision(2)
              << pathLength(trajectory) << " m\n";

    // Compare against ground truth when available
    if (!gt_poses.empty())
    {
        evaluateTrajectory(trajectory, gt_poses);

        const size_t n = std::min(trajectory.size(), gt_poses.size());
        saveReferenceTrajectory({gt_poses.begin(), gt_poses.begin() + n},
                                "trajectory_gt_kitti.txt");
    }

    // Save trajectory
    odometry.saveTrajectory("trajectory_kitti.txt");
    odometry.saveTrajectoryTUM("trajectory_tum.txt", timestamps);

    std::cout << "\n=== Done ===\n";
    std::cout << "Plot the trajectory with:\n";
    std::cout << "  evo_traj kitti trajectory_kitti.txt --plot\n";
    if (!gt_poses.empty())
    {
        std::cout << "Compare against ground truth with:\n";
        std::cout << "  evo_ape kitti trajectory_gt_kitti.txt trajectory_kitti.txt "
                     "-va --plot\n";
    }

    return 0;
}
