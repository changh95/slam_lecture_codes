/**
 * KISS-ICP Demo
 *
 * This example demonstrates the KISS-ICP (Keep It Simple and Stupid ICP) algorithm,
 * a modern, simple LiDAR odometry pipeline that achieves state-of-the-art results.
 *
 * Key innovations of KISS-ICP:
 * - Adaptive threshold for correspondence rejection
 * - Voxel-based efficient map representation
 * - Simple point-to-point ICP with robust design
 * - Motion compensation (deskewing)
 *
 * Reference: Vizzo et al., "KISS-ICP: In Defense of Point-to-Point ICP",
 *            IEEE RA-L 2023
 *
 * Note: This demo has two modes:
 * - WITH_KISS_ICP defined: Uses actual KISS-ICP library
 * - WITHOUT: Demonstrates concepts using PCL implementation
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <vector>
#include <deque>
#include <filesystem>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#ifdef WITH_KISS_ICP
#include <kiss_icp/pipeline/KissICP.hpp>
#endif

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/**
 * Simple KISS-ICP-like odometry implementation using PCL
 * This demonstrates the core concepts without the full KISS-ICP library
 */
class SimpleKISSICP {
public:
    struct Config {
        float max_range = 100.0f;
        float min_range = 3.0f;
        float voxel_size = 0.5f;
        int max_points_per_voxel = 20;
        float initial_threshold = 2.0f;
        float min_motion_th = 0.1f;
        int max_iterations = 50;
        int map_size = 20;  // Number of frames in local map
    };

    SimpleKISSICP(const Config& config = Config()) : config_(config) {
        current_pose_ = Eigen::Matrix4f::Identity();
        adaptive_threshold_ = config_.initial_threshold;
    }

    /**
     * Register a new frame and return the pose
     */
    Eigen::Matrix4f registerFrame(const CloudT::Ptr& raw_cloud) {
        frame_count_++;

        // 1. Preprocess: Range filter and downsample
        CloudT::Ptr processed = preprocess(raw_cloud);

        if (processed->empty()) {
            std::cerr << "Warning: Empty cloud after preprocessing" << std::endl;
            return current_pose_;
        }

        // 2. Initial guess from motion model
        Eigen::Matrix4f initial_guess = predictMotion();

        // 3. Register against local map (if we have one)
        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        if (!local_map_->empty()) {
            delta = registerToMap(processed, initial_guess);
        }

        // 4. Update pose
        current_pose_ = current_pose_ * delta;

        // 5. Update local map
        updateLocalMap(processed, current_pose_);

        // 6. Update motion model
        updateMotionModel(delta);

        // 7. Update adaptive threshold
        updateAdaptiveThreshold(delta);

        return current_pose_;
    }

    /**
     * Get current pose
     */
    Eigen::Matrix4f getPose() const { return current_pose_; }

    /**
     * Get local map
     */
    CloudT::Ptr getLocalMap() const { return local_map_; }

    /**
     * Reset the odometry
     */
    void reset() {
        current_pose_ = Eigen::Matrix4f::Identity();
        last_delta_ = Eigen::Matrix4f::Identity();
        local_map_->clear();
        frame_history_.clear();
        frame_count_ = 0;
        adaptive_threshold_ = config_.initial_threshold;
    }

private:
    Config config_;
    Eigen::Matrix4f current_pose_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f last_delta_ = Eigen::Matrix4f::Identity();
    CloudT::Ptr local_map_{new CloudT};
    std::deque<CloudT::Ptr> frame_history_;
    int frame_count_ = 0;
    float adaptive_threshold_;

    /**
     * Preprocess: range filter and voxel downsample
     */
    CloudT::Ptr preprocess(const CloudT::Ptr& input) {
        CloudT::Ptr filtered(new CloudT);

        // Range filter
        for (const auto& pt : input->points) {
            float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if (range >= config_.min_range && range <= config_.max_range) {
                filtered->push_back(pt);
            }
        }

        // Voxel downsample
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(filtered);
        voxel.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);

        CloudT::Ptr downsampled(new CloudT);
        voxel.filter(*downsampled);

        return downsampled;
    }

    /**
     * Motion prediction (constant velocity model)
     */
    Eigen::Matrix4f predictMotion() {
        return last_delta_;
    }

    /**
     * Register source cloud to local map
     */
    Eigen::Matrix4f registerToMap(const CloudT::Ptr& source,
                                   const Eigen::Matrix4f& initial_guess) {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(source);
        icp.setInputTarget(local_map_);

        icp.setMaximumIterations(config_.max_iterations);
        icp.setMaxCorrespondenceDistance(adaptive_threshold_);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);

        CloudT::Ptr aligned(new CloudT);
        icp.align(*aligned, initial_guess);

        if (icp.hasConverged()) {
            return icp.getFinalTransformation();
        }

        return initial_guess;
    }

    /**
     * Update local map with new frame
     */
    void updateLocalMap(const CloudT::Ptr& frame, const Eigen::Matrix4f& pose) {
        // Transform frame to world coordinates
        CloudT::Ptr transformed(new CloudT);
        pcl::transformPointCloud(*frame, *transformed, pose);

        // Add to history
        frame_history_.push_back(transformed);

        // Remove old frames if needed
        while (frame_history_.size() > static_cast<size_t>(config_.map_size)) {
            frame_history_.pop_front();
        }

        // Rebuild local map from history
        local_map_->clear();
        for (const auto& f : frame_history_) {
            *local_map_ += *f;
        }

        // Downsample the local map
        pcl::VoxelGrid<PointT> voxel;
        voxel.setInputCloud(local_map_);
        voxel.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);

        CloudT::Ptr map_down(new CloudT);
        voxel.filter(*map_down);
        local_map_ = map_down;
    }

    /**
     * Update motion model
     */
    void updateMotionModel(const Eigen::Matrix4f& delta) {
        last_delta_ = delta;
    }

    /**
     * Update adaptive threshold based on motion
     */
    void updateAdaptiveThreshold(const Eigen::Matrix4f& delta) {
        // Compute motion magnitude
        Eigen::Vector3f translation = delta.block<3, 1>(0, 3);
        float motion = translation.norm();

        // Adaptive threshold: larger when moving fast
        float model_deviation = motion / 0.1f;  // Assume 0.1m per frame nominal
        adaptive_threshold_ = std::max(config_.min_motion_th,
                                        config_.initial_threshold * std::min(1.0f, model_deviation));
    }
};

/**
 * Generate synthetic LiDAR sequence (simulating vehicle motion)
 */
std::vector<CloudT::Ptr> generateSequence(int num_frames,
                                           std::vector<Eigen::Matrix4f>& ground_truth_poses) {
    std::vector<CloudT::Ptr> sequence;
    ground_truth_poses.clear();

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_xy(-20.0f, 20.0f);
    std::uniform_real_distribution<float> noise(-0.02f, 0.02f);

    // Create static scene
    CloudT::Ptr scene(new CloudT);

    // Ground plane
    for (int i = 0; i < 2000; ++i) {
        PointT pt;
        pt.x = dist_xy(gen);
        pt.y = dist_xy(gen);
        pt.z = noise(gen);
        scene->push_back(pt);
    }

    // Walls
    for (int i = 0; i < 1000; ++i) {
        PointT pt;
        if (i % 2 == 0) {
            pt.x = 20.0f + noise(gen);
            pt.y = dist_xy(gen);
        } else {
            pt.x = -20.0f + noise(gen);
            pt.y = dist_xy(gen);
        }
        pt.z = std::abs(noise(gen) * 50);
        scene->push_back(pt);
    }

    for (int i = 0; i < 1000; ++i) {
        PointT pt;
        pt.x = dist_xy(gen);
        if (i % 2 == 0) {
            pt.y = 20.0f + noise(gen);
        } else {
            pt.y = -20.0f + noise(gen);
        }
        pt.z = std::abs(noise(gen) * 50);
        scene->push_back(pt);
    }

    // Generate frames along a trajectory
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
    float velocity = 0.5f;  // m/frame

    for (int frame = 0; frame < num_frames; ++frame) {
        // Update position (slight curve)
        float turn_rate = 0.02f * std::sin(frame * 0.1f);
        yaw += turn_rate;
        x += velocity * std::cos(yaw);
        y += velocity * std::sin(yaw);

        // Create pose matrix
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        pose.translation() << x, y, 0.0f;
        pose.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));

        ground_truth_poses.push_back(pose.matrix());

        // Transform scene to sensor frame
        Eigen::Matrix4f sensor_pose = pose.inverse().matrix();
        CloudT::Ptr frame_cloud(new CloudT);
        pcl::transformPointCloud(*scene, *frame_cloud, sensor_pose);

        // Filter by range (simulate sensor)
        CloudT::Ptr filtered(new CloudT);
        for (const auto& pt : frame_cloud->points) {
            float range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
            if (range < 30.0f && range > 2.0f) {
                PointT noisy_pt = pt;
                noisy_pt.x += noise(gen);
                noisy_pt.y += noise(gen);
                noisy_pt.z += noise(gen);
                filtered->push_back(noisy_pt);
            }
        }

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = true;

        sequence.push_back(filtered);
    }

    return sequence;
}

/**
 * Compute trajectory error
 */
void computeTrajectoryError(const std::vector<Eigen::Matrix4f>& estimated,
                             const std::vector<Eigen::Matrix4f>& ground_truth,
                             float& ate, float& rpe) {
    if (estimated.size() != ground_truth.size() || estimated.empty()) {
        ate = -1;
        rpe = -1;
        return;
    }

    // Absolute Trajectory Error (ATE)
    float total_ate = 0.0f;
    for (size_t i = 0; i < estimated.size(); ++i) {
        Eigen::Vector3f t_est = estimated[i].block<3, 1>(0, 3);
        Eigen::Vector3f t_gt = ground_truth[i].block<3, 1>(0, 3);
        total_ate += (t_est - t_gt).norm();
    }
    ate = total_ate / estimated.size();

    // Relative Pose Error (RPE)
    float total_rpe = 0.0f;
    int count = 0;
    for (size_t i = 1; i < estimated.size(); ++i) {
        Eigen::Matrix4f delta_est = estimated[i-1].inverse() * estimated[i];
        Eigen::Matrix4f delta_gt = ground_truth[i-1].inverse() * ground_truth[i];
        Eigen::Matrix4f error = delta_gt.inverse() * delta_est;
        total_rpe += error.block<3, 1>(0, 3).norm();
        count++;
    }
    rpe = (count > 0) ? total_rpe / count : 0.0f;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [sequence_dir]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments    - Run with synthetic sequence" << std::endl;
    std::cout << "  sequence_dir    - Directory containing PCD files" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << std::endl;
    std::cout << "  " << prog_name << " /path/to/lidar/sequence/" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== KISS-ICP Demo ===" << std::endl;
    std::cout << "Simple, modern LiDAR odometry\n" << std::endl;

#ifdef WITH_KISS_ICP
    std::cout << "KISS-ICP library: AVAILABLE" << std::endl;
#else
    std::cout << "KISS-ICP library: NOT AVAILABLE (using PCL-based implementation)" << std::endl;
    std::cout << "To enable KISS-ICP:" << std::endl;
    std::cout << "  1. pip install kiss-icp" << std::endl;
    std::cout << "  2. Or build from source: https://github.com/PRBonn/kiss-icp" << std::endl;
    std::cout << "  3. Rebuild with: cmake -DUSE_KISS_ICP=ON .." << std::endl;
#endif

    std::vector<CloudT::Ptr> sequence;
    std::vector<Eigen::Matrix4f> ground_truth_poses;
    bool has_ground_truth = false;

    if (argc == 1) {
        // Generate synthetic sequence
        std::cout << "\nGenerating synthetic LiDAR sequence..." << std::endl;

        int num_frames = 50;
        sequence = generateSequence(num_frames, ground_truth_poses);
        has_ground_truth = true;

        std::cout << "  Generated " << sequence.size() << " frames" << std::endl;
        std::cout << "  Points per frame: ~" << sequence[0]->size() << std::endl;

    } else if (argc == 2) {
        // Load from directory
        std::string dir_path = argv[1];
        std::cout << "\nLoading sequence from: " << dir_path << std::endl;

        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.path().extension() == ".pcd") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "Error: No PCD files found in " << dir_path << std::endl;
            return -1;
        }

        for (const auto& file : files) {
            CloudT::Ptr cloud(new CloudT);
            if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1) {
                std::cerr << "Warning: Could not read " << file << std::endl;
                continue;
            }
            sequence.push_back(cloud);
        }

        std::cout << "  Loaded " << sequence.size() << " frames" << std::endl;

    } else {
        printUsage(argv[0]);
        return -1;
    }

    // ====================================
    // Run odometry
    // ====================================

    std::cout << "\n=== Running LiDAR Odometry ===" << std::endl;

    SimpleKISSICP::Config config;
    config.voxel_size = 0.5f;
    config.max_range = 50.0f;
    config.min_range = 2.0f;
    config.initial_threshold = 1.0f;
    config.max_iterations = 30;
    config.map_size = 15;

    SimpleKISSICP odometry(config);
    std::vector<Eigen::Matrix4f> estimated_poses;

    double total_time = 0.0;

    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::setw(10) << "Frame"
              << std::setw(12) << "Points"
              << std::setw(15) << "Position"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (size_t i = 0; i < sequence.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix4f pose = odometry.registerFrame(sequence[i]);
        auto end = std::chrono::high_resolution_clock::now();

        double frame_time = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += frame_time;

        estimated_poses.push_back(pose);

        // Print progress every 10 frames
        if (i % 10 == 0 || i == sequence.size() - 1) {
            Eigen::Vector3f pos = pose.block<3, 1>(0, 3);
            std::cout << std::setw(10) << i
                      << std::setw(12) << sequence[i]->size()
                      << std::setw(15) << std::fixed << std::setprecision(2)
                      << "(" << pos.x() << "," << pos.y() << ")"
                      << std::setw(12) << std::setprecision(1) << frame_time << std::endl;
        }
    }

    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(1)
              << total_time << " ms" << std::endl;
    std::cout << "Average time per frame: " << total_time / sequence.size() << " ms" << std::endl;
    std::cout << "FPS: " << std::setprecision(1)
              << 1000.0 / (total_time / sequence.size()) << std::endl;

    // ====================================
    // Evaluate if ground truth available
    // ====================================

    if (has_ground_truth) {
        std::cout << "\n=== Trajectory Evaluation ===" << std::endl;

        float ate, rpe;
        computeTrajectoryError(estimated_poses, ground_truth_poses, ate, rpe);

        std::cout << "  Absolute Trajectory Error (ATE): " << std::fixed << std::setprecision(4)
                  << ate << " m" << std::endl;
        std::cout << "  Relative Pose Error (RPE): " << std::setprecision(4)
                  << rpe << " m" << std::endl;

        // Print trajectory comparison
        std::cout << "\n  Trajectory comparison (first 10 poses):" << std::endl;
        std::cout << std::setw(10) << "Frame"
                  << std::setw(25) << "Estimated (x,y)"
                  << std::setw(25) << "Ground Truth (x,y)"
                  << std::setw(12) << "Error" << std::endl;

        for (size_t i = 0; i < std::min(size_t(10), estimated_poses.size()); ++i) {
            Eigen::Vector3f est = estimated_poses[i].block<3, 1>(0, 3);
            Eigen::Vector3f gt = ground_truth_poses[i].block<3, 1>(0, 3);
            float err = (est - gt).norm();

            std::cout << std::setw(10) << i
                      << std::setw(25) << std::fixed << std::setprecision(2)
                      << "(" << est.x() << ", " << est.y() << ")"
                      << std::setw(25)
                      << "(" << gt.x() << ", " << gt.y() << ")"
                      << std::setw(12) << std::setprecision(3) << err << std::endl;
        }
    }

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "KISS-ICP key features:" << std::endl;
    std::cout << "  1. Simple point-to-point ICP (surprisingly effective)" << std::endl;
    std::cout << "  2. Adaptive threshold based on motion" << std::endl;
    std::cout << "  3. Voxel-based local map for efficiency" << std::endl;
    std::cout << "  4. No feature extraction needed" << std::endl;
    std::cout << "  5. Works well on various LiDAR sensors" << std::endl;
    std::cout << std::endl;
    std::cout << "Typical configuration:" << std::endl;
    std::cout << "  - voxel_size: 0.5-1.0m (outdoor), 0.1-0.25m (indoor)" << std::endl;
    std::cout << "  - max_range: 100m (outdoor), 30m (indoor)" << std::endl;
    std::cout << "  - min_range: 3-5m (to filter self-reflections)" << std::endl;
    std::cout << std::endl;

    return 0;
}
