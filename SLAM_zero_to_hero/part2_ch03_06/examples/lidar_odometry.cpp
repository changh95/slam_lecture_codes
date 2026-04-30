/**
 * @file lidar_odometry.cpp
 * @brief LiDAR Odometry using ICP for sequential scan registration
 *
 * This example demonstrates:
 * - LidarOdometry class for sequential scan-to-scan matching
 * - Voxel grid downsampling for efficient ICP
 * - Global pose accumulation
 * - Trajectory output for visualization
 *
 * The LidarOdometry class maintains the previous scan and global pose,
 * computing relative transformations between consecutive scans using ICP.
 *
 * Usage: ./lidar_odometry /path/to/velodyne/ [--visualize]
 *        ./lidar_odometry --generate [--visualize]
 */

#include <iostream>
#include <fstream>
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

namespace fs = std::filesystem;

// Type aliases
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

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
                   double max_correspondence_distance = 1.0,
                   double transformation_epsilon = 1e-6)
        : voxel_size_(voxel_size)
        , max_iterations_(max_iterations)
        , max_correspondence_distance_(max_correspondence_distance)
        , transformation_epsilon_(transformation_epsilon)
        , frame_count_(0)
    {
        // Initialize global pose to identity
        global_pose_ = Eigen::Matrix4f::Identity();

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
    }

    /**
     * @brief Process a new point cloud
     * @param cloud Input point cloud
     * @return true if registration succeeded
     */
    bool processCloud(const PointCloudT::Ptr& cloud)
    {
        if (cloud->empty())
        {
            std::cerr << "Warning: Empty cloud received\n";
            return false;
        }

        // Preprocess: downsample
        PointCloudT::Ptr filtered(new PointCloudT);
        voxel_filter_.setInputCloud(cloud);
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
            std::cout << "Frame 1: Initialized (no previous frame)\n";
            return true;
        }

        // ICP alignment: current (source) -> previous (target)
        icp_.setInputSource(filtered);
        icp_.setInputTarget(prev_cloud_);

        PointCloudT::Ptr aligned(new PointCloudT);
        icp_.align(*aligned);

        if (!icp_.hasConverged())
        {
            std::cerr << "Warning: ICP did not converge for frame " << frame_count_ << "\n";
            // Keep previous cloud, don't update pose
            return false;
        }

        double fitness = icp_.getFitnessScore();

        // Check if fitness is acceptable
        if (fitness > fitness_threshold_)
        {
            std::cerr << "Warning: High fitness score (" << fitness << ") for frame "
                      << frame_count_ << "\n";
            // Still update but flag it
        }

        // Get relative transformation (source -> target)
        // This is the motion from current to previous
        // We need inverse: previous to current (forward motion)
        Eigen::Matrix4f relative_transform = icp_.getFinalTransformation();

        // Update global pose: T_global_new = T_global_old * T_relative
        global_pose_ = global_pose_ * relative_transform;

        // Store trajectory point
        trajectory_.push_back(global_pose_);

        // Update previous cloud
        prev_cloud_ = filtered;

        // Print progress
        Eigen::Vector3f position = global_pose_.block<3, 1>(0, 3);
        std::cout << "Frame " << frame_count_ << ": "
                  << "Fitness=" << std::fixed << std::setprecision(4) << fitness
                  << " | Position=(" << std::setprecision(2)
                  << position.x() << ", " << position.y() << ", " << position.z() << ")\n";

        return true;
    }

    /**
     * @brief Get current global pose
     */
    Eigen::Matrix4f getPose() const
    {
        return global_pose_;
    }

    /**
     * @brief Get trajectory as vector of poses
     */
    std::vector<Eigen::Matrix4f> getTrajectory() const
    {
        return trajectory_;
    }

    /**
     * @brief Get current position
     */
    Eigen::Vector3f getPosition() const
    {
        return global_pose_.block<3, 1>(0, 3);
    }

    /**
     * @brief Get frame count
     */
    int getFrameCount() const
    {
        return frame_count_;
    }

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

        file.close();
        std::cout << "Trajectory saved to: " << filename << " (" << trajectory_.size() << " poses)\n";
        return true;
    }

    /**
     * @brief Save trajectory to TUM format (for evo tools)
     */
    bool saveTrajectoryTUM(const std::string& filename) const
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

            // Extract position
            float x = pose(0, 3);
            float y = pose(1, 3);
            float z = pose(2, 3);

            // Extract rotation matrix and convert to quaternion
            Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
            Eigen::Quaternionf q(R);

            // TUM format: timestamp tx ty tz qx qy qz qw
            file << std::fixed << std::setprecision(6) << static_cast<double>(i)
                 << " " << x << " " << y << " " << z
                 << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        }

        file.close();
        std::cout << "Trajectory (TUM format) saved to: " << filename << "\n";
        return true;
    }

    /**
     * @brief Reset odometry
     */
    void reset()
    {
        global_pose_ = Eigen::Matrix4f::Identity();
        prev_cloud_ = nullptr;
        trajectory_.clear();
        frame_count_ = 0;
    }

private:
    // ICP components
    pcl::IterativeClosestPoint<PointT, PointT> icp_;
    pcl::VoxelGrid<PointT> voxel_filter_;

    // State
    PointCloudT::Ptr prev_cloud_;
    Eigen::Matrix4f global_pose_;
    std::vector<Eigen::Matrix4f> trajectory_;

    // Parameters
    double voxel_size_;
    int max_iterations_;
    double max_correspondence_distance_;
    double transformation_epsilon_;
    double fitness_threshold_ = 0.5;

    int frame_count_;
};

/**
 * @brief Generate synthetic LiDAR scans for testing
 */
std::vector<PointCloudT::Ptr> generateSyntheticScans(int num_scans = 20)
{
    std::vector<PointCloudT::Ptr> scans;

    // Simulate a corridor environment
    auto generateCorridorScan = [](float sensor_x, float sensor_y, float sensor_yaw) {
        PointCloudT::Ptr scan(new PointCloudT);

        // Corridor walls at y = +/- 2m
        // Length from x = -10 to x = 20
        int points_per_wall = 200;

        for (int i = 0; i < points_per_wall; ++i)
        {
            float x = -10.0f + 30.0f * static_cast<float>(i) / points_per_wall;

            // Left wall
            {
                PointT p;
                float wall_x = x - sensor_x;
                float wall_y = 2.0f - sensor_y;
                // Rotate by sensor yaw
                p.x = wall_x * cos(-sensor_yaw) - wall_y * sin(-sensor_yaw);
                p.y = wall_x * sin(-sensor_yaw) + wall_y * cos(-sensor_yaw);
                p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
                // Add noise
                p.x += 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
                p.y += 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
                scan->points.push_back(p);
            }

            // Right wall
            {
                PointT p;
                float wall_x = x - sensor_x;
                float wall_y = -2.0f - sensor_y;
                p.x = wall_x * cos(-sensor_yaw) - wall_y * sin(-sensor_yaw);
                p.y = wall_x * sin(-sensor_yaw) + wall_y * cos(-sensor_yaw);
                p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
                p.x += 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
                p.y += 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
                scan->points.push_back(p);
            }
        }

        // Floor points
        for (int i = 0; i < 300; ++i)
        {
            PointT p;
            float floor_x = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 8.0f;
            float floor_y = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
            p.x = floor_x * cos(-sensor_yaw) - floor_y * sin(-sensor_yaw);
            p.y = floor_x * sin(-sensor_yaw) + floor_y * cos(-sensor_yaw);
            p.z = 0.0f + 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
            scan->points.push_back(p);
        }

        scan->width = scan->points.size();
        scan->height = 1;
        scan->is_dense = true;

        return scan;
    };

    // Generate scans along a trajectory
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
    float dx = 0.5f;  // Move 0.5m per scan
    float dyaw = 0.02f;  // Small rotation per scan

    for (int i = 0; i < num_scans; ++i)
    {
        scans.push_back(generateCorridorScan(x, y, yaw));

        // Update pose (simulate forward motion with slight curve)
        x += dx * cos(yaw);
        y += dx * sin(yaw);
        yaw += dyaw;
    }

    return scans;
}

/**
 * @brief Load point clouds from directory (KITTI format)
 */
std::vector<std::string> getPointCloudFiles(const std::string& directory)
{
    std::vector<std::string> files;

    for (const auto& entry : fs::directory_iterator(directory))
    {
        std::string path = entry.path().string();
        std::string ext = entry.path().extension().string();

        // Support .pcd and .bin (KITTI format)
        if (ext == ".pcd" || ext == ".bin")
        {
            files.push_back(path);
        }
    }

    // Sort files by name
    std::sort(files.begin(), files.end());

    return files;
}

/**
 * @brief Load KITTI binary point cloud
 */
PointCloudT::Ptr loadKITTIBin(const std::string& filename)
{
    PointCloudT::Ptr cloud(new PointCloudT);

    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file: " << filename << "\n";
        return cloud;
    }

    // KITTI format: x, y, z, intensity (4 floats per point)
    while (input.good() && !input.eof())
    {
        float x, y, z, intensity;
        input.read(reinterpret_cast<char*>(&x), sizeof(float));
        input.read(reinterpret_cast<char*>(&y), sizeof(float));
        input.read(reinterpret_cast<char*>(&z), sizeof(float));
        input.read(reinterpret_cast<char*>(&intensity), sizeof(float));

        if (input.good())
        {
            PointT p;
            p.x = x;
            p.y = y;
            p.z = z;
            cloud->points.push_back(p);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

int main(int argc, char** argv)
{
    std::cout << "=== LiDAR Odometry using ICP ===\n\n";

    bool generate_mode = false;
    bool visualize = false;
    std::string input_dir;

    // Parse arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--generate" || arg == "-g")
        {
            generate_mode = true;
        }
        else if (arg == "--visualize" || arg == "-v")
        {
            visualize = true;
        }
        else if (arg[0] != '-')
        {
            input_dir = arg;
        }
    }

    if (!generate_mode && input_dir.empty())
    {
        std::cout << "Usage: " << argv[0] << " /path/to/velodyne/ [--visualize]\n";
        std::cout << "       " << argv[0] << " --generate [--visualize]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  /path/to/velodyne/  Directory containing .pcd or .bin files\n";
        std::cout << "  --generate, -g      Generate synthetic LiDAR scans\n";
        std::cout << "  --visualize, -v     (reserved for future visualization)\n";
        return 0;
    }

    // Create odometry instance
    LidarOdometry odometry(
        0.2,    // voxel size (m)
        30,     // max iterations
        1.0,    // max correspondence distance (m)
        1e-6    // transformation epsilon
    );

    if (generate_mode)
    {
        std::cout << "\n--- Generating Synthetic LiDAR Scans ---\n";

        int num_scans = 30;
        std::vector<PointCloudT::Ptr> scans = generateSyntheticScans(num_scans);

        std::cout << "Generated " << scans.size() << " synthetic scans\n";
        std::cout << "\n--- Processing Scans ---\n";

        for (size_t i = 0; i < scans.size(); ++i)
        {
            odometry.processCloud(scans[i]);
        }
    }
    else
    {
        std::cout << "Loading point clouds from: " << input_dir << "\n";

        std::vector<std::string> files = getPointCloudFiles(input_dir);

        if (files.empty())
        {
            std::cerr << "Error: No point cloud files found in " << input_dir << "\n";
            return -1;
        }

        std::cout << "Found " << files.size() << " point cloud files\n";
        std::cout << "\n--- Processing Scans ---\n";

        for (const auto& file : files)
        {
            PointCloudT::Ptr cloud;

            // Check file extension
            std::string ext = fs::path(file).extension().string();
            if (ext == ".bin")
            {
                cloud = loadKITTIBin(file);
            }
            else
            {
                cloud.reset(new PointCloudT);
                if (pcl::io::loadPCDFile<PointT>(file, *cloud) == -1)
                {
                    std::cerr << "Warning: Could not load " << file << "\n";
                    continue;
                }
            }

            if (!cloud->empty())
            {
                odometry.processCloud(cloud);
            }
        }
    }

    // Print summary
    std::cout << "\n=== Odometry Summary ===\n";
    std::cout << "Total frames processed: " << odometry.getFrameCount() << "\n";

    Eigen::Vector3f final_pos = odometry.getPosition();
    std::cout << "Final position: ("
              << std::fixed << std::setprecision(3)
              << final_pos.x() << ", "
              << final_pos.y() << ", "
              << final_pos.z() << ") m\n";

    // Compute total distance traveled
    auto trajectory = odometry.getTrajectory();
    double total_distance = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i)
    {
        Eigen::Vector3f p1 = trajectory[i - 1].block<3, 1>(0, 3);
        Eigen::Vector3f p2 = trajectory[i].block<3, 1>(0, 3);
        total_distance += (p2 - p1).norm();
    }
    std::cout << "Total distance traveled: " << std::fixed << std::setprecision(2)
              << total_distance << " m\n";

    // Save trajectory
    odometry.saveTrajectory("trajectory_kitti.txt");
    odometry.saveTrajectoryTUM("trajectory_tum.txt");

    std::cout << "\n=== Done ===\n";
    std::cout << "Trajectory files saved for visualization with:\n";
    std::cout << "  - evo_traj tum trajectory_tum.txt --plot\n";
    std::cout << "  - evo_traj kitti trajectory_kitti.txt --plot\n";

    return 0;
}
