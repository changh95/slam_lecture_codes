/**
 * Normal Distributions Transform (NDT) Demo
 *
 * This example demonstrates the Normal Distributions Transform algorithm,
 * which represents the target point cloud as a grid of Gaussian distributions
 * and maximizes the likelihood of source points under this representation.
 *
 * Key concepts covered:
 * - NDT algorithm overview and grid representation
 * - Parameter tuning (resolution, step size)
 * - Using initial guesses for faster convergence
 * - Comparing NDT performance with different settings
 *
 * Reference: Biber & Strasser, "The Normal Distributions Transform", IROS 2003
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/**
 * Generate a synthetic point cloud simulating a typical LiDAR scene
 */
CloudT::Ptr generateSyntheticCloud(int num_points = 20000) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_xy(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-0.1f, 0.1f);
    std::uniform_real_distribution<float> noise(0.0f, 1.0f);

    // Ground plane
    for (int i = 0; i < num_points / 2; ++i) {
        PointT pt;
        pt.x = dist_xy(gen);
        pt.y = dist_xy(gen);
        pt.z = dist_z(gen);
        cloud->push_back(pt);
    }

    // Building-like structures
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        float building = noise(gen);

        if (building < 0.5f) {
            // First building
            pt.x = 5.0f + dist_z(gen);
            pt.y = -3.0f + noise(gen) * 4.0f;
            pt.z = noise(gen) * 3.0f;
        } else {
            // Second building
            pt.x = -5.0f + dist_z(gen);
            pt.y = 2.0f + noise(gen) * 3.0f;
            pt.z = noise(gen) * 2.5f;
        }
        cloud->push_back(pt);
    }

    // Some poles/vertical structures
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        float pole_x = (noise(gen) - 0.5f) * 16.0f;
        float pole_y = (noise(gen) - 0.5f) * 16.0f;

        pt.x = pole_x + dist_z(gen) * 0.1f;
        pt.y = pole_y + dist_z(gen) * 0.1f;
        pt.z = noise(gen) * 2.0f;
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Apply transformation to a point cloud
 */
CloudT::Ptr transformCloud(const CloudT::Ptr& input,
                            const Eigen::Matrix4f& transform) {
    CloudT::Ptr output(new CloudT);
    pcl::transformPointCloud(*input, *output, transform);
    return output;
}

/**
 * Create transformation matrix from translation and rotation
 */
Eigen::Matrix4f createTransform(float tx, float ty, float tz,
                                 float roll, float pitch, float yaw) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << tx, ty, tz;
    transform.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    return transform.matrix();
}

/**
 * Compute transformation error
 */
void computeTransformError(const Eigen::Matrix4f& estimated,
                            const Eigen::Matrix4f& ground_truth,
                            float& translation_error,
                            float& rotation_error_deg) {
    Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
    Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
    translation_error = (t_est - t_gt).norm();

    Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
    Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
    Eigen::Matrix3f R_error = R_est.transpose() * R_gt;
    float trace = R_error.trace();
    float cos_angle = std::max(-1.0f, std::min(1.0f, (trace - 1.0f) / 2.0f));
    rotation_error_deg = std::acos(cos_angle) * 180.0f / M_PI;
}

/**
 * Run NDT registration
 */
struct NDTResult {
    bool converged;
    int iterations;
    double fitness_score;
    float trans_error;
    float rot_error;
    double time_ms;
    Eigen::Matrix4f transform;
};

NDTResult runNDT(const CloudT::Ptr& source,
                  const CloudT::Ptr& target,
                  const Eigen::Matrix4f& ground_truth,
                  float resolution,
                  float step_size,
                  const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity()) {

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // Set input clouds
    ndt.setInputSource(source);
    ndt.setInputTarget(target);

    // NDT parameters
    ndt.setResolution(resolution);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(0.01);
    ndt.setMaximumIterations(50);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    ndt.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    NDTResult result;
    result.converged = ndt.hasConverged();
    result.fitness_score = ndt.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = ndt.getFinalTransformation();
    result.iterations = ndt.getFinalNumIteration();

    computeTransformError(result.transform, ground_truth, result.trans_error, result.rot_error);

    return result;
}

/**
 * Study the effect of resolution parameter
 */
void studyResolution(const CloudT::Ptr& source,
                      const CloudT::Ptr& target,
                      const Eigen::Matrix4f& ground_truth) {
    std::cout << "\n=== NDT Resolution Parameter Study ===" << std::endl;
    std::cout << "(Cell size affects accuracy vs speed tradeoff)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(12) << "Resolution"
              << std::setw(12) << "Converged"
              << std::setw(12) << "Iters"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    std::vector<float> resolutions = {0.5f, 1.0f, 2.0f, 3.0f, 5.0f};

    for (float res : resolutions) {
        NDTResult result = runNDT(source, target, ground_truth, res, 0.1f);

        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << res
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(12) << result.iterations
                  << std::setw(15) << std::setprecision(4) << result.trans_error
                  << std::setw(15) << std::setprecision(4) << result.rot_error
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - Smaller resolution = more cells = higher accuracy but slower" << std::endl;
    std::cout << "  - Larger resolution = fewer cells = faster but less accurate" << std::endl;
    std::cout << "  - Typical values: 0.5-2.0m for outdoor, 0.1-0.5m for indoor" << std::endl;
}

/**
 * Study the effect of step size parameter
 */
void studyStepSize(const CloudT::Ptr& source,
                    const CloudT::Ptr& target,
                    const Eigen::Matrix4f& ground_truth) {
    std::cout << "\n=== NDT Step Size Parameter Study ===" << std::endl;
    std::cout << "(Newton optimization step size affects convergence)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(12) << "Step Size"
              << std::setw(12) << "Converged"
              << std::setw(12) << "Iters"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    std::vector<float> step_sizes = {0.01f, 0.05f, 0.1f, 0.5f, 1.0f};

    for (float step : step_sizes) {
        NDTResult result = runNDT(source, target, ground_truth, 1.0f, step);

        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << step
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(12) << result.iterations
                  << std::setw(15) << std::setprecision(4) << result.trans_error
                  << std::setw(15) << std::setprecision(4) << result.rot_error
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - Too small step = slow convergence, may get stuck" << std::endl;
    std::cout << "  - Too large step = may overshoot optimum" << std::endl;
    std::cout << "  - Typical values: 0.05-0.5" << std::endl;
}

/**
 * Demonstrate the importance of initial guess
 */
void studyInitialGuess(const CloudT::Ptr& source,
                        const CloudT::Ptr& target,
                        const Eigen::Matrix4f& ground_truth) {
    std::cout << "\n=== Initial Guess Study ===" << std::endl;
    std::cout << "(How initial guess affects NDT convergence)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(25) << "Initial Guess"
              << std::setw(12) << "Converged"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    // Different initial guesses
    std::vector<std::pair<std::string, Eigen::Matrix4f>> guesses;

    guesses.push_back({"Identity", Eigen::Matrix4f::Identity()});
    guesses.push_back({"Close (0.1m off)", createTransform(0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.02f)});
    guesses.push_back({"Medium (0.5m off)", createTransform(0.5f, 0.3f, 0.1f, 0.0f, 0.0f, 0.1f)});
    guesses.push_back({"Far (2m off)", createTransform(2.0f, 1.0f, 0.5f, 0.1f, 0.1f, 0.3f)});

    for (const auto& [name, guess] : guesses) {
        NDTResult result = runNDT(source, target, ground_truth, 1.0f, 0.1f, guess);

        std::cout << std::setw(25) << name
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(15) << std::fixed << std::setprecision(4) << result.trans_error
                  << std::setw(15) << std::setprecision(4) << result.rot_error
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - Good initial guess significantly improves results" << std::endl;
    std::cout << "  - In odometry, use previous pose estimate as initial guess" << std::endl;
    std::cout << "  - For loop closure, may need global registration first" << std::endl;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.pcd target.pcd] [resolution]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments           - Use synthetic point clouds" << std::endl;
    std::cout << "  source target          - Use provided PCD files" << std::endl;
    std::cout << "  source target res      - Use PCD files with custom resolution" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << std::endl;
    std::cout << "  " << prog_name << " scan1.pcd scan2.pcd" << std::endl;
    std::cout << "  " << prog_name << " scan1.pcd scan2.pcd 1.0" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Normal Distributions Transform (NDT) Demo ===" << std::endl;
    std::cout << "Grid-based registration with Gaussian distributions\n" << std::endl;

    CloudT::Ptr source(new CloudT);
    CloudT::Ptr target(new CloudT);
    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();
    float resolution = 1.0f;
    bool use_synthetic = true;

    if (argc == 1) {
        // Generate synthetic data
        std::cout << "Generating synthetic point cloud (LiDAR-like scene)..." << std::endl;

        target = generateSyntheticCloud(20000);

        // Ground truth transformation
        float tx = 0.5f, ty = 0.3f, tz = 0.1f;
        float roll = 0.02f, pitch = 0.03f, yaw = 0.1f;

        ground_truth = createTransform(tx, ty, tz, roll, pitch, yaw);
        source = transformCloud(target, ground_truth);

        std::cout << "  Source points: " << source->size() << std::endl;
        std::cout << "  Target points: " << target->size() << std::endl;
        std::cout << "  Ground truth translation: [" << tx << ", " << ty << ", " << tz << "]" << std::endl;
        std::cout << "  Ground truth rotation (rad): [" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;

    } else if (argc >= 3) {
        use_synthetic = false;

        std::cout << "Loading point clouds from files..." << std::endl;

        if (pcl::io::loadPCDFile<PointT>(argv[1], *source) == -1) {
            std::cerr << "Error: Could not read source file: " << argv[1] << std::endl;
            return -1;
        }
        if (pcl::io::loadPCDFile<PointT>(argv[2], *target) == -1) {
            std::cerr << "Error: Could not read target file: " << argv[2] << std::endl;
            return -1;
        }

        if (argc >= 4) {
            resolution = std::stof(argv[3]);
        }

        std::cout << "  Source: " << argv[1] << " (" << source->size() << " points)" << std::endl;
        std::cout << "  Target: " << argv[2] << " (" << target->size() << " points)" << std::endl;
        std::cout << "  Resolution: " << resolution << " m" << std::endl;

    } else {
        printUsage(argv[0]);
        return -1;
    }

    // Downsample if clouds are very large
    if (source->size() > 100000 || target->size() > 100000) {
        std::cout << "\nDownsampling large clouds..." << std::endl;

        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(0.2f, 0.2f, 0.2f);

        CloudT::Ptr source_down(new CloudT);
        CloudT::Ptr target_down(new CloudT);

        voxel.setInputCloud(source);
        voxel.filter(*source_down);

        voxel.setInputCloud(target);
        voxel.filter(*target_down);

        std::cout << "  Source: " << source->size() << " -> " << source_down->size() << std::endl;
        std::cout << "  Target: " << target->size() << " -> " << target_down->size() << std::endl;

        source = source_down;
        target = target_down;
    }

    // ====================================
    // Basic NDT registration
    // ====================================

    std::cout << "\n=== Basic NDT Registration ===" << std::endl;

    NDTResult basic_result = runNDT(source, target, ground_truth, resolution, 0.1f);

    std::cout << "  Converged: " << (basic_result.converged ? "YES" : "NO") << std::endl;
    std::cout << "  Iterations: " << basic_result.iterations << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << basic_result.fitness_score << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4)
              << basic_result.trans_error << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4)
              << basic_result.rot_error << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << basic_result.time_ms << " ms" << std::endl;

    // ====================================
    // Parameter studies (only for synthetic)
    // ====================================

    if (use_synthetic) {
        studyResolution(source, target, ground_truth);
        studyStepSize(source, target, ground_truth);
        studyInitialGuess(source, target, ground_truth);
    }

    // ====================================
    // Final transformation
    // ====================================

    std::cout << "\n=== Final NDT Transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << basic_result.transform << std::endl;

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "NDT key characteristics:" << std::endl;
    std::cout << "  1. Represents target as grid of Gaussian distributions" << std::endl;
    std::cout << "  2. No explicit correspondences needed (unlike ICP)" << std::endl;
    std::cout << "  3. Smooth cost function - good for optimization" << std::endl;
    std::cout << "  4. Fast - O(n) complexity for source points" << std::endl;
    std::cout << std::endl;
    std::cout << "Recommended parameter settings:" << std::endl;
    std::cout << "  - Outdoor LiDAR: resolution=1.0-2.0m, step_size=0.1" << std::endl;
    std::cout << "  - Indoor/dense: resolution=0.2-0.5m, step_size=0.05" << std::endl;
    std::cout << "  - Always provide good initial guess when possible" << std::endl;
    std::cout << std::endl;

    return 0;
}
