/**
 * Point Cloud Registration Method Comparison
 *
 * This example provides a comprehensive comparison of different point cloud
 * registration methods: ICP, GICP, and NDT.
 *
 * Comparison criteria:
 * - Accuracy (translation and rotation error)
 * - Speed (processing time)
 * - Robustness to initial pose error
 * - Behavior with different point cloud densities
 *
 * This helps you choose the right method for your application.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/**
 * Result structure for each registration method
 */
struct RegistrationResult {
    std::string method_name;
    bool converged;
    double fitness_score;
    float translation_error;
    float rotation_error_deg;
    double time_ms;
    Eigen::Matrix4f transform;
};

/**
 * Generate a synthetic point cloud with various structures
 */
CloudT::Ptr generateSyntheticCloud(int num_points = 15000) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_xy(-8.0f, 8.0f);
    std::uniform_real_distribution<float> noise(-0.02f, 0.02f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    // Ground plane (40% of points)
    int ground_pts = num_points * 40 / 100;
    for (int i = 0; i < ground_pts; ++i) {
        PointT pt;
        pt.x = dist_xy(gen);
        pt.y = dist_xy(gen);
        pt.z = noise(gen);
        cloud->push_back(pt);
    }

    // Building facades (30% of points)
    int wall_pts = num_points * 30 / 100;
    for (int i = 0; i < wall_pts; ++i) {
        PointT pt;
        float wall = uniform(gen);
        if (wall < 0.25f) {
            pt.x = 8.0f + noise(gen);
            pt.y = dist_xy(gen);
            pt.z = uniform(gen) * 4.0f;
        } else if (wall < 0.5f) {
            pt.x = -8.0f + noise(gen);
            pt.y = dist_xy(gen);
            pt.z = uniform(gen) * 3.5f;
        } else if (wall < 0.75f) {
            pt.x = dist_xy(gen);
            pt.y = 8.0f + noise(gen);
            pt.z = uniform(gen) * 3.0f;
        } else {
            pt.x = dist_xy(gen);
            pt.y = -8.0f + noise(gen);
            pt.z = uniform(gen) * 4.5f;
        }
        cloud->push_back(pt);
    }

    // Poles and thin structures (15% of points)
    int pole_pts = num_points * 15 / 100;
    std::uniform_real_distribution<float> pole_pos(-7.0f, 7.0f);
    for (int i = 0; i < pole_pts; ++i) {
        PointT pt;
        // Create several poles at fixed positions
        int pole_id = i % 6;
        float px = (pole_id % 3 - 1) * 4.0f;
        float py = (pole_id / 3 - 0.5f) * 8.0f;

        pt.x = px + noise(gen) * 0.05f;
        pt.y = py + noise(gen) * 0.05f;
        pt.z = uniform(gen) * 3.0f;
        cloud->push_back(pt);
    }

    // Random objects/clutter (15% of points)
    int obj_pts = num_points * 15 / 100;
    for (int i = 0; i < obj_pts; ++i) {
        PointT pt;
        float cx = (uniform(gen) - 0.5f) * 10.0f;
        float cy = (uniform(gen) - 0.5f) * 10.0f;
        float size = 0.3f + uniform(gen) * 0.5f;

        pt.x = cx + (uniform(gen) - 0.5f) * size;
        pt.y = cy + (uniform(gen) - 0.5f) * size;
        pt.z = uniform(gen) * size;
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Create transformation matrix
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
void computeError(const Eigen::Matrix4f& estimated,
                  const Eigen::Matrix4f& ground_truth,
                  float& trans_error, float& rot_error_deg) {
    Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
    Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
    trans_error = (t_est - t_gt).norm();

    Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
    Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
    Eigen::Matrix3f R_err = R_est.transpose() * R_gt;
    float trace = R_err.trace();
    float cos_angle = std::max(-1.0f, std::min(1.0f, (trace - 1.0f) / 2.0f));
    rot_error_deg = std::acos(cos_angle) * 180.0f / M_PI;
}

/**
 * Run ICP registration
 */
RegistrationResult runICP(const CloudT::Ptr& source,
                           const CloudT::Ptr& target,
                           const Eigen::Matrix4f& ground_truth,
                           const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity()) {
    RegistrationResult result;
    result.method_name = "ICP";

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(2.0);
    icp.setEuclideanFitnessEpsilon(1e-6);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = icp.hasConverged();
    result.fitness_score = icp.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = icp.getFinalTransformation();

    computeError(result.transform, ground_truth,
                 result.translation_error, result.rotation_error_deg);

    return result;
}

/**
 * Run GICP registration
 */
RegistrationResult runGICP(const CloudT::Ptr& source,
                            const CloudT::Ptr& target,
                            const Eigen::Matrix4f& ground_truth,
                            const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity()) {
    RegistrationResult result;
    result.method_name = "GICP";

    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    gicp.setMaximumIterations(50);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(2.0);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.setCorrespondenceRandomness(20);
    gicp.setMaximumOptimizerIterations(20);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    gicp.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = gicp.hasConverged();
    result.fitness_score = gicp.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = gicp.getFinalTransformation();

    computeError(result.transform, ground_truth,
                 result.translation_error, result.rotation_error_deg);

    return result;
}

/**
 * Run NDT registration
 */
RegistrationResult runNDT(const CloudT::Ptr& source,
                           const CloudT::Ptr& target,
                           const Eigen::Matrix4f& ground_truth,
                           float resolution = 1.0f,
                           const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity()) {
    RegistrationResult result;
    result.method_name = "NDT";

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;
    ndt.setInputSource(source);
    ndt.setInputTarget(target);
    ndt.setResolution(resolution);
    ndt.setStepSize(0.1);
    ndt.setTransformationEpsilon(0.01);
    ndt.setMaximumIterations(50);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    ndt.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = ndt.hasConverged();
    result.fitness_score = ndt.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = ndt.getFinalTransformation();

    computeError(result.transform, ground_truth,
                 result.translation_error, result.rotation_error_deg);

    return result;
}

/**
 * Print results table
 */
void printResults(const std::vector<RegistrationResult>& results) {
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::setw(10) << "Method"
              << std::setw(12) << "Converged"
              << std::setw(15) << "Fitness"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::setw(10) << r.method_name
                  << std::setw(12) << (r.converged ? "YES" : "NO")
                  << std::setw(15) << std::fixed << std::setprecision(6) << r.fitness_score
                  << std::setw(15) << std::setprecision(4) << r.translation_error
                  << std::setw(15) << std::setprecision(4) << r.rotation_error_deg
                  << std::setw(12) << std::setprecision(1) << r.time_ms << std::endl;
    }
    std::cout << std::string(90, '-') << std::endl;
}

/**
 * Test 1: Basic accuracy comparison
 */
void testBasicAccuracy(const CloudT::Ptr& target) {
    std::cout << "\n=== Test 1: Basic Accuracy Comparison ===" << std::endl;
    std::cout << "Small transformation, identity initial guess\n" << std::endl;

    // Small transformation
    float tx = 0.3f, ty = 0.2f, tz = 0.1f;
    float roll = 0.03f, pitch = 0.02f, yaw = 0.05f;

    Eigen::Matrix4f gt = createTransform(tx, ty, tz, roll, pitch, yaw);

    CloudT::Ptr source(new CloudT);
    pcl::transformPointCloud(*target, *source, gt);

    std::cout << "Ground truth: t=[" << tx << "," << ty << "," << tz << "], "
              << "r=[" << roll << "," << pitch << "," << yaw << "] rad" << std::endl;

    std::vector<RegistrationResult> results;
    results.push_back(runICP(source, target, gt));
    results.push_back(runGICP(source, target, gt));
    results.push_back(runNDT(source, target, gt));

    printResults(results);
}

/**
 * Test 2: Robustness to initial pose error
 */
void testInitialPoseRobustness(const CloudT::Ptr& target) {
    std::cout << "\n=== Test 2: Robustness to Initial Pose Error ===" << std::endl;
    std::cout << "Testing with increasingly wrong initial guesses\n" << std::endl;

    // True transformation
    float tx = 0.5f, ty = 0.3f, tz = 0.1f;
    float roll = 0.05f, pitch = 0.03f, yaw = 0.1f;

    Eigen::Matrix4f gt = createTransform(tx, ty, tz, roll, pitch, yaw);

    CloudT::Ptr source(new CloudT);
    pcl::transformPointCloud(*target, *source, gt);

    // Different initial guess errors
    std::vector<std::pair<std::string, Eigen::Matrix4f>> initial_guesses = {
        {"Perfect", gt},
        {"Close (0.2m)", createTransform(tx + 0.2f, ty + 0.1f, tz, roll, pitch, yaw + 0.05f)},
        {"Medium (0.5m)", createTransform(tx + 0.5f, ty + 0.3f, tz + 0.1f, roll, pitch, yaw + 0.15f)},
        {"Far (1.0m)", createTransform(tx + 1.0f, ty + 0.5f, tz + 0.2f, roll + 0.1f, pitch, yaw + 0.3f)},
        {"Identity", Eigen::Matrix4f::Identity()}
    };

    for (const auto& [name, guess] : initial_guesses) {
        std::cout << "\nInitial guess: " << name << std::endl;

        std::vector<RegistrationResult> results;
        results.push_back(runICP(source, target, gt, guess));
        results.push_back(runGICP(source, target, gt, guess));
        results.push_back(runNDT(source, target, gt, 1.0f, guess));

        printResults(results);
    }
}

/**
 * Test 3: Point cloud density comparison
 */
void testDensityEffect() {
    std::cout << "\n=== Test 3: Effect of Point Cloud Density ===" << std::endl;
    std::cout << "Testing with different downsampling levels\n" << std::endl;

    CloudT::Ptr full_cloud = generateSyntheticCloud(30000);

    float tx = 0.4f, ty = 0.25f, tz = 0.1f;
    float roll = 0.04f, pitch = 0.03f, yaw = 0.08f;
    Eigen::Matrix4f gt = createTransform(tx, ty, tz, roll, pitch, yaw);

    std::vector<float> voxel_sizes = {0.1f, 0.2f, 0.5f, 1.0f};

    for (float voxel_size : voxel_sizes) {
        // Downsample
        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(voxel_size, voxel_size, voxel_size);

        CloudT::Ptr target(new CloudT);
        voxel.setInputCloud(full_cloud);
        voxel.filter(*target);

        CloudT::Ptr source(new CloudT);
        pcl::transformPointCloud(*target, *source, gt);

        std::cout << "\nVoxel size: " << voxel_size << "m, Points: " << target->size() << std::endl;

        std::vector<RegistrationResult> results;
        results.push_back(runICP(source, target, gt));
        results.push_back(runGICP(source, target, gt));
        results.push_back(runNDT(source, target, gt, voxel_size * 2));

        printResults(results);
    }
}

/**
 * Test 4: Speed comparison with varying cloud sizes
 */
void testSpeedScaling() {
    std::cout << "\n=== Test 4: Speed Scaling with Cloud Size ===" << std::endl;
    std::cout << "Processing time vs number of points\n" << std::endl;

    float tx = 0.3f, ty = 0.2f, tz = 0.1f;
    float roll = 0.03f, pitch = 0.02f, yaw = 0.05f;
    Eigen::Matrix4f gt = createTransform(tx, ty, tz, roll, pitch, yaw);

    std::vector<int> cloud_sizes = {1000, 5000, 10000, 20000, 40000};

    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::setw(12) << "Points"
              << std::setw(15) << "ICP (ms)"
              << std::setw(15) << "GICP (ms)"
              << std::setw(15) << "NDT (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int size : cloud_sizes) {
        CloudT::Ptr target = generateSyntheticCloud(size);
        CloudT::Ptr source(new CloudT);
        pcl::transformPointCloud(*target, *source, gt);

        auto icp_result = runICP(source, target, gt);
        auto gicp_result = runGICP(source, target, gt);
        auto ndt_result = runNDT(source, target, gt);

        std::cout << std::setw(12) << size
                  << std::setw(15) << std::fixed << std::setprecision(1) << icp_result.time_ms
                  << std::setw(15) << gicp_result.time_ms
                  << std::setw(15) << ndt_result.time_ms << std::endl;
    }
    std::cout << std::string(60, '-') << std::endl;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.pcd target.pcd]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments    - Run all tests with synthetic data" << std::endl;
    std::cout << "  source target   - Run basic comparison with provided files" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Point Cloud Registration Method Comparison ===" << std::endl;
    std::cout << "ICP vs GICP vs NDT\n" << std::endl;

    if (argc == 1) {
        // Run full test suite with synthetic data
        std::cout << "Generating synthetic point cloud..." << std::endl;
        CloudT::Ptr target = generateSyntheticCloud(15000);
        std::cout << "Generated " << target->size() << " points\n" << std::endl;

        testBasicAccuracy(target);
        testInitialPoseRobustness(target);
        testDensityEffect();
        testSpeedScaling();

    } else if (argc == 3) {
        // Compare on provided files
        CloudT::Ptr source(new CloudT);
        CloudT::Ptr target(new CloudT);

        if (pcl::io::loadPCDFile<PointT>(argv[1], *source) == -1) {
            std::cerr << "Error: Could not read " << argv[1] << std::endl;
            return -1;
        }
        if (pcl::io::loadPCDFile<PointT>(argv[2], *target) == -1) {
            std::cerr << "Error: Could not read " << argv[2] << std::endl;
            return -1;
        }

        std::cout << "Source: " << argv[1] << " (" << source->size() << " points)" << std::endl;
        std::cout << "Target: " << argv[2] << " (" << target->size() << " points)" << std::endl;

        // Assume identity as ground truth
        Eigen::Matrix4f gt = Eigen::Matrix4f::Identity();

        std::cout << "\n=== Registration Results ===" << std::endl;
        std::cout << "(Using identity as reference - actual errors may differ)" << std::endl;

        std::vector<RegistrationResult> results;
        results.push_back(runICP(source, target, gt));
        results.push_back(runGICP(source, target, gt));
        results.push_back(runNDT(source, target, gt));

        printResults(results);

        // Print transformations
        for (const auto& r : results) {
            std::cout << "\n" << r.method_name << " Transformation:" << std::endl;
            std::cout << std::fixed << std::setprecision(6) << r.transform << std::endl;
        }

    } else {
        printUsage(argv[0]);
        return -1;
    }

    // ====================================
    // Summary and Recommendations
    // ====================================

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "=== Method Selection Guide ===" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\n| Scenario                    | Recommended Method | Notes            |" << std::endl;
    std::cout << "|-----------------------------|-------------------|------------------|" << std::endl;
    std::cout << "| Real-time odometry          | NDT               | Fast, good enough|" << std::endl;
    std::cout << "| High accuracy needed        | GICP              | Best accuracy    |" << std::endl;
    std::cout << "| Sparse point clouds         | NDT               | Grid-based       |" << std::endl;
    std::cout << "| Dense point clouds          | ICP/GICP          | Point-based      |" << std::endl;
    std::cout << "| Structured environments     | NDT               | Planar surfaces  |" << std::endl;
    std::cout << "| Unknown initial pose        | Global + ICP/GICP | Need global first|" << std::endl;
    std::cout << "| Loop closure refinement     | GICP              | Most robust      |" << std::endl;

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. ICP: Fastest, but sensitive to noise and initial guess" << std::endl;
    std::cout << "2. GICP: Best accuracy, handles planar surfaces well, slower" << std::endl;
    std::cout << "3. NDT: Good balance of speed/accuracy, great for sparse data" << std::endl;
    std::cout << "4. All methods benefit from good initial guess" << std::endl;
    std::cout << "5. Preprocessing (filtering, downsampling) is crucial" << std::endl;
    std::cout << std::endl;

    return 0;
}
