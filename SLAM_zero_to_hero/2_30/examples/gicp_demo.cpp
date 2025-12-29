/**
 * Generalized ICP (GICP) Demo
 *
 * This example demonstrates the Generalized Iterative Closest Point algorithm,
 * which extends standard ICP by modeling point clouds as Gaussian distributions.
 * GICP combines point-to-point and point-to-plane ICP in a probabilistic framework.
 *
 * Key concepts covered:
 * - GICP algorithm overview
 * - Setting up GICP parameters
 * - Understanding covariance models
 * - Comparing GICP with standard ICP
 *
 * Reference: Segal et al., "Generalized-ICP", RSS 2009
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
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/**
 * Generate a synthetic point cloud (ground plane with some structures)
 */
CloudT::Ptr generateSyntheticCloud(int num_points = 10000) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_xy(-5.0f, 5.0f);
    std::uniform_real_distribution<float> dist_z(-0.1f, 0.1f);
    std::uniform_real_distribution<float> noise(0.0f, 1.0f);

    // Ground plane
    for (int i = 0; i < num_points / 2; ++i) {
        PointT pt;
        pt.x = dist_xy(gen);
        pt.y = dist_xy(gen);
        pt.z = dist_z(gen);  // Near-planar
        cloud->push_back(pt);
    }

    // Vertical wall
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        pt.x = 3.0f + dist_z(gen);
        pt.y = dist_xy(gen);
        pt.z = noise(gen) * 2.0f;
        cloud->push_back(pt);
    }

    // Box structure
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        float side = noise(gen);
        if (side < 0.33f) {
            pt.x = -2.0f + noise(gen) * 0.5f;
            pt.y = noise(gen) * 0.5f - 0.25f;
            pt.z = noise(gen) * 0.5f;
        } else if (side < 0.66f) {
            pt.x = -2.0f + noise(gen) * 0.5f;
            pt.y = noise(gen) * 0.5f - 0.25f + 0.5f;
            pt.z = noise(gen) * 0.5f;
        } else {
            pt.x = -2.0f + noise(gen) * 0.5f;
            pt.y = noise(gen) * 0.5f - 0.25f;
            pt.z = 0.5f + dist_z(gen);
        }
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Apply a transformation to create a transformed version of the source cloud
 */
CloudT::Ptr transformCloud(const CloudT::Ptr& input,
                            float tx, float ty, float tz,
                            float roll, float pitch, float yaw) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << tx, ty, tz;
    transform.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));

    CloudT::Ptr output(new CloudT);
    pcl::transformPointCloud(*input, *output, transform);

    return output;
}

/**
 * Compute transformation error
 */
void computeTransformError(const Eigen::Matrix4f& estimated,
                            const Eigen::Matrix4f& ground_truth,
                            float& translation_error,
                            float& rotation_error_deg) {
    // Translation error
    Eigen::Vector3f t_est = estimated.block<3, 1>(0, 3);
    Eigen::Vector3f t_gt = ground_truth.block<3, 1>(0, 3);
    translation_error = (t_est - t_gt).norm();

    // Rotation error
    Eigen::Matrix3f R_est = estimated.block<3, 3>(0, 0);
    Eigen::Matrix3f R_gt = ground_truth.block<3, 3>(0, 0);
    Eigen::Matrix3f R_error = R_est.transpose() * R_gt;
    float trace = R_error.trace();
    float cos_angle = std::max(-1.0f, std::min(1.0f, (trace - 1.0f) / 2.0f));
    rotation_error_deg = std::acos(cos_angle) * 180.0f / M_PI;
}

/**
 * Run standard ICP for comparison
 */
void runStandardICP(const CloudT::Ptr& source,
                    const CloudT::Ptr& target,
                    const Eigen::Matrix4f& ground_truth,
                    Eigen::Matrix4f& result,
                    double& time_ms) {
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);

    // ICP parameters
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(2.0);
    icp.setEuclideanFitnessEpsilon(1e-6);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result = icp.getFinalTransformation();

    float trans_err, rot_err;
    computeTransformError(result, ground_truth, trans_err, rot_err);

    std::cout << "\n--- Standard ICP Results ---" << std::endl;
    std::cout << "  Converged: " << (icp.hasConverged() ? "YES" : "NO") << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << icp.getFitnessScore() << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4) << trans_err << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4) << rot_err << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << time_ms << " ms" << std::endl;
}

/**
 * Run GICP
 */
void runGICP(const CloudT::Ptr& source,
             const CloudT::Ptr& target,
             const Eigen::Matrix4f& ground_truth,
             Eigen::Matrix4f& result,
             double& time_ms) {
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(target);

    // GICP parameters
    gicp.setMaximumIterations(50);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(2.0);
    gicp.setEuclideanFitnessEpsilon(1e-6);

    // GICP-specific parameters
    gicp.setCorrespondenceRandomness(20);      // Number of neighbors for covariance
    gicp.setMaximumOptimizerIterations(20);    // Inner loop iterations

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    gicp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result = gicp.getFinalTransformation();

    float trans_err, rot_err;
    computeTransformError(result, ground_truth, trans_err, rot_err);

    std::cout << "\n--- GICP Results ---" << std::endl;
    std::cout << "  Converged: " << (gicp.hasConverged() ? "YES" : "NO") << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << gicp.getFitnessScore() << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4) << trans_err << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4) << rot_err << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << time_ms << " ms" << std::endl;
}

/**
 * Test GICP with different correspondence randomness settings
 */
void testCorrespondenceRandomness(const CloudT::Ptr& source,
                                   const CloudT::Ptr& target,
                                   const Eigen::Matrix4f& ground_truth) {
    std::cout << "\n=== Correspondence Randomness Parameter Study ===" << std::endl;
    std::cout << "(Number of neighbors used to compute covariances)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    std::cout << std::setw(15) << "Randomness"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    std::vector<int> randomness_values = {5, 10, 20, 30, 50};

    for (int k : randomness_values) {
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
        gicp.setInputSource(source);
        gicp.setInputTarget(target);
        gicp.setMaximumIterations(50);
        gicp.setTransformationEpsilon(1e-8);
        gicp.setMaxCorrespondenceDistance(2.0);
        gicp.setCorrespondenceRandomness(k);

        CloudT::Ptr aligned(new CloudT);

        auto start = std::chrono::high_resolution_clock::now();
        gicp.align(*aligned);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        Eigen::Matrix4f result = gicp.getFinalTransformation();

        float trans_err, rot_err;
        computeTransformError(result, ground_truth, trans_err, rot_err);

        std::cout << std::setw(15) << k
                  << std::setw(15) << std::fixed << std::setprecision(4) << trans_err
                  << std::setw(15) << std::setprecision(4) << rot_err
                  << std::setw(12) << std::setprecision(1) << time_ms << std::endl;
    }
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.pcd target.pcd]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments    - Use synthetic point clouds for demo" << std::endl;
    std::cout << "  source target   - Use provided PCD files" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << std::endl;
    std::cout << "  " << prog_name << " scan1.pcd scan2.pcd" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Generalized ICP (GICP) Demo ===" << std::endl;
    std::cout << "Probabilistic ICP with plane-to-plane matching\n" << std::endl;

    CloudT::Ptr source(new CloudT);
    CloudT::Ptr target(new CloudT);
    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();
    bool use_synthetic = true;

    if (argc == 1) {
        // Generate synthetic data
        std::cout << "Generating synthetic point cloud..." << std::endl;

        target = generateSyntheticCloud(10000);

        // Define ground truth transformation
        float tx = 0.3f, ty = 0.2f, tz = 0.1f;
        float roll = 0.05f, pitch = 0.03f, yaw = 0.08f;

        // Create ground truth matrix
        Eigen::Affine3f gt_transform = Eigen::Affine3f::Identity();
        gt_transform.translation() << tx, ty, tz;
        gt_transform.rotate(Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));
        gt_transform.rotate(Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()));
        gt_transform.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
        ground_truth = gt_transform.matrix();

        // Source is transformed version of target
        source = transformCloud(target, tx, ty, tz, roll, pitch, yaw);

        std::cout << "  Source points: " << source->size() << std::endl;
        std::cout << "  Target points: " << target->size() << std::endl;
        std::cout << "  Ground truth translation: [" << tx << ", " << ty << ", " << tz << "]" << std::endl;
        std::cout << "  Ground truth rotation (rad): [" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;

    } else if (argc == 3) {
        // Load from files
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

        std::cout << "  Source: " << argv[1] << " (" << source->size() << " points)" << std::endl;
        std::cout << "  Target: " << argv[2] << " (" << target->size() << " points)" << std::endl;

        // For real data, we assume identity as "ground truth" for error reference
        std::cout << "  Note: Using identity as reference (no ground truth available)" << std::endl;

    } else {
        printUsage(argv[0]);
        return -1;
    }

    // Downsample if needed
    if (source->size() > 50000 || target->size() > 50000) {
        std::cout << "\nDownsampling large clouds for faster processing..." << std::endl;

        pcl::VoxelGrid<PointT> voxel;
        voxel.setLeafSize(0.1f, 0.1f, 0.1f);

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
    // Compare Standard ICP vs GICP
    // ====================================

    std::cout << "\n=== Comparing ICP and GICP ===" << std::endl;

    Eigen::Matrix4f icp_result, gicp_result;
    double icp_time, gicp_time;

    runStandardICP(source, target, ground_truth, icp_result, icp_time);
    runGICP(source, target, ground_truth, gicp_result, gicp_time);

    // ====================================
    // Parameter Study (only for synthetic data)
    // ====================================

    if (use_synthetic) {
        testCorrespondenceRandomness(source, target, ground_truth);
    }

    // ====================================
    // Print final transformation
    // ====================================

    std::cout << "\n=== Final GICP Transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << gicp_result << std::endl;

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "GICP advantages over standard ICP:" << std::endl;
    std::cout << "  1. Models local surface structure as Gaussian distributions" << std::endl;
    std::cout << "  2. Handles uncertainty in both source and target clouds" << std::endl;
    std::cout << "  3. Better convergence on planar surfaces" << std::endl;
    std::cout << "  4. More robust to noise and varying point density" << std::endl;
    std::cout << std::endl;
    std::cout << "Key parameters:" << std::endl;
    std::cout << "  - CorrespondenceRandomness: Number of neighbors for covariance" << std::endl;
    std::cout << "  - MaximumOptimizerIterations: Inner optimization loop iterations" << std::endl;
    std::cout << "  - MaxCorrespondenceDistance: Maximum point-to-point distance" << std::endl;
    std::cout << std::endl;

    return 0;
}
