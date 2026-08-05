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
 * Runs on two KITTI velodyne scans and scores both methods against the KITTI
 * ground-truth poses. With no arguments it uses the pair bundled with this
 * chapter (sequence 04, frames 0 and 1, 1.31 m apart).
 *
 * Usage: ./gicp_demo [source.bin target.bin]
 *
 * Reference: Segal et al., "Generalized-ICP", RSS 2009
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include "demo_common.hpp"

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/// Voxel leaf for the registration clouds. A 64-beam KITTI scan is ~124k points
/// over an 80 m radius; 0.3 m brings it to ~29k, which keeps GICP's per-point
/// covariance estimation tractable without throwing away street structure.
constexpr float kVoxelLeaf = 0.3f;

/// Correspondence distance. Consecutive KITTI scans are 1-2 m apart, so the
/// search radius has to cover that motion before ICP has corrected for it.
constexpr double kMaxCorrespondenceDistance = 2.0;

/**
 * Run standard ICP for comparison
 */
void runStandardICP(const demo::KittiPair& pair,
                    Eigen::Matrix4f& result,
                    double& time_ms,
                    demo::RegistrationViz* viz = nullptr) {
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(pair.source);
    icp.setInputTarget(pair.target);

    // ICP parameters
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    icp.setEuclideanFitnessEpsilon(1e-6);

    demo::attachIterationLogging(icp, viz, "icp", 255, 170, 60);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result = icp.getFinalTransformation();

    const demo::PoseError err = demo::poseError(result, pair.ground_truth);

    std::cout << "\n--- Standard ICP Results ---" << std::endl;
    std::cout << "  Converged: " << (icp.hasConverged() ? "YES" : "NO") << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << icp.getFitnessScore() << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4) << err.translation_m
              << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4) << err.rotation_deg
              << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << time_ms << " ms" << std::endl;
}

/**
 * Run GICP
 */
void runGICP(const demo::KittiPair& pair,
             Eigen::Matrix4f& result,
             double& time_ms,
             demo::RegistrationViz* viz = nullptr) {
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(pair.source);
    gicp.setInputTarget(pair.target);

    // GICP parameters
    gicp.setMaximumIterations(50);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    gicp.setEuclideanFitnessEpsilon(1e-6);

    // GICP-specific parameters
    gicp.setCorrespondenceRandomness(20);      // Number of neighbors for covariance
    gicp.setMaximumOptimizerIterations(20);    // Inner loop iterations

    demo::attachIterationLogging(gicp, viz, "gicp", 60, 220, 100);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    gicp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result = gicp.getFinalTransformation();

    const demo::PoseError err = demo::poseError(result, pair.ground_truth);

    std::cout << "\n--- GICP Results ---" << std::endl;
    std::cout << "  Converged: " << (gicp.hasConverged() ? "YES" : "NO") << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << gicp.getFitnessScore() << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4) << err.translation_m
              << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4) << err.rotation_deg
              << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << time_ms << " ms" << std::endl;
}

/**
 * Test GICP with different correspondence randomness settings
 */
void testCorrespondenceRandomness(const demo::KittiPair& pair) {
    std::cout << "\n=== Correspondence Randomness Parameter Study ===" << std::endl;
    std::cout << "(Number of neighbors used to compute covariances)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    std::cout << std::setw(15) << "Randomness"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    const std::vector<int> randomness_values = {5, 10, 20, 30, 50};

    for (int k : randomness_values) {
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
        gicp.setInputSource(pair.source);
        gicp.setInputTarget(pair.target);
        gicp.setMaximumIterations(50);
        gicp.setTransformationEpsilon(1e-8);
        gicp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
        gicp.setCorrespondenceRandomness(k);

        CloudT::Ptr aligned(new CloudT);

        auto start = std::chrono::high_resolution_clock::now();
        gicp.align(*aligned);
        auto end = std::chrono::high_resolution_clock::now();

        const double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        const demo::PoseError err =
            demo::poseError(gicp.getFinalTransformation(), pair.ground_truth);

        std::cout << std::setw(15) << k
                  << std::setw(15) << std::fixed << std::setprecision(4) << err.translation_m
                  << std::setw(15) << std::setprecision(4) << err.rotation_deg
                  << std::setw(12) << std::setprecision(1) << time_ms << std::endl;
    }
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.bin target.bin]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments    - Use the bundled KITTI pair (seq 04, frames 0-1)"
              << std::endl;
    std::cout << "  source target   - Use the given KITTI velodyne .bin (or .pcd) scans"
              << std::endl;
    std::cout << std::endl;
    std::cout << "The KITTI calib.txt and ground-truth poses are located relative to"
              << std::endl;
    std::cout << "the scans, so only the scan paths are needed:" << std::endl;
    std::cout << "  " << prog_name << std::endl;
    std::cout << "  " << prog_name << " <kitti>/sequences/04/velodyne/000000.bin"
              << " <kitti>/sequences/04/velodyne/000001.bin" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Generalized ICP (GICP) Demo ===" << std::endl;
    std::cout << "Probabilistic ICP with plane-to-plane matching, on KITTI\n" << std::endl;

    if (argc != 1 && argc != 3) {
        printUsage(argv[0]);
        return -1;
    }

    demo::KittiPair pair = (argc == 3) ? demo::loadKittiPair(argv[1], argv[2])
                                       : demo::loadKittiPair();
    if (!pair.source || !pair.target) {
        return -1;
    }
    demo::printKittiPair(pair);

    // Downsample. Voxelizing both scans identically keeps the comparison fair -
    // every method below sees exactly the same points.
    std::cout << "\nVoxel-downsampling to " << kVoxelLeaf << " m..." << std::endl;
    const std::size_t source_raw = pair.source->size();
    const std::size_t target_raw = pair.target->size();
    pair.source = demo::voxelDownsample(*pair.source, kVoxelLeaf);
    pair.target = demo::voxelDownsample(*pair.target, kVoxelLeaf);
    std::cout << "  Source: " << source_raw << " -> " << pair.source->size() << std::endl;
    std::cout << "  Target: " << target_raw << " -> " << pair.target->size() << std::endl;

    // ====================================
    // Compare Standard ICP vs GICP
    // ====================================

    std::cout << "\n=== Comparing ICP and GICP ===" << std::endl;

    // Stream inputs, per-iteration steps, and results to a rerun viewer
    // (no-op without SDK/viewer)
    demo::RegistrationViz viz("gicp_demo");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    Eigen::Matrix4f icp_result, gicp_result;
    double icp_time, gicp_time;

    runStandardICP(pair, icp_result, icp_time, &viz);
    runGICP(pair, gicp_result, gicp_time, &viz);

    viz.logAligned("aligned_icp", *pair.source, icp_result, 255, 170, 60);
    viz.logAligned("aligned_gicp", *pair.source, gicp_result, 60, 220, 100);

    // ====================================
    // Parameter Study
    // ====================================

    testCorrespondenceRandomness(pair);

    // ====================================
    // Print final transformation
    // ====================================

    std::cout << "\n=== Final GICP Transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << gicp_result << std::endl;

    if (pair.has_ground_truth) {
        std::cout << "\n=== Ground Truth ===" << std::endl;
        std::cout << pair.ground_truth << std::endl;
    }

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "GICP advantages over standard ICP:" << std::endl;
    std::cout << "  1. Models local surface structure as Gaussian distributions" << std::endl;
    std::cout << "  2. Handles uncertainty in both source and target clouds" << std::endl;
    std::cout << "  3. Better convergence on planar surfaces (road, walls)" << std::endl;
    std::cout << "  4. More robust to noise and varying point density" << std::endl;
    std::cout << std::endl;
    std::cout << "Key parameters:" << std::endl;
    std::cout << "  - CorrespondenceRandomness: Number of neighbors for covariance" << std::endl;
    std::cout << "  - MaximumOptimizerIterations: Inner optimization loop iterations" << std::endl;
    std::cout << "  - MaxCorrespondenceDistance: Maximum point-to-point distance" << std::endl;
    std::cout << std::endl;

    return 0;
}
