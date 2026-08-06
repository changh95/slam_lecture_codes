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
 * Runs on two KITTI velodyne scans and scores against the KITTI ground-truth
 * poses. With no arguments it uses the pair bundled with this chapter
 * (sequence 04, frames 0 and 1, 1.31 m apart).
 *
 * Usage: ./ndt_demo [source.bin target.bin [resolution]]
 *
 * Reference: Biber & Strasser, "The Normal Distributions Transform", IROS 2003
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>

#include "demo_common.hpp"

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/// Same voxel leaf as the other demos in this chapter, so the numbers are
/// directly comparable across gicp_demo / ndt_demo / method_comparison
constexpr float kVoxelLeaf = 0.3f;

/// Default NDT cell size for outdoor LiDAR
constexpr float kDefaultResolution = 1.0f;

/// Maximum step length of the More-Thuente line search, in meters.
///
/// This has to be set against the displacement, not copied from an indoor
/// tutorial. Consecutive KITTI scans are 1.3-1.5 m apart, and with the 0.1 m
/// step that indoor examples use, NDT's first step is shorter than
/// TransformationEpsilon - so it reports "converged" after one iteration having
/// barely moved, leaving essentially the whole vehicle motion as error. The step
/// size study below shows the cliff between 0.1 and 0.5.
constexpr float kDefaultStepSize = 0.5f;

struct NDTResult {
    bool converged;
    int iterations;
    double fitness_score;
    demo::PoseError error;
    double time_ms;
    Eigen::Matrix4f transform;
};

/**
 * Run NDT registration
 */
NDTResult runNDT(const demo::KittiPair& pair,
                 float resolution,
                 float step_size,
                 const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                 demo::RegistrationViz* viz = nullptr,
                 demo::ErrorTrace* trace = nullptr) {

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // NDT parameters. The resolution has to be set BEFORE the target cloud:
    // setInputTarget() builds the voxel-covariance grid there and then, at
    // whatever resolution is currently set, and a later setResolution() discards
    // that grid and builds a second one. Setting it first also keeps the reported
    // time to a single grid build.
    ndt.setResolution(resolution);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(0.01);
    ndt.setMaximumIterations(50);

    // Set input clouds
    ndt.setInputSource(pair.source);
    ndt.setInputTarget(pair.target);

    demo::attachIterationLogging(ndt, viz, "NDT", 60, 220, 100, pair.source,
                                 pair.has_ground_truth ? &pair.ground_truth : nullptr,
                                 trace, &initial_guess);

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
    result.error = demo::poseError(result.transform, pair.ground_truth);

    return result;
}

/**
 * Study the effect of resolution parameter
 */
void studyResolution(const demo::KittiPair& pair) {
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

    const std::vector<float> resolutions = {0.5f, 1.0f, 2.0f, 3.0f, 5.0f};

    for (float res : resolutions) {
        const NDTResult result = runNDT(pair, res, kDefaultStepSize);

        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << res
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(12) << result.iterations
                  << std::setw(15) << std::setprecision(4) << result.error.translation_m
                  << std::setw(15) << std::setprecision(4) << result.error.rotation_deg
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - Smaller resolution = more cells = finer detail but a narrower basin"
              << std::endl;
    std::cout << "    of convergence, so a coarse initial guess is more likely to fail"
              << std::endl;
    std::cout << "  - Larger resolution = fewer cells = more forgiving but less precise"
              << std::endl;
    std::cout << "  - Typical values: 0.5-2.0m for outdoor, 0.1-0.5m for indoor" << std::endl;
}

/**
 * Study the effect of step size parameter
 */
void studyStepSize(const demo::KittiPair& pair) {
    std::cout << "\n=== NDT Step Size Parameter Study ===" << std::endl;
    std::cout << "(More-Thuente line search step size affects convergence)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(12) << "Step Size"
              << std::setw(12) << "Converged"
              << std::setw(12) << "Iters"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    const std::vector<float> step_sizes = {0.01f, 0.05f, 0.1f, 0.5f, 1.0f};

    for (float step : step_sizes) {
        const NDTResult result = runNDT(pair, kDefaultResolution, step);

        std::cout << std::setw(12) << std::fixed << std::setprecision(2) << step
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(12) << result.iterations
                  << std::setw(15) << std::setprecision(4) << result.error.translation_m
                  << std::setw(15) << std::setprecision(4) << result.error.rotation_deg
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - The step size is the MAXIMUM length of the line search step, so it"
              << std::endl;
    std::cout << "    has to be set against the displacement being recovered. At KITTI's"
              << std::endl;
    std::cout << "    1.3-1.5 m scan spacing the small steps stall: the first step comes"
              << std::endl;
    std::cout << "    out shorter than TransformationEpsilon, NDT stops after one"
              << std::endl;
    std::cout << "    iteration, and nearly the whole vehicle motion is left as error."
              << std::endl;
    std::cout << "  - Note that those rows still report Converged = YES. PCL's flag only"
              << std::endl;
    std::cout << "    means the update fell below the epsilon - it is not a statement"
              << std::endl;
    std::cout << "    about correctness. Always read it next to the iteration count."
              << std::endl;
    std::cout << "  - Too large a step = may overshoot" << std::endl;
    std::cout << "  - Typical values: 0.05-0.5 indoor, 0.5-1.0 for KITTI-scale motion"
              << std::endl;
}

/**
 * Demonstrate the importance of the initial guess
 *
 * The guesses are built from the true transform, so "Exact" is what a perfect
 * motion model would hand NDT and "Identity" is what it gets with no motion
 * model at all - on KITTI that is already the full 1.3-1.5 m of vehicle motion.
 */
void studyInitialGuess(const demo::KittiPair& pair) {
    std::cout << "\n=== Initial Guess Study ===" << std::endl;
    std::cout << "(How the initial guess affects NDT convergence)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(25) << "Initial Guess"
              << std::setw(12) << "Converged"
              << std::setw(15) << "Trans Err (m)"
              << std::setw(15) << "Rot Err (deg)"
              << std::setw(12) << "Time (ms)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    const Eigen::Matrix4f& gt = pair.ground_truth;

    const std::vector<std::pair<std::string, Eigen::Matrix4f>> guesses = {
        {"Exact", gt},
        {"0.2 m / 1 deg off", gt * demo::makeTransform(0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.017f)},
        {"0.5 m / 3 deg off", gt * demo::makeTransform(0.5f, 0.3f, 0.1f, 0.0f, 0.0f, 0.052f)},
        {"2.0 m / 10 deg off", gt * demo::makeTransform(2.0f, 1.0f, 0.5f, 0.0f, 0.0f, 0.175f)},
        {"Identity (no model)", Eigen::Matrix4f::Identity()},
    };

    for (const auto& [name, guess] : guesses) {
        const NDTResult result = runNDT(pair, kDefaultResolution, kDefaultStepSize, guess);

        std::cout << std::setw(25) << name
                  << std::setw(12) << (result.converged ? "YES" : "NO")
                  << std::setw(15) << std::fixed << std::setprecision(4)
                  << result.error.translation_m
                  << std::setw(15) << std::setprecision(4) << result.error.rotation_deg
                  << std::setw(12) << std::setprecision(1) << result.time_ms << std::endl;
    }

    std::cout << "\nObservations:" << std::endl;
    std::cout << "  - A good initial guess significantly improves the result" << std::endl;
    std::cout << "  - In odometry, use the previous pose estimate (constant velocity)"
              << std::endl;
    std::cout << "  - For loop closure, run global registration first (see teaser_demo)"
              << std::endl;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.bin target.bin [resolution]]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments        - Use the bundled KITTI pair (seq 04, frames 0-1)"
              << std::endl;
    std::cout << "  source target       - Use the given KITTI velodyne .bin (or .pcd) scans"
              << std::endl;
    std::cout << "  source target res   - ... with a custom NDT cell size in meters"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << std::endl;
    std::cout << "  " << prog_name << " <kitti>/sequences/04/velodyne/000000.bin"
              << " <kitti>/sequences/04/velodyne/000001.bin 1.0" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Normal Distributions Transform (NDT) Demo ===" << std::endl;
    std::cout << "Grid-based registration with Gaussian distributions, on KITTI\n"
              << std::endl;

    if (argc != 1 && argc != 3 && argc != 4) {
        printUsage(argv[0]);
        return -1;
    }

    demo::KittiPair pair = (argc >= 3) ? demo::loadKittiPair(argv[1], argv[2])
                                       : demo::loadKittiPair();
    if (!pair.source || !pair.target) {
        return -1;
    }
    demo::printKittiPair(pair);

    float resolution = kDefaultResolution;
    if (argc == 4) {
        resolution = std::stof(argv[3]);
    }
    std::cout << "  NDT resolution: " << resolution << " m" << std::endl;

    // Downsample
    std::cout << "\nVoxel-downsampling to " << kVoxelLeaf << " m..." << std::endl;
    const std::size_t source_raw = pair.source->size();
    const std::size_t target_raw = pair.target->size();
    pair.source = demo::voxelDownsample(*pair.source, kVoxelLeaf);
    pair.target = demo::voxelDownsample(*pair.target, kVoxelLeaf);
    std::cout << "  Source: " << source_raw << " -> " << pair.source->size() << std::endl;
    std::cout << "  Target: " << target_raw << " -> " << pair.target->size() << std::endl;

    // ====================================
    // Basic NDT registration
    // ====================================

    std::cout << "\n=== Basic NDT Registration ===" << std::endl;
    std::cout << "(identity initial guess - no motion model)" << std::endl;

    // Stream inputs, per-iteration steps, and the result to a rerun viewer
    // (no-op without SDK/viewer)
    demo::RegistrationViz viz("ndt_demo");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    demo::ErrorTrace ndt_trace;
    const NDTResult basic_result = runNDT(pair, resolution, kDefaultStepSize,
                                          Eigen::Matrix4f::Identity(), &viz, &ndt_trace);
    viz.logErrorCurves({ndt_trace});

    std::cout << "  Converged: " << (basic_result.converged ? "YES" : "NO") << std::endl;
    std::cout << "  Iterations: " << basic_result.iterations << std::endl;
    std::cout << "  Fitness score: " << std::fixed << std::setprecision(6)
              << basic_result.fitness_score << std::endl;
    std::cout << "  Translation error: " << std::setprecision(4)
              << basic_result.error.translation_m << " m" << std::endl;
    std::cout << "  Rotation error: " << std::setprecision(4)
              << basic_result.error.rotation_deg << " deg" << std::endl;
    std::cout << "  Time: " << std::setprecision(2) << basic_result.time_ms << " ms"
              << std::endl;

    viz.logAligned("aligned_ndt", *pair.source, basic_result.transform, 60, 220, 100);

    // ====================================
    // Parameter studies
    // ====================================

    studyResolution(pair);
    studyStepSize(pair);
    if (pair.has_ground_truth) {
        studyInitialGuess(pair);
    } else {
        std::cout << "\n=== Initial Guess Study ===" << std::endl;
        std::cout << "Skipped: the guesses are built from the ground-truth transform,"
                  << std::endl;
        std::cout << "which is not available for this scan pair." << std::endl;
    }

    // ====================================
    // Final transformation
    // ====================================

    std::cout << "\n=== Final NDT Transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << basic_result.transform << std::endl;

    if (pair.has_ground_truth) {
        std::cout << "\n=== Ground Truth ===" << std::endl;
        std::cout << pair.ground_truth << std::endl;
    }

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "NDT key characteristics:" << std::endl;
    std::cout << "  1. Represents the target as a grid of Gaussian distributions"
              << std::endl;
    std::cout << "  2. No explicit correspondences needed (unlike ICP)" << std::endl;
    std::cout << "  3. Smooth cost function - good for Newton optimization" << std::endl;
    std::cout << "  4. Cost is linear in the number of source points" << std::endl;
    std::cout << std::endl;
    std::cout << "Recommended parameter settings:" << std::endl;
    std::cout << "  - Outdoor LiDAR (KITTI): resolution=1.0-2.0m, step_size=0.1" << std::endl;
    std::cout << "  - Indoor/dense: resolution=0.2-0.5m, step_size=0.05" << std::endl;
    std::cout << "  - Always provide a good initial guess when possible" << std::endl;
    std::cout << std::endl;

    return 0;
}
