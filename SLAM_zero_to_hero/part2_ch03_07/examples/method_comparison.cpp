/**
 * Point Cloud Registration Method Comparison
 *
 * This example provides a comprehensive comparison of different point cloud
 * registration methods: ICP, GICP, and NDT, on KITTI velodyne scans scored
 * against the KITTI ground-truth poses.
 *
 * Comparison criteria:
 * - Accuracy (translation and rotation error)
 * - Speed (processing time)
 * - Robustness to initial pose error
 * - Behavior with different point cloud densities
 * - How far apart two scans can be before local registration fails
 *
 * This helps you choose the right method for your application.
 *
 * Usage: ./method_comparison [source.bin target.bin]
 */

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

#include "demo_common.hpp"

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

/// Same voxel leaf as gicp_demo / ndt_demo, so numbers carry across the chapter
constexpr float kVoxelLeaf = 0.3f;
constexpr float kNdtResolution = 1.0f;
/// More-Thuente maximum step length. 0.1 m - the value indoor NDT examples use -
/// is shorter than TransformationEpsilon at KITTI's 1.3-1.5 m scan spacing, so NDT
/// would report convergence after a single iteration without having moved. See the
/// step size study in ndt_demo.
constexpr float kNdtStepSize = 0.5f;
constexpr double kMaxCorrespondenceDistance = 2.0;

/**
 * Result of running one method on one scan pair
 */
struct RegistrationResult {
    std::string method_name;
    bool converged;
    double fitness_score;
    demo::PoseError error;
    double time_ms;
    Eigen::Matrix4f transform;
};

/**
 * Run ICP registration
 */
RegistrationResult runICP(const CloudT::Ptr& source, const CloudT::Ptr& target,
                          const Eigen::Matrix4f& ground_truth,
                          const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                          demo::RegistrationViz* viz = nullptr) {
    RegistrationResult result;
    result.method_name = "ICP";

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    icp.setEuclideanFitnessEpsilon(1e-6);

    demo::attachIterationLogging(icp, viz, "ICP", 255, 170, 60);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = icp.hasConverged();
    result.fitness_score = icp.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = icp.getFinalTransformation();
    result.error = demo::poseError(result.transform, ground_truth);

    return result;
}

/**
 * Run GICP registration
 */
RegistrationResult runGICP(const CloudT::Ptr& source, const CloudT::Ptr& target,
                           const Eigen::Matrix4f& ground_truth,
                           const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                           demo::RegistrationViz* viz = nullptr) {
    RegistrationResult result;
    result.method_name = "GICP";

    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setInputSource(source);
    gicp.setInputTarget(target);
    gicp.setMaximumIterations(50);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.setCorrespondenceRandomness(20);
    gicp.setMaximumOptimizerIterations(20);

    demo::attachIterationLogging(gicp, viz, "GICP", 60, 220, 100);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    gicp.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = gicp.hasConverged();
    result.fitness_score = gicp.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = gicp.getFinalTransformation();
    result.error = demo::poseError(result.transform, ground_truth);

    return result;
}

/**
 * Run NDT registration
 */
RegistrationResult runNDT(const CloudT::Ptr& source, const CloudT::Ptr& target,
                          const Eigen::Matrix4f& ground_truth,
                          float resolution = kNdtResolution,
                          const Eigen::Matrix4f& initial_guess = Eigen::Matrix4f::Identity(),
                          demo::RegistrationViz* viz = nullptr) {
    RegistrationResult result;
    result.method_name = "NDT";

    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // Set the resolution BEFORE the target cloud. setInputTarget() builds the
    // voxel-covariance grid immediately, using whatever resolution is set at that
    // moment; a later setResolution() throws that grid away and builds another.
    // Besides the wasted work, the discarded grid is built at PCL's 1 m default,
    // which on a sparse cloud has too few points per cell and prints a scary
    // "Grid will not be searchable" warning that has nothing to do with the run.
    ndt.setResolution(resolution);
    ndt.setStepSize(kNdtStepSize);
    ndt.setTransformationEpsilon(0.01);
    ndt.setMaximumIterations(50);
    ndt.setInputSource(source);
    ndt.setInputTarget(target);

    demo::attachIterationLogging(ndt, viz, "NDT", 80, 140, 255);

    CloudT::Ptr aligned(new CloudT);

    auto start = std::chrono::high_resolution_clock::now();
    ndt.align(*aligned, initial_guess);
    auto end = std::chrono::high_resolution_clock::now();

    result.converged = ndt.hasConverged();
    result.fitness_score = ndt.getFitnessScore();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.transform = ndt.getFinalTransformation();
    result.error = demo::poseError(result.transform, ground_truth);

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
                  << std::setw(15) << std::setprecision(4) << r.error.translation_m
                  << std::setw(15) << std::setprecision(4) << r.error.rotation_deg
                  << std::setw(12) << std::setprecision(1) << r.time_ms << std::endl;
    }
    std::cout << std::string(90, '-') << std::endl;
}

/**
 * Test 1: accuracy on the scan pair, with no motion model
 */
void testBasicAccuracy(const demo::KittiPair& pair, demo::RegistrationViz* viz) {
    std::cout << "\n=== Test 1: Basic Accuracy Comparison ===" << std::endl;
    std::cout << "Identity initial guess - the full vehicle motion is left for"
              << std::endl;
    std::cout << "registration to recover\n" << std::endl;

    std::vector<RegistrationResult> results;
    results.push_back(runICP(pair.source, pair.target, pair.ground_truth,
                             Eigen::Matrix4f::Identity(), viz));
    results.push_back(runGICP(pair.source, pair.target, pair.ground_truth,
                              Eigen::Matrix4f::Identity(), viz));
    results.push_back(runNDT(pair.source, pair.target, pair.ground_truth, kNdtResolution,
                             Eigen::Matrix4f::Identity(), viz));

    printResults(results);

    const std::array<std::array<uint8_t, 3>, 3> method_colors{
        {{255, 170, 60}, {60, 220, 100}, {80, 140, 255}}};
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& c = method_colors[i % method_colors.size()];
        viz->logAligned("aligned_" + results[i].method_name, *pair.source,
                        results[i].transform, c[0], c[1], c[2]);
    }

    for (const auto& r : results) {
        std::cout << "\n" << r.method_name << " transformation:" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << r.transform << std::endl;
    }
    if (pair.has_ground_truth) {
        std::cout << "\nGround truth:" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << pair.ground_truth << std::endl;
    }
}

/**
 * Test 2: robustness to initial pose error
 *
 * The guesses are perturbations of the true transform, so they model a motion
 * model of decreasing quality. "Identity" is the no-model case.
 */
void testInitialPoseRobustness(const demo::KittiPair& pair) {
    std::cout << "\n=== Test 2: Robustness to Initial Pose Error ===" << std::endl;
    std::cout << "Testing with increasingly wrong initial guesses\n" << std::endl;

    const Eigen::Matrix4f& gt = pair.ground_truth;

    const std::vector<std::pair<std::string, Eigen::Matrix4f>> initial_guesses = {
        {"Exact", gt},
        {"0.2 m / 1 deg", gt * demo::makeTransform(0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.017f)},
        {"0.5 m / 3 deg", gt * demo::makeTransform(0.5f, 0.3f, 0.1f, 0.0f, 0.0f, 0.052f)},
        {"1.0 m / 6 deg", gt * demo::makeTransform(1.0f, 0.5f, 0.2f, 0.0f, 0.0f, 0.105f)},
        {"Identity", Eigen::Matrix4f::Identity()},
    };

    for (const auto& [name, guess] : initial_guesses) {
        std::cout << "\nInitial guess: " << name << std::endl;

        std::vector<RegistrationResult> results;
        results.push_back(runICP(pair.source, pair.target, gt, guess));
        results.push_back(runGICP(pair.source, pair.target, gt, guess));
        results.push_back(runNDT(pair.source, pair.target, gt, kNdtResolution, guess));

        printResults(results);
    }
}

/**
 * Test 3: point cloud density, and the speed that comes with it
 *
 * The two raw scans are voxelized at increasing leaf sizes, which is exactly the
 * knob a real pipeline turns to hit its time budget.
 */
void testDensityEffect(const demo::KittiPair& raw_pair) {
    std::cout << "\n=== Test 3: Effect of Point Cloud Density ===" << std::endl;
    std::cout << "Voxelizing the same two scans at increasing leaf sizes\n" << std::endl;

    const std::vector<float> voxel_sizes = {0.2f, 0.4f, 0.8f, 1.6f};

    for (float voxel_size : voxel_sizes) {
        demo::KittiPair pair = raw_pair;
        pair.source = demo::voxelDownsample(*raw_pair.source, voxel_size);
        pair.target = demo::voxelDownsample(*raw_pair.target, voxel_size);

        // NDT's cell size has to track the point spacing. PCL fits a Gaussian per
        // cell and needs at least 6 points to do it; on a surface sampled every
        // `leaf` meters a cell of size r holds about (r/leaf)^2 points, so r has
        // to stay around 3x the leaf or the grid comes out empty and NDT
        // silently returns its initial guess.
        const float ndt_resolution = std::max(kNdtResolution, voxel_size * 3.0f);

        std::cout << "\nVoxel leaf: " << voxel_size << " m, points: " << pair.source->size()
                  << " (source) / " << pair.target->size() << " (target)"
                  << ", NDT resolution: " << ndt_resolution << " m" << std::endl;

        std::vector<RegistrationResult> results;
        results.push_back(runICP(pair.source, pair.target, pair.ground_truth));
        results.push_back(runGICP(pair.source, pair.target, pair.ground_truth));
        results.push_back(runNDT(pair.source, pair.target, pair.ground_truth, ndt_resolution));

        printResults(results);
    }
}

/**
 * Test 4: how far apart two scans can be before local registration fails
 *
 * Registers the source frame against frames further and further ahead in the
 * sequence. On KITTI that is 1.3-1.5 m of vehicle motion per frame, so the gap
 * sweep walks the displacement from ~1.5 m to ~30 m while the overlap shrinks.
 * Needs the rest of the sequence next to the source scan, so it is skipped for
 * the two-scan sample bundled with this chapter.
 */
void testFrameGap(const demo::KittiPair& pair) {
    std::cout << "\n=== Test 4: Effect of Scan Separation ===" << std::endl;

    const std::vector<int> gaps = {1, 2, 5, 10, 20};

    if (pair.lidar_poses.empty() || pair.source_frame < 0) {
        std::cout << "Skipped: needs a KITTI sequence with ground-truth poses."
                  << std::endl;
        return;
    }

    std::cout << "Registering frame " << pair.source_frame
              << " against frames further ahead in the sequence\n" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::setw(8) << "Gap"
              << std::setw(12) << "GT |t| (m)"
              << std::setw(14) << "ICP err (m)"
              << std::setw(14) << "GICP err (m)"
              << std::setw(14) << "NDT err (m)"
              << std::setw(28) << "time ICP/GICP/NDT (ms)" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    int tested = 0;
    for (int gap : gaps) {
        const int target_frame = pair.source_frame + gap;
        if (target_frame >= static_cast<int>(pair.lidar_poses.size())) continue;

        const std::string target_path =
            demo::kittiScanPath(pair.sequence_files.velodyne_dir, target_frame);
        CloudT::Ptr target_raw = demo::loadCloud(target_path);
        if (!target_raw) continue;

        const Eigen::Matrix4f gt = pair.lidar_poses[target_frame].inverse() *
                                   pair.lidar_poses[pair.source_frame];
        CloudT::Ptr target = demo::voxelDownsample(*target_raw, kVoxelLeaf);

        const auto icp = runICP(pair.source, target, gt);
        const auto gicp = runGICP(pair.source, target, gt);
        const auto ndt = runNDT(pair.source, target, gt);

        std::cout << std::setw(8) << gap
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << gt.block<3, 1>(0, 3).norm()
                  << std::setw(14) << std::setprecision(3) << icp.error.translation_m
                  << std::setw(14) << std::setprecision(3) << gicp.error.translation_m
                  << std::setw(14) << std::setprecision(3) << ndt.error.translation_m
                  << std::setw(10) << std::setprecision(0) << icp.time_ms
                  << std::setw(9) << gicp.time_ms
                  << std::setw(9) << ndt.time_ms << std::endl;
        ++tested;
    }
    std::cout << std::string(90, '-') << std::endl;

    if (tested == 0) {
        std::cout << "No further frames found next to " << pair.source_file << "."
                  << std::endl;
        std::cout << "Point the demo at a full KITTI sequence to run this test."
                  << std::endl;
    } else {
        std::cout << "\nAll three methods are local: they need the initial overlap to be"
                  << std::endl;
        std::cout << "good enough that nearest-neighbour correspondences are mostly right."
                  << std::endl;
        std::cout << "Once the gap grows past a few meters the identity initial guess is"
                  << std::endl;
        std::cout << "no longer inside the basin of convergence and the error jumps to the"
                  << std::endl;
        std::cout << "order of the displacement itself. That is the regime global"
                  << std::endl;
        std::cout << "registration is for - see teaser_demo." << std::endl;
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
}

int main(int argc, char** argv) {
    std::cout << "=== Point Cloud Registration Method Comparison ===" << std::endl;
    std::cout << "ICP vs GICP vs NDT, on KITTI\n" << std::endl;

    if (argc != 1 && argc != 3) {
        printUsage(argv[0]);
        return -1;
    }

    demo::KittiPair raw_pair = (argc == 3) ? demo::loadKittiPair(argv[1], argv[2])
                                           : demo::loadKittiPair();
    if (!raw_pair.source || !raw_pair.target) {
        return -1;
    }
    demo::printKittiPair(raw_pair);

    // Test 3 voxelizes the raw scans itself, so keep them and hand the other
    // tests a downsampled copy
    demo::KittiPair pair = raw_pair;
    pair.source = demo::voxelDownsample(*raw_pair.source, kVoxelLeaf);
    pair.target = demo::voxelDownsample(*raw_pair.target, kVoxelLeaf);
    std::cout << "\nVoxel-downsampling to " << kVoxelLeaf << " m: source "
              << raw_pair.source->size() << " -> " << pair.source->size() << ", target "
              << raw_pair.target->size() << " -> " << pair.target->size() << std::endl;

    // Stream to a rerun viewer (no-op without SDK/viewer): target and initial
    // source, every method's optimization steps on the "iteration" timeline,
    // and each aligned result
    demo::RegistrationViz viz("method_comparison");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    testBasicAccuracy(pair, &viz);

    if (pair.has_ground_truth) {
        testInitialPoseRobustness(pair);
    } else {
        std::cout << "\n=== Test 2: Robustness to Initial Pose Error ===" << std::endl;
        std::cout << "Skipped: the guesses are built from the ground-truth transform,"
                  << std::endl;
        std::cout << "which is not available for this scan pair." << std::endl;
    }

    testDensityEffect(raw_pair);
    testFrameGap(pair);

    // ====================================
    // Summary and Recommendations
    // ====================================

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "=== Method Selection Guide ===" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << "\n| Scenario                    | Recommended Method | Notes            |"
              << std::endl;
    std::cout << "|-----------------------------|-------------------|------------------|"
              << std::endl;
    std::cout << "| Real-time odometry          | NDT               | Fast, good enough|"
              << std::endl;
    std::cout << "| High accuracy needed        | GICP              | Best accuracy    |"
              << std::endl;
    std::cout << "| Sparse point clouds         | NDT               | Grid-based       |"
              << std::endl;
    std::cout << "| Dense point clouds          | ICP/GICP          | Point-based      |"
              << std::endl;
    std::cout << "| Structured environments     | NDT               | Planar surfaces  |"
              << std::endl;
    std::cout << "| Unknown initial pose        | Global + ICP/GICP | Need global first|"
              << std::endl;
    std::cout << "| Loop closure refinement     | GICP              | Most robust      |"
              << std::endl;

    std::cout << "\n=== Key Takeaways ===" << std::endl;
    std::cout << "1. ICP: Fastest, but sensitive to noise and initial guess" << std::endl;
    std::cout << "2. GICP: Best accuracy, handles planar surfaces well, slower" << std::endl;
    std::cout << "3. NDT: Good balance of speed/accuracy, great for sparse data" << std::endl;
    std::cout << "4. All three are local methods - they all need a good initial guess"
              << std::endl;
    std::cout << "5. Preprocessing (filtering, downsampling) is crucial" << std::endl;
    std::cout << std::endl;

    return 0;
}
