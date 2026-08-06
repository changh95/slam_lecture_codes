/**
 * GICP experiment: the algorithm, and then the implementation
 *
 * Two questions get asked here, and they are deliberately separated because
 * conflating them is the usual way this comparison goes wrong:
 *
 *   1. Does GICP beat plain ICP?  PCL ICP vs PCL GICP - same library, same
 *      correspondence machinery, only the cost function differs. Any difference
 *      is the algorithm.
 *
 *   2. Does the implementation matter?  PCL GICP vs small_gicp GICP vs
 *      fast_gicp's CUDA VGICP - broadly the same algorithm, three very different
 *      implementations. Any difference is engineering, not mathematics.
 *
 * Read the table down the first two rows for question 1 and down the last three
 * for question 2. If the four rows were collapsed into "ICP vs a fast GICP" the
 * reader could not tell which of the two effects they were looking at.
 *
 * A note on the third row: fast_gicp's CUDA offering is FastVGICPCuda, which is
 * *voxelized* GICP - it matches against per-voxel distributions rather than
 * per-point ones. It is labelled VGICP rather than GICP throughout for that
 * reason. There is no plain point-to-point GICP on the GPU in fast_gicp.
 *
 * Runs on two KITTI velodyne scans and scores every method against the KITTI
 * ground-truth poses. With no arguments it uses the pair bundled with this
 * chapter (sequence 04, frames 0 and 1, 1.31 m apart).
 *
 * Usage: ./gicp_demo [source.bin target.bin]
 *
 * Reference: Segal et al., "Generalized-ICP", RSS 2009
 */

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
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

/// Shared iteration budget. Every backend gets the same one, otherwise the
/// timings compare different amounts of work - fast_gicp's LsqRegistration
/// defaults to 64 iterations where PCL's ICP defaults to 10.
constexpr int kMaxIterations = 50;

/// Neighbours used for covariance estimation, matched across PCL and small_gicp
constexpr int kCovarianceNeighbours = 20;

/// Threads for the multi-threaded CPU backend. The GPU has no equivalent knob.
constexpr int kNumThreads = 8;

/// VGICP voxel resolution, for the CUDA backend
constexpr double kVoxelResolution = 1.0;

namespace {

const std::array<uint8_t, 3> kIcpColor{255, 170, 60};
const std::array<uint8_t, 3> kGicpColor{60, 220, 100};
const std::array<uint8_t, 3> kSmallGicpColor{80, 140, 255};
const std::array<uint8_t, 3> kCudaColor{230, 110, 220};

demo::RunResult runPclIcp(const demo::KittiPair& pair, demo::RegistrationViz* viz,
                          demo::ErrorTrace* trace) {
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(kMaxIterations);
    icp.setTransformationEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    icp.setEuclideanFitnessEpsilon(1e-6);

    demo::attachIterationLogging(icp, viz, "PCL ICP", kIcpColor[0], kIcpColor[1],
                                 kIcpColor[2], pair.source,
                                 pair.has_ground_truth ? &pair.ground_truth : nullptr,
                                 trace);

    return demo::runPcl("PCL ICP", icp, pair.source, pair.target, pair.ground_truth,
                        Eigen::Matrix4f::Identity(), trace);
}

demo::RunResult runPclGicp(const demo::KittiPair& pair, demo::RegistrationViz* viz,
                           demo::ErrorTrace* trace) {
    pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
    gicp.setMaximumIterations(kMaxIterations);
    gicp.setTransformationEpsilon(1e-8);
    gicp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    gicp.setEuclideanFitnessEpsilon(1e-6);
    gicp.setCorrespondenceRandomness(kCovarianceNeighbours);
    gicp.setMaximumOptimizerIterations(20);

    demo::attachIterationLogging(gicp, viz, "PCL GICP", kGicpColor[0], kGicpColor[1],
                                 kGicpColor[2], pair.source,
                                 pair.has_ground_truth ? &pair.ground_truth : nullptr,
                                 trace);

    return demo::runPcl("PCL GICP", gicp, pair.source, pair.target, pair.ground_truth,
                        Eigen::Matrix4f::Identity(), trace);
}

demo::SmallGicpConfig smallGicpConfig(int num_threads = kNumThreads) {
    demo::SmallGicpConfig cfg;
    cfg.num_threads = num_threads;
    cfg.num_neighbors = kCovarianceNeighbours;
    cfg.max_correspondence_distance = kMaxCorrespondenceDistance;
    cfg.max_iterations = kMaxIterations;
    return cfg;
}

using CudaVgicp = demo::Traced<fast_gicp::FastVGICPCuda<PointT, PointT>>;

std::unique_ptr<CudaVgicp> makeCudaVgicp() {
    auto reg = std::make_unique<CudaVgicp>();
    reg->setResolution(kVoxelResolution);
    reg->setMaximumIterations(kMaxIterations);
    reg->setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
    // Deliberately NOT calling setCorrespondenceRandomness(): upstream defines it
    // as an empty function body, so it silently does nothing and k stays at 20
    // forever. Calling it would teach a parameter that does not exist.
    return reg;
}

/// Pay the CUDA context and JIT costs before anything is timed
void warmUpGpu(const demo::KittiPair& pair) {
    auto reg = makeCudaVgicp();
    reg->setMaximumIterations(2);
    reg->setInputTarget(pair.target);
    reg->setInputSource(pair.source);
    CloudT aligned;
    reg->align(aligned);
}

/// The main table: four backends, identity initial guess, same points, same budget
void compareBackends(const demo::KittiPair& pair, demo::RegistrationViz* viz) {
    std::cout << "\n=== Experiment 1: ICP vs GICP, and PCL vs small_gicp vs CUDA ==="
              << std::endl;
    std::cout << "Identity initial guess - the full 1.3 m of vehicle motion is left"
              << std::endl;
    std::cout << "for registration to recover.\n" << std::endl;

    std::vector<demo::RunResult> results;
    std::vector<demo::ErrorTrace> traces;

    demo::ErrorTrace icp_trace, gicp_trace;
    results.push_back(runPclIcp(pair, viz, &icp_trace));
    results.push_back(runPclGicp(pair, viz, &gicp_trace));
    traces.push_back(icp_trace);
    traces.push_back(gicp_trace);

    {
        std::vector<demo::TracedStep> poses;
        auto r = demo::runSmallGicpGICP("small_gicp GICP (cpu)", *pair.source,
                                        *pair.target, pair.ground_truth,
                                        Eigen::Matrix4f::Identity(),
                                        smallGicpConfig(), &poses);
        results.push_back(r);
        traces.push_back(demo::traceFromPoses("small_gicp GICP (cpu)", kSmallGicpColor[0],
                                              kSmallGicpColor[1], kSmallGicpColor[2],
                                              poses, pair.ground_truth));
        demo::logTracedSteps(viz, "small_gicp GICP (cpu)", kSmallGicpColor[0],
                             kSmallGicpColor[1], kSmallGicpColor[2], *pair.source, poses);
        viz->logAligned("aligned_small_gicp", *pair.source, r.transform,
                        kSmallGicpColor[0], kSmallGicpColor[1], kSmallGicpColor[2]);
    }

    {
        std::vector<demo::TracedStep> poses;
        auto reg = makeCudaVgicp();
        auto r = demo::runFastGicp("fast_gicp VGICP (cuda)", *reg, pair.source,
                                   pair.target, pair.ground_truth,
                                   Eigen::Matrix4f::Identity(), &poses);
        results.push_back(r);
        traces.push_back(demo::traceFromPoses("fast_gicp VGICP (cuda)", kCudaColor[0],
                                              kCudaColor[1], kCudaColor[2], poses,
                                              pair.ground_truth));
        demo::logTracedSteps(viz, "fast_gicp VGICP (cuda)", kCudaColor[0], kCudaColor[1],
                             kCudaColor[2], *pair.source, poses);
        viz->logAligned("aligned_cuda_vgicp", *pair.source, r.transform, kCudaColor[0],
                        kCudaColor[1], kCudaColor[2]);
    }

    demo::printRunResults(results);

    std::cout << "\nRead the first two rows against each other for the algorithm"
              << std::endl;
    std::cout << "question, and the last three for the implementation question."
              << std::endl;
    std::cout << "Prep and align are split because the backends divide the work"
              << std::endl;
    std::cout << "differently: PCL computes covariances inside align(), while the"
              << std::endl;
    std::cout << "other two build trees and voxel maps when the clouds are handed"
              << std::endl;
    std::cout << "over. Comparing align() alone would flatter them - compare totals."
              << std::endl;

    viz->logErrorCurves(traces);

    viz->logAligned("aligned_pcl_icp", *pair.source, results[0].transform, kIcpColor[0],
                    kIcpColor[1], kIcpColor[2]);
    viz->logAligned("aligned_pcl_gicp", *pair.source, results[1].transform, kGicpColor[0],
                    kGicpColor[1], kGicpColor[2]);

    std::cout << "\n=== small_gicp GICP transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    const auto& best = results.size() > 2 ? results[2] : results[1];
    std::cout << best.transform << std::endl;
    if (pair.has_ground_truth) {
        std::cout << "\n=== Ground truth ===" << std::endl;
        std::cout << pair.ground_truth << std::endl;
    }
}

/**
 * How much wrongness in the initial guess each backend survives
 *
 * This is also where ICP's error shows its true nature. Hand ICP the ground
 * truth itself and it does not stay there - it walks back out to the same ~5.7 cm
 * it reaches from identity. That is a bias, not a convergence failure: point-to-
 * point matching pairs points across the Velodyne's scan rings, and on a road
 * surface those pairings are systematically offset, so the fixed point ICP is
 * pulled to is stable, repeatable and wrong. GICP's surface model is what removes
 * it, which is the clearest single argument for the algorithm.
 */
void testInitialGuess(const demo::KittiPair& pair) {
    if (!pair.has_ground_truth) return;

    std::cout << "\n=== Initial guess robustness ===" << std::endl;
    std::cout << "The guesses are perturbations of the true transform, so they model a"
              << std::endl;
    std::cout << "motion model of decreasing quality. Identity is the no-model case."
              << std::endl;
    std::cout << "Watch the Exact row: a method that cannot hold the right answer when"
              << std::endl;
    std::cout << "handed it is biased, not merely slow to converge.\n" << std::endl;

    const Eigen::Matrix4f& gt = pair.ground_truth;
    const std::vector<std::pair<std::string, Eigen::Matrix4f>> guesses = {
        {"Exact", gt},
        {"0.2 m / 1 deg", gt * demo::makeTransform(0.2f, 0.1f, 0.0f, 0.0f, 0.0f, 0.017f)},
        {"0.5 m / 3 deg", gt * demo::makeTransform(0.5f, 0.3f, 0.1f, 0.0f, 0.0f, 0.052f)},
        {"1.0 m / 6 deg", gt * demo::makeTransform(1.0f, 0.5f, 0.2f, 0.0f, 0.0f, 0.105f)},
        {"Identity", Eigen::Matrix4f::Identity()},
    };

    for (const auto& [name, guess] : guesses) {
        std::cout << "\nInitial guess: " << name << std::endl;
        std::vector<demo::RunResult> results;

        {
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setMaximumIterations(kMaxIterations);
            icp.setTransformationEpsilon(1e-8);
            icp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
            icp.setEuclideanFitnessEpsilon(1e-6);
            results.push_back(demo::runPcl("PCL ICP", icp, pair.source, pair.target, gt,
                                           guess));
        }
        {
            pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
            gicp.setMaximumIterations(kMaxIterations);
            gicp.setTransformationEpsilon(1e-8);
            gicp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
            gicp.setEuclideanFitnessEpsilon(1e-6);
            gicp.setCorrespondenceRandomness(kCovarianceNeighbours);
            results.push_back(demo::runPcl("PCL GICP", gicp, pair.source, pair.target, gt,
                                           guess));
        }
        results.push_back(demo::runSmallGicpGICP("small_gicp GICP (cpu)", *pair.source,
                                                 *pair.target, gt, guess,
                                                 smallGicpConfig()));
        {
            auto reg = makeCudaVgicp();
            results.push_back(demo::runFastGicp("fast_gicp VGICP (cuda)", *reg,
                                                pair.source, pair.target, gt, guess));
        }
        demo::printRunResults(results);
    }
}

/**
 * Where small_gicp's speed actually comes from
 *
 * This replaces the old correspondence-randomness sweep, which moved the error
 * by less than the run-to-run noise and taught nothing. Thread count is the knob
 * that matters for these libraries, and it is the one PCL's GICP does not have.
 */
void testThreadScaling(const demo::KittiPair& pair) {
    std::cout << "\n=== Thread scaling (small_gicp GICP) ===" << std::endl;
    std::cout << "Same points, same iteration budget - only the thread count moves.\n"
              << std::endl;
    std::cout << std::string(104, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Threads" << std::right
              << std::setw(8) << "Conv" << std::setw(7) << "Iters"
              << std::setw(14) << "Trans Err (m)" << std::setw(14) << "Rot Err (deg)"
              << std::setw(13) << "Prep (ms)" << std::setw(13) << "Align (ms)"
              << std::setw(13) << "Total (ms)" << std::endl;
    std::cout << std::string(104, '-') << std::endl;

    std::vector<demo::RunResult> results;
    for (int threads : {1, 2, 4, 8, 16}) {
        auto r = demo::runSmallGicpGICP(std::to_string(threads), *pair.source,
                                        *pair.target, pair.ground_truth,
                                        Eigen::Matrix4f::Identity(),
                                        smallGicpConfig(threads));
        results.push_back(r);
    }
    for (const auto& r : results) {
        std::cout << std::left << std::setw(22) << r.method << std::right
                  << std::setw(8) << (r.converged ? "YES" : "NO")
                  << std::setw(7) << r.iterations << std::fixed
                  << std::setw(14) << std::setprecision(4) << r.error.translation_m
                  << std::setw(14) << std::setprecision(4) << r.error.rotation_deg
                  << std::setw(13) << std::setprecision(1) << r.preprocess_ms
                  << std::setw(13) << std::setprecision(1) << r.align_ms
                  << std::setw(13) << std::setprecision(1) << r.total_ms << std::endl;
    }
    std::cout << std::string(104, '-') << std::endl;
    std::cout << "The accuracy column should barely move: threading changes how fast"
              << std::endl;
    std::cout << "the same answer is reached, not what the answer is. Where it stops"
              << std::endl;
    std::cout << "improving is where this cloud is too small to feed more cores."
              << std::endl;
}

/**
 * CPU / GPU crossover against cloud size
 *
 * A GPU has to earn its fixed overheads back. Sweeping the voxel leaf changes
 * how many points there are to work on, which is exactly the axis that decides
 * whether offloading pays.
 */
void testCpuGpuCrossover(const demo::KittiPair& raw_pair) {
    std::cout << "\n=== CPU / GPU crossover vs cloud size ===" << std::endl;
    std::cout << "Voxelizing the same scan pair at decreasing leaf sizes.\n" << std::endl;
    std::cout << std::string(96, '-') << std::endl;
    std::cout << std::right << std::setw(8) << "Leaf" << std::setw(10) << "Points"
              << std::setw(15) << "CPU total(ms)" << std::setw(15) << "GPU total(ms)"
              << std::setw(11) << "speedup"
              << std::setw(16) << "CPU err (m)" << std::setw(16) << "GPU err (m)"
              << std::endl;
    std::cout << std::string(96, '-') << std::endl;

    for (float leaf : {1.6f, 0.8f, 0.4f, 0.2f}) {
        demo::KittiPair pair = raw_pair;
        pair.source = demo::voxelDownsample(*raw_pair.source, leaf);
        pair.target = demo::voxelDownsample(*raw_pair.target, leaf);

        const auto cpu = demo::runSmallGicpGICP("cpu", *pair.source, *pair.target,
                                                pair.ground_truth,
                                                Eigen::Matrix4f::Identity(),
                                                smallGicpConfig());
        auto reg = makeCudaVgicp();
        const auto gpu = demo::runFastGicp("gpu", *reg, pair.source, pair.target,
                                           pair.ground_truth);

        std::cout << std::right << std::fixed << std::setw(8) << std::setprecision(1)
                  << leaf << std::setw(10) << pair.source->size()
                  << std::setw(15) << std::setprecision(1) << cpu.total_ms
                  << std::setw(15) << std::setprecision(1) << gpu.total_ms
                  << std::setw(10) << std::setprecision(2)
                  << (gpu.total_ms > 0.0 ? cpu.total_ms / gpu.total_ms : 0.0) << "x"
                  << std::setw(16) << std::setprecision(4) << cpu.error.translation_m
                  << std::setw(16) << std::setprecision(4) << gpu.error.translation_m
                  << std::endl;
    }
    std::cout << std::string(96, '-') << std::endl;
    std::cout << "Note the two error columns are not expected to match: the CPU row is"
              << std::endl;
    std::cout << "point-to-point GICP and the GPU row is voxelized GICP, so this sweep"
              << std::endl;
    std::cout << "compares what each backend actually offers, not one algorithm twice."
              << std::endl;
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

}  // namespace

int main(int argc, char** argv) {
    std::cout << "=== GICP experiment ===" << std::endl;
    std::cout << "ICP vs GICP, then PCL vs small_gicp vs CUDA, on KITTI\n" << std::endl;

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

    // Downsample. Voxelizing both scans identically keeps the comparison fair -
    // every method below sees exactly the same points.
    demo::KittiPair pair = raw_pair;
    std::cout << "\nVoxel-downsampling to " << kVoxelLeaf << " m..." << std::endl;
    pair.source = demo::voxelDownsample(*raw_pair.source, kVoxelLeaf);
    pair.target = demo::voxelDownsample(*raw_pair.target, kVoxelLeaf);
    std::cout << "  Source: " << raw_pair.source->size() << " -> " << pair.source->size()
              << std::endl;
    std::cout << "  Target: " << raw_pair.target->size() << " -> " << pair.target->size()
              << std::endl;

    // Stream inputs, per-iteration steps, and results to a rerun viewer
    // (no-op without SDK/viewer)
    demo::RegistrationViz viz("gicp_demo");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    // Creating the CUDA context, allocating on the device and any PTX JIT all
    // happen on the first call. Charging that to the first measured run would
    // report the GPU as hundreds of times slower than it is.
    std::cout << "\nWarming up the GPU (context creation is not part of any timing)..."
              << std::endl;
    warmUpGpu(pair);

    compareBackends(pair, &viz);
    testInitialGuess(pair);

    testThreadScaling(pair);

    testCpuGpuCrossover(raw_pair);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "GICP over plain ICP:" << std::endl;
    std::cout << "  Models local surface structure as Gaussian distributions, on both"
              << std::endl;
    std::cout << "  clouds, so planar structure like road and walls constrains the fit"
              << std::endl;
    std::cout << "  instead of fighting it. Point-to-point ICP pairs points across the"
              << std::endl;
    std::cout << "  scan rings, which biases it on exactly that geometry." << std::endl;
    std::cout << std::endl;
    std::cout << "Implementation over algorithm:" << std::endl;
    std::cout << "  Same cost function, very different cost. Multi-threading, a faster"
              << std::endl;
    std::cout << "  KdTree and leaner data structures account for most of it; the GPU"
              << std::endl;
    std::cout << "  adds more once there are enough points to keep it busy." << std::endl;
    std::cout << std::endl;

    return 0;
}
