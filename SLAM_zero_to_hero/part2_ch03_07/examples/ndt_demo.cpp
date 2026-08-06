/**
 * NDT experiment: PCL's CPU NDT against fast_gicp's CUDA NDT
 *
 * Only these two exist to compare: NDTCuda is the only NDT in fast_gicp and is
 * CUDA-gated, and small_gicp has no NDT at all - its VGICP is GICP against a
 * voxel map, a different cost function, not NDT renamed.
 *
 * fast_gicp's two distance modes:
 *   P2D  source points against the target's voxel distributions (classic NDT)
 *   D2D  source voxelized too, distribution against distribution. The default.
 *
 * The GPU is much faster and, on this data, less accurate - hence both columns.
 *
 * Usage: ./ndt_demo [source.bin target.bin [resolution]]   (no args: bundled pair)
 *
 * Reference: Biber & Strasser, "The Normal Distributions Transform", IROS 2003
 */

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
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

constexpr float kVoxelLeaf = 0.3f;

/// The two backends do not peak at the same cell size, so each runs at its own -
/// a shared value only handicaps whichever it does not suit. See the resolution
/// study: PCL is best at 2.0 m and stalls below 1.0 m, while both CUDA modes peak
/// at 0.75 m and fall apart at 0.5 m.
constexpr float kNdtResolution = 1.0f;
constexpr float kCudaNdtResolution = 0.75f;

/// More-Thuente maximum step length, for PCL's NDT. 0.1 m - the value indoor
/// NDT examples use - is shorter than TransformationEpsilon at KITTI's 1.3-1.5 m
/// scan spacing, so NDT reports convergence after a single iteration without
/// having moved. See the step size study below.
constexpr float kNdtStepSize = 0.5f;

constexpr int kMaxIterations = 50;

namespace {

const std::array<uint8_t, 3> kPclColor{80, 140, 255};
const std::array<uint8_t, 3> kD2dColor{230, 110, 220};
const std::array<uint8_t, 3> kP2dColor{90, 220, 210};

demo::RunResult runPclNdt(const demo::KittiPair& pair, float resolution,
                          float step_size = kNdtStepSize,
                          const Eigen::Matrix4f& guess = Eigen::Matrix4f::Identity(),
                          demo::RegistrationViz* viz = nullptr,
                          demo::ErrorTrace* trace = nullptr,
                          const std::string& label = "PCL NDT") {
    pcl::NormalDistributionsTransform<PointT, PointT> ndt;

    // Set the resolution BEFORE the target cloud. setInputTarget() builds the
    // voxel-covariance grid immediately, using whatever resolution is set at that
    // moment; a later setResolution() throws that grid away and builds another.
    // Besides the wasted work, the discarded grid is built at PCL's 1 m default,
    // which on a sparse cloud has too few points per cell and prints a scary
    // "Grid will not be searchable" warning that has nothing to do with the run.
    ndt.setResolution(resolution);
    ndt.setStepSize(step_size);
    ndt.setTransformationEpsilon(0.01);
    ndt.setMaximumIterations(kMaxIterations);

    demo::attachIterationLogging(ndt, viz, label, kPclColor[0], kPclColor[1],
                                 kPclColor[2], pair.source,
                                 pair.has_ground_truth ? &pair.ground_truth : nullptr,
                                 trace, &guess);

    return demo::runPcl(label, ndt, pair.source, pair.target, pair.ground_truth, guess,
                        trace);
}

using CudaNdt = demo::Traced<fast_gicp::NDTCuda<PointT, PointT>>;

std::unique_ptr<CudaNdt> makeCudaNdt(float resolution,
                                     fast_gicp::NDTDistanceMode mode) {
    auto reg = std::make_unique<CudaNdt>();
    reg->setResolution(resolution);
    reg->setDistanceMode(mode);
    reg->setMaximumIterations(kMaxIterations);
    // NDTCuda has no setNumThreads and no settable regularization - the voxel
    // covariances are always regularized with MIN_EIG, hardcoded upstream. Its
    // neighbour search defaults to DIRECT7, unlike FastVGICP's DIRECT1.
    return reg;
}

void warmUpGpu(const demo::KittiPair& pair) {
    auto reg = makeCudaNdt(kCudaNdtResolution, fast_gicp::NDTDistanceMode::D2D);
    reg->setMaximumIterations(2);
    reg->setInputTarget(pair.target);
    reg->setInputSource(pair.source);
    CloudT aligned;
    reg->align(aligned);
}

void compareBackends(const demo::KittiPair& pair, float pcl_resolution,
                     float cuda_resolution, demo::RegistrationViz* viz) {
    std::cout << "\n=== Experiment 2: CPU NDT vs CUDA NDT ===" << std::endl;
    std::cout << "Identity initial guess. Each backend runs at its own best cell"
              << std::endl;
    std::cout << "size - PCL " << std::fixed << std::setprecision(2) << pcl_resolution
              << " m, CUDA " << cuda_resolution << " m - since they do not peak at"
              << std::endl;
    std::cout << "the same value. See the resolution study below.\n" << std::endl;

    std::vector<demo::RunResult> results;
    std::vector<demo::ErrorTrace> traces;

    demo::ErrorTrace pcl_trace;
    auto pcl_result = runPclNdt(pair, pcl_resolution, kNdtStepSize,
                                Eigen::Matrix4f::Identity(), viz, &pcl_trace);
    results.push_back(pcl_result);
    traces.push_back(pcl_trace);
    viz->logAligned("aligned_pcl_ndt", *pair.source, pcl_result.transform, kPclColor[0],
                    kPclColor[1], kPclColor[2]);

    const std::array<std::pair<const char*, fast_gicp::NDTDistanceMode>, 2> modes{
        {{"NDTCuda (D2D)", fast_gicp::NDTDistanceMode::D2D},
         {"NDTCuda (P2D)", fast_gicp::NDTDistanceMode::P2D}}};

    for (std::size_t i = 0; i < modes.size(); ++i) {
        const auto& [name, mode] = modes[i];
        const auto& color = (i == 0) ? kD2dColor : kP2dColor;

        std::vector<demo::TracedStep> poses;
        auto reg = makeCudaNdt(cuda_resolution, mode);
        auto r = demo::runFastGicp(name, *reg, pair.source, pair.target,
                                   pair.ground_truth, Eigen::Matrix4f::Identity(),
                                   &poses);
        results.push_back(r);
        traces.push_back(demo::traceFromPoses(name, color[0], color[1], color[2], poses,
                                              pair.ground_truth));
        demo::logTracedSteps(viz, name, color[0], color[1], color[2], *pair.source,
                             poses);
        viz->logAligned(std::string("aligned_") + (i == 0 ? "d2d" : "p2d"), *pair.source,
                        r.transform, color[0], color[1], color[2]);
    }

    demo::printRunResults(results);

    std::cout << "\nNDTCuda builds its voxel maps at the top of its own align(), so"
              << std::endl;
    std::cout << "unlike the other backends that cost lands in the align column rather"
              << std::endl;
    std::cout << "than prep. Compare on total." << std::endl;

    viz->logErrorCurves(traces);

    std::cout << "\n=== PCL NDT transformation ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << pcl_result.transform << std::endl;
    if (pair.has_ground_truth) {
        std::cout << "\n=== Ground truth ===" << std::endl;
        std::cout << pair.ground_truth << std::endl;
    }
}

/// Cell size, swept on both backends
void testResolution(const demo::KittiPair& pair) {
    std::cout << "\n=== Resolution study ===" << std::endl;
    std::cout << "Cell size trades precision against the width of the basin of"
              << std::endl;
    std::cout << "convergence, and it is the one parameter both backends share.\n"
              << std::endl;
    std::cout << std::string(104, '-') << std::endl;
    std::cout << std::left << std::setw(22) << "Backend / resolution" << std::right
              << std::setw(8) << "Conv" << std::setw(7) << "Iters"
              << std::setw(14) << "Trans Err (m)" << std::setw(14) << "Rot Err (deg)"
              << std::setw(13) << "Prep (ms)" << std::setw(13) << "Align (ms)"
              << std::setw(13) << "Total (ms)" << std::endl;
    std::cout << std::string(104, '-') << std::endl;

    // The sweep runs below the 0.5 m the two CUDA modes prefer, so their optimum
    // is located rather than assumed to sit at the edge of the range.
    const std::vector<float> resolutions{0.25f, 0.5f, 0.75f, 1.0f, 2.0f, 3.0f, 5.0f};

    const auto label = [](const char* backend, float resolution) {
        std::ostringstream out;
        out << backend << " " << std::fixed << std::setprecision(2) << resolution << " m";
        return out.str();
    };

    std::vector<demo::RunResult> results;
    for (float resolution : resolutions) {
        results.push_back(runPclNdt(pair, resolution, kNdtStepSize,
                                    Eigen::Matrix4f::Identity(), nullptr, nullptr,
                                    label("PCL", resolution)));
    }
    for (float resolution : resolutions) {
        auto reg = makeCudaNdt(resolution, fast_gicp::NDTDistanceMode::D2D);
        results.push_back(demo::runFastGicp(label("CUDA D2D", resolution), *reg,
                                            pair.source, pair.target,
                                            pair.ground_truth));
    }
    for (float resolution : resolutions) {
        auto reg = makeCudaNdt(resolution, fast_gicp::NDTDistanceMode::P2D);
        results.push_back(demo::runFastGicp(label("CUDA P2D", resolution), *reg,
                                            pair.source, pair.target,
                                            pair.ground_truth));
    }

    for (const auto& r : results) {
        std::cout << std::left << std::setw(22) << r.method << std::right
                  << std::setw(8) << (r.converged ? "YES" : "NO");
        if (r.iterations >= 0) std::cout << std::setw(7) << r.iterations;
        else                   std::cout << std::setw(7) << "-";
        std::cout << std::fixed
                  << std::setw(14) << std::setprecision(4) << r.error.translation_m
                  << std::setw(14) << std::setprecision(4) << r.error.rotation_deg
                  << std::setw(13) << std::setprecision(1) << r.preprocess_ms
                  << std::setw(13) << std::setprecision(1) << r.align_ms
                  << std::setw(13) << std::setprecision(1) << r.total_ms << std::endl;
    }
    std::cout << std::string(104, '-') << std::endl;
    std::cout << "Smaller cells resolve more detail but narrow the basin, so a coarse"
              << std::endl;
    std::cout << "initial guess is likelier to fail. Typical outdoor values are"
              << std::endl;
    std::cout << "0.5-2.0 m; indoor work runs 0.1-0.5 m." << std::endl;
}

/**
 * The step-size trap, on PCL's NDT
 *
 * Kept even though it is PCL-specific, because it is the single easiest way to
 * get a silently wrong NDT result, and PCL's NDT is half of this experiment.
 * fast_gicp's NDT has no equivalent knob - it does not use a More-Thuente line
 * search - so there is nothing to sweep on the GPU side.
 */
void testStepSize(const demo::KittiPair& pair) {
    std::cout << "\n=== Step size study (PCL NDT) ===" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    std::cout << std::right << std::setw(12) << "Step Size" << std::setw(12) << "Converged"
              << std::setw(10) << "Iters" << std::setw(16) << "Trans Err (m)"
              << std::setw(16) << "Rot Err (deg)" << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (float step : {0.01f, 0.05f, 0.10f, 0.50f, 1.00f}) {
        const auto r = runPclNdt(pair, kNdtResolution, step);
        std::cout << std::right << std::fixed << std::setw(12) << std::setprecision(2)
                  << step << std::setw(12) << (r.converged ? "YES" : "NO")
                  << std::setw(10) << r.iterations
                  << std::setw(16) << std::setprecision(4) << r.error.translation_m
                  << std::setw(16) << std::setprecision(4) << r.error.rotation_deg
                  << std::endl;
    }
    std::cout << std::string(78, '-') << std::endl;
    std::cout << "The step size is the MAXIMUM length of the line search step, so it"
              << std::endl;
    std::cout << "has to be set against the displacement being recovered. At KITTI's"
              << std::endl;
    std::cout << "1.3-1.5 m scan spacing the small steps stall: the first step comes"
              << std::endl;
    std::cout << "out shorter than TransformationEpsilon, NDT stops after one"
              << std::endl;
    std::cout << "iteration, and nearly the whole vehicle motion is left as error."
              << std::endl;
    std::cout << "Those rows still report Converged = YES. PCL's flag only means the"
              << std::endl;
    std::cout << "update fell below the epsilon - it is not a statement about"
              << std::endl;
    std::cout << "correctness. Always read it next to the iteration count." << std::endl;
}

/// How much wrongness in the initial guess each backend survives
void testInitialGuess(const demo::KittiPair& pair) {
    if (!pair.has_ground_truth) return;

    std::cout << "\n=== Initial guess robustness ===" << std::endl;
    std::cout << "The guesses are perturbations of the true transform, so they model a"
              << std::endl;
    std::cout << "motion model of decreasing quality. Identity is the no-model case.\n"
              << std::endl;

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
        results.push_back(runPclNdt(pair, kNdtResolution, kNdtStepSize, guess));
        {
            auto reg = makeCudaNdt(kCudaNdtResolution, fast_gicp::NDTDistanceMode::D2D);
            results.push_back(demo::runFastGicp("NDTCuda (D2D)", *reg, pair.source,
                                                pair.target, gt, guess));
        }
        demo::printRunResults(results);
    }

    std::cout << "\nNDT's basin is narrow in rotation and wide in translation: the"
              << std::endl;
    std::cout << "identity guess is off by the full 1.31 m and is handled fine, while"
              << std::endl;
    std::cout << "the 1.0 m / 6 deg guess is off by less translation and fails. A motion"
              << std::endl;
    std::cout << "model that gets the heading wrong hurts more than having no model."
              << std::endl;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.bin target.bin [resolution]]"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments      - Use the bundled KITTI pair (seq 04, frames 0-1)"
              << std::endl;
    std::cout << "  source target     - Use the given KITTI velodyne .bin (or .pcd) scans"
              << std::endl;
    std::cout << "  resolution        - NDT cell size in meters (default "
              << kNdtResolution << " PCL / " << kCudaNdtResolution
              << " CUDA; giving one applies it to both)" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout << "=== NDT experiment ===" << std::endl;
    std::cout << "PCL's CPU NDT vs fast_gicp's CUDA NDT, on KITTI\n" << std::endl;

    if (argc != 1 && argc != 3 && argc != 4) {
        printUsage(argv[0]);
        return -1;
    }

    demo::KittiPair pair = (argc >= 3) ? demo::loadKittiPair(argv[1], argv[2])
                                       : demo::loadKittiPair();
    if (!pair.source || !pair.target) {
        return -1;
    }

    // An explicit cell size on the command line applies to both backends, since
    // asking for one value means wanting them compared at it. Without it each
    // takes its own tuned default.
    const bool shared = (argc == 4);
    const float pcl_resolution = shared ? std::stof(argv[3]) : kNdtResolution;
    const float cuda_resolution = shared ? std::stof(argv[3]) : kCudaNdtResolution;

    demo::printKittiPair(pair);
    std::cout << "  NDT resolution: PCL " << std::fixed << std::setprecision(2)
              << pcl_resolution << " m, CUDA " << cuda_resolution << " m" << std::endl;

    std::cout << "\nVoxel-downsampling to " << kVoxelLeaf << " m..." << std::endl;
    const std::size_t source_raw = pair.source->size();
    const std::size_t target_raw = pair.target->size();
    pair.source = demo::voxelDownsample(*pair.source, kVoxelLeaf);
    pair.target = demo::voxelDownsample(*pair.target, kVoxelLeaf);
    std::cout << "  Source: " << source_raw << " -> " << pair.source->size() << std::endl;
    std::cout << "  Target: " << target_raw << " -> " << pair.target->size() << std::endl;

    demo::RegistrationViz viz("ndt_demo");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    std::cout << "\nWarming up the GPU (context creation is not part of any timing)..."
              << std::endl;
    warmUpGpu(pair);

    compareBackends(pair, pcl_resolution, cuda_resolution, &viz);
    testResolution(pair);
    testStepSize(pair);
    testInitialGuess(pair);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "NDT represents the target as a grid of Gaussians, so there are no"
              << std::endl;
    std::cout << "explicit correspondences to search and the cost is linear in the"
              << std::endl;
    std::cout << "number of source points - which is what makes it suit a GPU."
              << std::endl;
    std::cout << std::endl;
    std::cout << "Speed is not free, though. Read the accuracy columns next to the"
              << std::endl;
    std::cout << "timings before concluding the GPU version is simply better."
              << std::endl;
    std::cout << std::endl;

    return 0;
}
