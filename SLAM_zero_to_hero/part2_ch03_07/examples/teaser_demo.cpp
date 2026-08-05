/**
 * TEASER++ Demo
 *
 * This example demonstrates the TEASER++ (Truncated least squares Estimation
 * And SEmidefinite Relaxation) algorithm for robust global point cloud
 * registration on KITTI velodyne scans.
 *
 * TEASER++ is a globally optimal, certifiably robust registration method that:
 * - Works with >95% outliers in correspondences
 * - Provides optimality guarantees
 * - Does not require an initial guess (global registration)
 *
 * Key concepts covered:
 * - Global registration vs local registration
 * - Outlier-robust registration
 * - Using TEASER++ for loop closure and relocalization
 *
 * This demo links the real TEASER++ library - there is no fallback
 * implementation, so what the numbers below describe is TEASER++ itself. The
 * CMake target is only built when teaserpp is found.
 *
 * The interesting case for global registration is two scans that no motion
 * model can bridge. Consecutive KITTI scans are only ~1.5 m apart, so also try
 * a real loop closure - see the README for frame pairs in sequence 00 where the
 * vehicle revisits a street from the opposite direction.
 *
 * Usage: ./teaser_demo [source.bin target.bin]
 *
 * Reference: Yang et al., "TEASER: Fast and Certifiable Point Cloud
 *            Registration", IEEE T-RO 2020
 */

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/transformation_estimation_svd.h>

#include <teaser/registration.h>

#include "demo_common.hpp"

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FPFHT = pcl::FPFHSignature33;
using FPFHCloudT = pcl::PointCloud<FPFHT>;

// ---------------------------------------------------------------------------
// Scales. Feature-based global registration is far more sensitive to these than
// ICP is: the descriptor has to see enough surface to be distinctive, and every
// stage after it inherits the keypoint spacing.
// ---------------------------------------------------------------------------

/// Keypoint spacing. A raw KITTI scan is ~124k points; 0.5 m leaves ~17k of them.
/// Halving this from 1.0 m roughly doubles the surviving correspondences and, on
/// the loop-closure pairs, is what takes the global rotation estimate from a few
/// degrees to a few tenths - the descriptors need enough keypoints on the same
/// surfaces for the reciprocal match to find them.
constexpr float kFeatureLeaf = 0.5f;

/// Normal estimation radius - a few keypoint spacings, so a road patch has
/// enough neighbours to define a plane
constexpr float kNormalRadius = 2.0f;

/// FPFH radius. Must exceed the normal radius; this is the size of the
/// neighbourhood the 33-bin descriptor summarises.
constexpr float kFpfhRadius = 4.0f;

/// TEASER's noise bound: how far a true correspondence may be off. At 1.0 m
/// keypoint spacing the quantisation alone is that order, so 0.5 m.
constexpr double kNoiseBound = 0.5;

/// Cap on correspondences handed to TEASER. The max-clique step is the
/// expensive part, so bound it rather than let the scan size decide.
constexpr std::size_t kMaxCorrespondences = 2000;

/// A correspondence counts as a true inlier when the ground-truth transform
/// brings its two points this close
constexpr double kInlierThreshold = 1.0;

/// Keep the solver from disappearing into an exact max-clique search
/// (TEASER's own default time limit is one hour)
constexpr double kMaxCliqueTimeLimit = 60.0;

/**
 * Swallows stdout for the duration of a scope
 *
 * TEASER++ is built with ENABLE_DIAGNOSTIC_PRINT on, which is what makes the
 * headline run below narrate its stages - scale, max clique, GNC-TLS rotation,
 * translation - and that narration is worth seeing once. The outlier sweep calls
 * the solver once per row, where the same chatter would bury the table.
 */
class StdoutSilencer {
public:
    StdoutSilencer() : previous_(std::cout.rdbuf(sink_.rdbuf())) {}
    ~StdoutSilencer() { std::cout.rdbuf(previous_); }

    StdoutSilencer(const StdoutSilencer&) = delete;
    StdoutSilencer& operator=(const StdoutSilencer&) = delete;

private:
    std::ostringstream sink_;
    std::streambuf* previous_;
};

struct GlobalResult {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    bool valid = false;
    std::size_t inliers = 0;
    double time_ms = 0.0;
};

/**
 * Compute normals for a point cloud
 */
NormalCloudT::Ptr computeNormals(const CloudT::Ptr& cloud, float radius) {
    pcl::NormalEstimationOMP<PointT, NormalT> ne;
    ne.setInputCloud(cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);

    NormalCloudT::Ptr normals(new NormalCloudT);
    ne.compute(*normals);

    return normals;
}

/**
 * Compute FPFH features
 */
FPFHCloudT::Ptr computeFPFH(const CloudT::Ptr& cloud, const NormalCloudT::Ptr& normals,
                            float radius) {
    pcl::FPFHEstimationOMP<PointT, NormalT, FPFHT> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    fpfh.setSearchMethod(tree);
    fpfh.setRadiusSearch(radius);

    FPFHCloudT::Ptr features(new FPFHCloudT);
    fpfh.compute(*features);

    return features;
}

/**
 * Reciprocal FPFH matches, closest first, capped at kMaxCorrespondences
 *
 * Reciprocal means source -> target and target -> source agree, which throws
 * away a large fraction of the one-way matches for free. What survives is still
 * mostly wrong on street scans, which is the point of the exercise.
 */
std::vector<std::pair<int, int>> matchFeatures(const FPFHCloudT::Ptr& source_features,
                                               const FPFHCloudT::Ptr& target_features) {
    pcl::registration::CorrespondenceEstimation<FPFHT, FPFHT> est;
    est.setInputSource(source_features);
    est.setInputTarget(target_features);

    pcl::Correspondences correspondences;
    est.determineReciprocalCorrespondences(correspondences);

    std::sort(correspondences.begin(), correspondences.end(),
              [](const pcl::Correspondence& a, const pcl::Correspondence& b) {
                  return a.distance < b.distance;
              });

    std::vector<std::pair<int, int>> pairs;
    const std::size_t keep = std::min(correspondences.size(), kMaxCorrespondences);
    pairs.reserve(keep);
    for (std::size_t i = 0; i < keep; ++i) {
        pairs.push_back({correspondences[i].index_query, correspondences[i].index_match});
    }
    return pairs;
}

/**
 * Run TEASER++ on a correspondence set
 */
GlobalResult runTeaser(const CloudT& source, const CloudT& target,
                       const std::vector<std::pair<int, int>>& correspondences,
                       teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE mode) {
    GlobalResult result;
    if (correspondences.empty()) {
        return result;
    }

    // TEASER takes the matched points as two parallel 3xN matrices, already
    // paired up: column i of src corresponds to column i of tgt
    const auto n = static_cast<Eigen::Index>(correspondences.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> src(3, n);
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt(3, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& [si, ti] = correspondences[static_cast<std::size_t>(i)];
        src.col(i) = source[si].getVector3fMap().cast<double>();
        tgt.col(i) = target[ti].getVector3fMap().cast<double>();
    }

    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = kNoiseBound;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;   // rigid body: LiDAR scans share a scale
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;
    params.inlier_selection_mode = mode;
    params.max_clique_time_limit = kMaxCliqueTimeLimit;

    teaser::RobustRegistrationSolver solver(params);

    const auto start = std::chrono::high_resolution_clock::now();
    solver.solve(src, tgt);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto solution = solver.getSolution();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.valid = solution.valid;
    result.inliers = solver.getInlierMaxClique().size();
    result.transform.block<3, 3>(0, 0) = solution.rotation.cast<float>();
    result.transform.block<3, 1>(0, 3) = solution.translation.cast<float>();

    return result;
}

/**
 * RANSAC + SVD on the same correspondence set - the classic robust estimator,
 * used here as the baseline TEASER++ is measured against
 */
GlobalResult runRansac(const CloudT::Ptr& source, const CloudT::Ptr& target,
                       const std::vector<std::pair<int, int>>& correspondences) {
    GlobalResult result;
    if (correspondences.size() < 3) {
        return result;
    }

    pcl::CorrespondencesPtr input(new pcl::Correspondences);
    input->reserve(correspondences.size());
    for (const auto& [si, ti] : correspondences) {
        input->push_back(pcl::Correspondence(si, ti, 0.0f));
    }

    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ransac;
    ransac.setInputSource(source);
    ransac.setInputTarget(target);
    ransac.setInlierThreshold(kInlierThreshold);
    ransac.setMaximumIterations(10000);

    pcl::Correspondences inliers;

    const auto start = std::chrono::high_resolution_clock::now();
    ransac.getRemainingCorrespondences(*input, inliers);
    const auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.inliers = inliers.size();

    if (inliers.size() >= 3) {
        pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
        svd.estimateRigidTransformation(*source, *target, inliers, result.transform);
        result.valid = true;
    }
    return result;
}

/**
 * Split the residual translation error along the target sensor's own axes
 *
 * A single error magnitude hides which direction the estimate is weak in, and on
 * street scans those directions are not equivalent. The velodyne frame is
 * x-forward, y-left, z-up, and the residual is already expressed in the target
 * frame, so its components read off directly.
 */
void printErrorDirections(const Eigen::Matrix4f& estimated,
                          const Eigen::Matrix4f& ground_truth) {
    const Eigen::Vector3f err =
        estimated.block<3, 1>(0, 3) - ground_truth.block<3, 1>(0, 3);

    std::cout << "\n  Residual translation, by direction (velodyne axes):" << std::endl;
    std::cout << "    along  track (x): " << std::fixed << std::setprecision(3) << err.x()
              << " m" << std::endl;
    std::cout << "    across track (y): " << err.y() << " m" << std::endl;
    std::cout << "    vertical     (z): " << err.z() << " m" << std::endl;
    std::cout << "  A street is close to translation-invariant along its own axis, so the"
              << std::endl;
    std::cout << "  along-track component is the weakly observable one and is normally the"
              << std::endl;
    std::cout << "  largest of the three. Sliding a scan a meter down the road barely"
              << std::endl;
    std::cout << "  changes how well it fits; sliding it sideways into a wall does."
              << std::endl;
}

/**
 * Which correspondences the ground truth confirms
 */
std::vector<std::pair<int, int>> trueInliers(
    const CloudT& source, const CloudT& target,
    const std::vector<std::pair<int, int>>& correspondences,
    const Eigen::Matrix4f& ground_truth) {
    std::vector<std::pair<int, int>> inliers;
    for (const auto& [si, ti] : correspondences) {
        const Eigen::Vector4f sp(source[si].x, source[si].y, source[si].z, 1.0f);
        const Eigen::Vector3f mapped = (ground_truth * sp).head<3>();
        if ((mapped - target[ti].getVector3fMap()).norm() < kInlierThreshold) {
            inliers.push_back({si, ti});
        }
    }
    return inliers;
}

/**
 * Measure how both estimators degrade as the correspondence set is poisoned
 *
 * The ground truth splits the FPFH matches into true inliers and the rest, then
 * each row builds a fresh set of the same size with a prescribed fraction of
 * random (therefore wrong) pairs. Same data, same size, only the inlier ratio
 * changes - so the columns are directly comparable.
 */
void measureOutlierRobustness(const CloudT::Ptr& source, const CloudT::Ptr& target,
                              const std::vector<std::pair<int, int>>& correspondences,
                              const Eigen::Matrix4f& ground_truth) {
    std::cout << "\n=== Outlier Robustness (measured) ===" << std::endl;

    const std::vector<std::pair<int, int>> inliers =
        trueInliers(*source, *target, correspondences, ground_truth);

    std::cout << "Ground truth confirms " << inliers.size() << " of " << correspondences.size()
              << " FPFH matches (" << std::fixed << std::setprecision(1)
              << 100.0 * static_cast<double>(inliers.size()) /
                     static_cast<double>(correspondences.size())
              << " % inliers, threshold " << std::setprecision(2) << kInlierThreshold
              << " m)" << std::endl;

    if (inliers.size() < 10) {
        std::cout << "Too few confirmed inliers to build controlled sets - skipping."
                  << std::endl;
        std::cout << "Note that this says as much about the ground truth as about the"
                  << std::endl;
        std::cout << "features. A match is only counted when the ground-truth transform"
                  << std::endl;
        std::cout << "brings its two points within " << std::setprecision(2)
                  << kInlierThreshold << " m, so on a revisit pair - where the"
                  << std::endl;
        std::cout << "ground truth has thousands of frames of drift in it and can sit"
                  << std::endl;
        std::cout << "further out than that on its own (see the residual breakdown above)"
                  << std::endl;
        std::cout << "- hardly anything can be confirmed however good the features are."
                  << std::endl;
        std::cout << "Run the sweep on a consecutive pair, where the ground truth is tight."
                  << std::endl;
        return;
    }

    // Every row keeps the same total, so cost differences are not size effects.
    // The row with the FEWEST outliers needs the most confirmed inliers, so it
    // sets the size: at a 50 % outlier ratio half of the set has to be real.
    const std::vector<double> ratios = {0.5, 0.7, 0.9, 0.95, 0.99};
    const double min_ratio = *std::min_element(ratios.begin(), ratios.end());
    const std::size_t total = std::min(
        kMaxCorrespondences,
        static_cast<std::size_t>(static_cast<double>(inliers.size()) / (1.0 - min_ratio)));

    std::cout << "Each row below is a set of " << total << " correspondences with the given"
              << std::endl;
    std::cout << "fraction replaced by random pairs.\n" << std::endl;

    std::cout << std::string(88, '-') << std::endl;
    std::cout << std::setw(10) << "Outliers"
              << std::setw(14) << "TEASER++ (m)"
              << std::setw(14) << "TEASER++ (deg)"
              << std::setw(11) << "ms"
              << std::setw(14) << "RANSAC (m)"
              << std::setw(14) << "RANSAC (deg)"
              << std::setw(11) << "ms" << std::endl;
    std::cout << std::string(88, '-') << std::endl;

    std::mt19937 gen(42);  // fixed seed: the table is reproducible
    std::uniform_int_distribution<int> src_pick(0, static_cast<int>(source->size()) - 1);
    std::uniform_int_distribution<int> tgt_pick(0, static_cast<int>(target->size()) - 1);

    for (double ratio : ratios) {
        const std::size_t want_inliers =
            std::max<std::size_t>(3, static_cast<std::size_t>(total * (1.0 - ratio)));
        if (want_inliers > inliers.size()) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(0) << ratio * 100
                      << "%   skipped: needs " << want_inliers
                      << " confirmed inliers, only " << inliers.size() << " available"
                      << std::endl;
            continue;
        }

        std::vector<std::pair<int, int>> mix;
        mix.reserve(total);

        std::vector<std::size_t> order(inliers.size());
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), gen);
        for (std::size_t i = 0; i < want_inliers; ++i) {
            mix.push_back(inliers[order[i]]);
        }
        while (mix.size() < total) {
            mix.push_back({src_pick(gen), tgt_pick(gen)});
        }
        std::shuffle(mix.begin(), mix.end(), gen);

        GlobalResult teaser, ransac;
        {
            const StdoutSilencer quiet;
            // KCORE_HEU keeps the max-clique step affordable at these sizes; the
            // exact solver is used for the headline run above
            teaser = runTeaser(
                *source, *target, mix,
                teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::KCORE_HEU);
            ransac = runRansac(source, target, mix);
        }

        const demo::PoseError te = demo::poseError(teaser.transform, ground_truth);
        const demo::PoseError re = demo::poseError(ransac.transform, ground_truth);

        std::cout << std::setw(9) << std::fixed << std::setprecision(0) << ratio * 100 << "%"
                  << std::setw(14) << std::setprecision(3) << te.translation_m
                  << std::setw(14) << std::setprecision(3) << te.rotation_deg
                  << std::setw(11) << std::setprecision(0) << teaser.time_ms
                  << std::setw(14) << std::setprecision(3) << re.translation_m
                  << std::setw(14) << std::setprecision(3) << re.rotation_deg
                  << std::setw(11) << std::setprecision(0) << ransac.time_ms << std::endl;
    }
    std::cout << std::string(88, '-') << std::endl;

    std::cout << "\nTEASER++ decouples scale, rotation and translation and solves each with"
              << std::endl;
    std::cout << "truncated least squares, the rotation via graduated non-convexity, after"
              << std::endl;
    std::cout << "a max-clique pass on the invariant (TIM) graph prunes the outliers. That"
              << std::endl;
    std::cout << "pruning is what survives ratios where sampling a clean minimal set at"
              << std::endl;
    std::cout << "random has become hopeless for RANSAC." << std::endl;
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
    std::cout << "Global registration earns its cost on pairs no motion model can bridge."
              << std::endl;
    std::cout << "Try a loop closure in sequence 00, where the vehicle drives back down the"
              << std::endl;
    std::cout << "same street facing the other way:" << std::endl;
    std::cout << "  " << prog_name << " <kitti>/sequences/00/velodyne/001539.bin"
              << " <kitti>/sequences/00/velodyne/004540.bin" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== TEASER++ Global Registration Demo ===" << std::endl;
    std::cout << "Robust, certifiably optimal registration on KITTI\n" << std::endl;
    std::cout << "TEASER++ library: linked (this demo has no fallback implementation)\n"
              << std::endl;

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

    // Downsample to the keypoint set the descriptors are computed on
    std::cout << "\nVoxel-downsampling to " << kFeatureLeaf << " m for feature extraction..."
              << std::endl;
    const std::size_t source_raw = pair.source->size();
    const std::size_t target_raw = pair.target->size();
    pair.source = demo::voxelDownsample(*pair.source, kFeatureLeaf);
    pair.target = demo::voxelDownsample(*pair.target, kFeatureLeaf);
    std::cout << "  Source: " << source_raw << " -> " << pair.source->size() << std::endl;
    std::cout << "  Target: " << target_raw << " -> " << pair.target->size() << std::endl;

    demo::RegistrationViz viz("teaser_demo");
    viz.logCloudByHeight("target", *pair.target);
    viz.logCloud("source_initial", *pair.source, 235, 80, 80);

    // ====================================
    // Global registration, no initial guess
    // ====================================

    std::cout << "\n=== Global Registration ===" << std::endl;
    std::cout << "Goal: find the transformation WITHOUT an initial guess" << std::endl;

    std::cout << "\n  Computing normals (radius " << kNormalRadius << " m)..." << std::endl;
    const auto source_normals = computeNormals(pair.source, kNormalRadius);
    const auto target_normals = computeNormals(pair.target, kNormalRadius);

    std::cout << "  Computing FPFH features (radius " << kFpfhRadius << " m)..." << std::endl;
    const auto source_fpfh = computeFPFH(pair.source, source_normals, kFpfhRadius);
    const auto target_fpfh = computeFPFH(pair.target, target_normals, kFpfhRadius);

    std::cout << "  Matching features (reciprocal nearest neighbours)..." << std::endl;
    const std::vector<std::pair<int, int>> correspondences =
        matchFeatures(source_fpfh, target_fpfh);
    std::cout << "  Correspondences: " << correspondences.size() << std::endl;

    if (correspondences.size() < 3) {
        std::cerr << "Error: too few correspondences to register." << std::endl;
        return -1;
    }

    viz.logCorrespondences("correspondences", *pair.source, *pair.target, correspondences,
                           120, 120, 140);

    std::cout << "\n  Running TEASER++ (exact max clique, noise bound " << kNoiseBound
              << " m)..." << std::endl;
    const GlobalResult global = runTeaser(
        *pair.source, *pair.target, correspondences,
        teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_EXACT);

    const demo::PoseError global_err = demo::poseError(global.transform, pair.ground_truth);

    std::cout << "\n  TEASER++ results:" << std::endl;
    std::cout << "    Solution valid:     " << (global.valid ? "YES" : "NO") << std::endl;
    std::cout << "    Max-clique inliers: " << global.inliers << " / "
              << correspondences.size() << std::endl;
    std::cout << "    Translation error:  " << std::fixed << std::setprecision(4)
              << global_err.translation_m << " m" << std::endl;
    std::cout << "    Rotation error:     " << std::setprecision(4) << global_err.rotation_deg
              << " deg" << std::endl;
    std::cout << "    Time:               " << std::setprecision(1) << global.time_ms << " ms"
              << std::endl;

    viz.logAligned("aligned_teaser", *pair.source, global.transform, 60, 220, 100);

    // ====================================
    // Refine with GICP
    // ====================================

    // Coarse to fine. A global estimate can be a couple of meters out - on the
    // loop-closure pairs it usually is - and GICP can only pull in what its
    // correspondence distance reaches, so a single fine pass would leave that
    // error sitting there. Shrinking the search radius over three passes lets the
    // first one close the meters and the last one deliver the precision.
    std::cout << "\n  Refining with GICP (coarse to fine)..." << std::endl;

    const std::vector<double> refine_distances = {8.0, 3.0, 1.0};

    Eigen::Matrix4f refined = global.transform;
    const auto refine_start = std::chrono::high_resolution_clock::now();

    for (double max_distance : refine_distances) {
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
        gicp.setInputSource(pair.source);
        gicp.setInputTarget(pair.target);
        gicp.setMaximumIterations(50);
        gicp.setMaxCorrespondenceDistance(max_distance);
        gicp.setCorrespondenceRandomness(20);
        demo::attachIterationLogging(gicp, &viz, "gicp_refine", 255, 170, 60);

        CloudT::Ptr refined_cloud(new CloudT);
        gicp.align(*refined_cloud, refined);
        refined = gicp.getFinalTransformation();

        const demo::PoseError step_err = demo::poseError(refined, pair.ground_truth);
        std::cout << "    max correspondence " << std::fixed << std::setprecision(1)
                  << max_distance << " m -> " << std::setprecision(4)
                  << step_err.translation_m << " m, " << step_err.rotation_deg << " deg"
                  << std::endl;
    }
    const auto refine_end = std::chrono::high_resolution_clock::now();

    const demo::PoseError refined_err = demo::poseError(refined, pair.ground_truth);

    std::cout << "    Translation error:  " << std::fixed << std::setprecision(4)
              << refined_err.translation_m << " m" << std::endl;
    std::cout << "    Rotation error:     " << std::setprecision(4) << refined_err.rotation_deg
              << " deg" << std::endl;
    std::cout << "    Time:               " << std::setprecision(1)
              << std::chrono::duration<double, std::milli>(refine_end - refine_start).count()
              << " ms" << std::endl;

    viz.logAligned("aligned_refined", *pair.source, refined, 255, 170, 60);

    if (pair.has_ground_truth) {
        printErrorDirections(refined, pair.ground_truth);
    }

    std::cout << "\n=== Estimated Transformation (TEASER++ then GICP) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6) << refined << std::endl;
    if (pair.has_ground_truth) {
        std::cout << "=== Ground Truth ===" << std::endl;
        std::cout << pair.ground_truth << std::endl;
    }

    // ====================================
    // Outlier robustness, measured on this scan pair
    // ====================================

    if (pair.has_ground_truth) {
        measureOutlierRobustness(pair.source, pair.target, correspondences,
                                 pair.ground_truth);
    } else {
        std::cout << "\n=== Outlier Robustness (measured) ===" << std::endl;
        std::cout << "Skipped: splitting the matches into inliers and outliers needs the"
                  << std::endl;
        std::cout << "ground-truth transform, which is not available for this scan pair."
                  << std::endl;
    }

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "When to use TEASER++:" << std::endl;
    std::cout << "  1. Loop closure detection and verification" << std::endl;
    std::cout << "  2. Relocalization after tracking loss" << std::endl;
    std::cout << "  3. Initial alignment before ICP/GICP" << std::endl;
    std::cout << "  4. When correspondences contain many outliers" << std::endl;
    std::cout << std::endl;
    std::cout << "Typical pipeline (what this demo runs):" << std::endl;
    std::cout << "  1. Extract features (FPFH here; SHOT, learned descriptors, ...)"
              << std::endl;
    std::cout << "  2. Find putative correspondences" << std::endl;
    std::cout << "  3. Run TEASER++ for a robust transformation" << std::endl;
    std::cout << "  4. Refine with ICP/GICP" << std::endl;
    std::cout << std::endl;

    return 0;
}
