/**
 * TEASER++ Demo
 *
 * This example demonstrates the TEASER++ (Truncated least squares Estimation
 * And SEmidefinite Relaxation) algorithm for robust global point cloud registration.
 *
 * TEASER++ is a globally optimal, certifiably robust registration method that:
 * - Works with >95% outliers in correspondences
 * - Provides optimality guarantees
 * - Does not require initial guess (global registration)
 *
 * Key concepts covered:
 * - Global registration vs local registration
 * - Outlier-robust registration
 * - Using TEASER++ for loop closure and relocalization
 *
 * Reference: Yang et al., "TEASER: Fast and Certifiable Point Cloud Registration",
 *            IEEE T-RO 2020
 *
 * Note: This demo has two modes:
 * - WITH_TEASER defined: Uses actual TEASER++ library
 * - WITHOUT: Demonstrates concepts using PCL's feature-based matching
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#ifdef WITH_TEASER
#include <teaser/registration.h>
#include <teaser/certification.h>
#endif

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FPFHT = pcl::FPFHSignature33;
using FPFHCloudT = pcl::PointCloud<FPFHT>;

/**
 * Generate a synthetic point cloud
 */
CloudT::Ptr generateSyntheticCloud(int num_points = 5000) {
    CloudT::Ptr cloud(new CloudT);
    cloud->reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> noise(-0.02f, 0.02f);

    // Create a room-like structure
    // Floor
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        pt.x = dist(gen);
        pt.y = dist(gen);
        pt.z = noise(gen);
        cloud->push_back(pt);
    }

    // Walls
    for (int i = 0; i < num_points / 4; ++i) {
        PointT pt;
        float side = (float)(i % 4);
        if (side < 1) {
            pt.x = 5.0f + noise(gen);
            pt.y = dist(gen);
            pt.z = std::abs(dist(gen));
        } else if (side < 2) {
            pt.x = -5.0f + noise(gen);
            pt.y = dist(gen);
            pt.z = std::abs(dist(gen));
        } else if (side < 3) {
            pt.x = dist(gen);
            pt.y = 5.0f + noise(gen);
            pt.z = std::abs(dist(gen));
        } else {
            pt.x = dist(gen);
            pt.y = -5.0f + noise(gen);
            pt.z = std::abs(dist(gen));
        }
        cloud->push_back(pt);
    }

    // Some objects
    for (int i = 0; i < num_points / 2; ++i) {
        PointT pt;
        float obj = (float)(i % 3);
        if (obj < 1) {
            // Sphere-ish object
            float theta = (float)i / 100.0f;
            float phi = (float)i / 50.0f;
            pt.x = 2.0f + 0.5f * std::sin(theta) * std::cos(phi);
            pt.y = 1.0f + 0.5f * std::sin(theta) * std::sin(phi);
            pt.z = 0.5f + 0.5f * std::cos(theta);
        } else if (obj < 2) {
            // Box
            pt.x = -2.0f + noise(gen) * 5 + 0.5f;
            pt.y = -2.0f + noise(gen) * 5 + 0.5f;
            pt.z = noise(gen) * 5 + 0.5f;
        } else {
            // Cylinder
            float angle = (float)i * 0.1f;
            pt.x = 0.0f + 0.3f * std::cos(angle);
            pt.y = -3.0f + 0.3f * std::sin(angle);
            pt.z = (float)(i % 100) / 100.0f;
        }
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Compute normals for a point cloud
 */
NormalCloudT::Ptr computeNormals(const CloudT::Ptr& cloud, float radius = 0.5f) {
    pcl::NormalEstimation<PointT, NormalT> ne;
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
FPFHCloudT::Ptr computeFPFH(const CloudT::Ptr& cloud,
                             const NormalCloudT::Ptr& normals,
                             float radius = 1.0f) {
    pcl::FPFHEstimation<PointT, NormalT, FPFHT> fpfh;
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

#ifdef WITH_TEASER
/**
 * Run TEASER++ registration
 */
Eigen::Matrix4f runTEASER(const CloudT::Ptr& source,
                           const CloudT::Ptr& target,
                           const std::vector<std::pair<int, int>>& correspondences,
                           double noise_bound) {

    // Convert to TEASER format
    teaser::PointCloud src_cloud, tgt_cloud;

    for (const auto& corr : correspondences) {
        const auto& sp = source->points[corr.first];
        const auto& tp = target->points[corr.second];
        src_cloud.push_back({sp.x, sp.y, sp.z});
        tgt_cloud.push_back({tp.x, tp.y, tp.z});
    }

    // TEASER++ parameters
    teaser::RobustRegistrationSolver::Params params;
    params.noise_bound = noise_bound;
    params.cbar2 = 1.0;
    params.estimate_scaling = false;
    params.rotation_max_iterations = 100;
    params.rotation_gnc_factor = 1.4;
    params.rotation_estimation_algorithm =
        teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = 0.005;

    teaser::RobustRegistrationSolver solver(params);

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(src_cloud, tgt_cloud);
    auto end = std::chrono::high_resolution_clock::now();

    auto solution = solver.getSolution();

    std::cout << "  TEASER++ solution valid: " << (solution.valid ? "YES" : "NO") << std::endl;
    std::cout << "  Time: " << std::chrono::duration<double, std::milli>(end - start).count()
              << " ms" << std::endl;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = solution.rotation.cast<float>();
    transform.block<3, 1>(0, 3) = solution.translation.cast<float>();

    return transform;
}

#else
/**
 * RANSAC-based global registration (fallback when TEASER++ not available)
 */
Eigen::Matrix4f runRANSACRegistration(const CloudT::Ptr& source,
                                       const CloudT::Ptr& target,
                                       const FPFHCloudT::Ptr& source_features,
                                       const FPFHCloudT::Ptr& target_features) {

    std::cout << "\n  Note: TEASER++ not available, using RANSAC-based registration" << std::endl;

    // Find correspondences based on FPFH features
    pcl::registration::CorrespondenceEstimation<FPFHT, FPFHT> est;
    est.setInputSource(source_features);
    est.setInputTarget(target_features);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    est.determineReciprocalCorrespondences(*correspondences);

    std::cout << "  Initial correspondences: " << correspondences->size() << std::endl;

    // RANSAC-based correspondence rejection
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> ransac;
    ransac.setInputSource(source);
    ransac.setInputTarget(target);
    ransac.setInlierThreshold(0.5);
    ransac.setMaximumIterations(10000);

    pcl::CorrespondencesPtr inliers(new pcl::Correspondences);
    ransac.getRemainingCorrespondences(*correspondences, *inliers);

    std::cout << "  Inlier correspondences: " << inliers->size() << std::endl;

    // Estimate transformation from inliers
    pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
    Eigen::Matrix4f transform;
    svd.estimateRigidTransformation(*source, *target, *inliers, transform);

    return transform;
}
#endif

/**
 * Demonstrate the concept of global registration
 */
void demonstrateGlobalRegistration(const CloudT::Ptr& source,
                                    const CloudT::Ptr& target,
                                    const Eigen::Matrix4f& ground_truth) {

    std::cout << "\n=== Global Registration Demo ===" << std::endl;
    std::cout << "Goal: Find transformation WITHOUT initial guess" << std::endl;

    // Compute features
    std::cout << "\n  Computing normals..." << std::endl;
    auto source_normals = computeNormals(source);
    auto target_normals = computeNormals(target);

    std::cout << "  Computing FPFH features..." << std::endl;
    auto source_fpfh = computeFPFH(source, source_normals);
    auto target_fpfh = computeFPFH(target, target_normals);

    auto start = std::chrono::high_resolution_clock::now();

#ifdef WITH_TEASER
    // Find correspondences for TEASER++
    pcl::registration::CorrespondenceEstimation<FPFHT, FPFHT> est;
    est.setInputSource(source_fpfh);
    est.setInputTarget(target_fpfh);

    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
    est.determineReciprocalCorrespondences(*correspondences);

    std::vector<std::pair<int, int>> corr_pairs;
    for (const auto& c : *correspondences) {
        corr_pairs.push_back({c.index_query, c.index_match});
    }

    std::cout << "  Feature correspondences: " << corr_pairs.size() << std::endl;

    Eigen::Matrix4f global_transform = runTEASER(source, target, corr_pairs, 0.05);
#else
    Eigen::Matrix4f global_transform = runRANSACRegistration(
        source, target, source_fpfh, target_fpfh);
#endif

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    float trans_err, rot_err;
    computeTransformError(global_transform, ground_truth, trans_err, rot_err);

    std::cout << "\n  Global Registration Results:" << std::endl;
    std::cout << "    Translation error: " << std::fixed << std::setprecision(4)
              << trans_err << " m" << std::endl;
    std::cout << "    Rotation error: " << std::setprecision(4)
              << rot_err << " deg" << std::endl;
    std::cout << "    Total time: " << std::setprecision(1) << time_ms << " ms" << std::endl;

    // Refine with ICP
    std::cout << "\n  Refining with ICP..." << std::endl;

    CloudT::Ptr aligned_global(new CloudT);
    pcl::transformPointCloud(*source, *aligned_global, global_transform);

    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(aligned_global);
    icp.setInputTarget(target);
    icp.setMaximumIterations(30);
    icp.setMaxCorrespondenceDistance(0.5);

    CloudT::Ptr final_aligned(new CloudT);
    icp.align(*final_aligned);

    Eigen::Matrix4f refined_transform = icp.getFinalTransformation() * global_transform;
    computeTransformError(refined_transform, ground_truth, trans_err, rot_err);

    std::cout << "  After ICP refinement:" << std::endl;
    std::cout << "    Translation error: " << std::fixed << std::setprecision(4)
              << trans_err << " m" << std::endl;
    std::cout << "    Rotation error: " << std::setprecision(4)
              << rot_err << " deg" << std::endl;
}

/**
 * Demonstrate outlier robustness
 */
void demonstrateOutlierRobustness() {
    std::cout << "\n=== Outlier Robustness Concept ===" << std::endl;
    std::cout << "TEASER++ can handle high outlier ratios in correspondences:\n" << std::endl;

    std::cout << "  Outlier Ratio    |  Standard Methods  |  TEASER++" << std::endl;
    std::cout << "  -----------------+--------------------+-----------" << std::endl;
    std::cout << "     10%           |     Works well     |  Works" << std::endl;
    std::cout << "     30%           |     Degraded       |  Works" << std::endl;
    std::cout << "     50%           |     Often fails    |  Works" << std::endl;
    std::cout << "     70%           |     Usually fails  |  Works" << std::endl;
    std::cout << "     90%           |     Fails          |  Works" << std::endl;
    std::cout << "     95%           |     Fails          |  Works" << std::endl;
    std::cout << std::endl;

    std::cout << "Key insight: TEASER++ uses truncated least squares (TLS) and" << std::endl;
    std::cout << "graduated non-convexity (GNC) to handle outliers robustly." << std::endl;
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [source.pcd target.pcd]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  No arguments    - Use synthetic point clouds" << std::endl;
    std::cout << "  source target   - Use provided PCD files" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This demo works best when TEASER++ is installed." << std::endl;
    std::cout << "Without TEASER++, it falls back to RANSAC-based registration." << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== TEASER++ Global Registration Demo ===" << std::endl;
    std::cout << "Robust, certifiably optimal point cloud registration\n" << std::endl;

#ifdef WITH_TEASER
    std::cout << "TEASER++ library: AVAILABLE" << std::endl;
#else
    std::cout << "TEASER++ library: NOT AVAILABLE (using RANSAC fallback)" << std::endl;
    std::cout << "To enable TEASER++:" << std::endl;
    std::cout << "  1. Install TEASER++: https://github.com/MIT-SPARK/TEASER-plusplus" << std::endl;
    std::cout << "  2. Rebuild with: cmake -DUSE_TEASER=ON .." << std::endl;
#endif

    CloudT::Ptr source(new CloudT);
    CloudT::Ptr target(new CloudT);
    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();

    if (argc == 1) {
        // Generate synthetic data
        std::cout << "\nGenerating synthetic point clouds..." << std::endl;

        target = generateSyntheticCloud(5000);

        // Large transformation (typical for loop closure / relocalization)
        float tx = 2.0f, ty = 1.5f, tz = 0.3f;
        float roll = 0.1f, pitch = 0.05f, yaw = 0.8f;  // ~45 degrees yaw

        ground_truth = createTransform(tx, ty, tz, roll, pitch, yaw);

        CloudT::Ptr transformed(new CloudT);
        pcl::transformPointCloud(*target, *transformed, ground_truth);
        source = transformed;

        std::cout << "  Source points: " << source->size() << std::endl;
        std::cout << "  Target points: " << target->size() << std::endl;
        std::cout << "  Ground truth translation: [" << tx << ", " << ty << ", " << tz << "]" << std::endl;
        std::cout << "  Ground truth yaw: " << (yaw * 180.0f / M_PI) << " degrees" << std::endl;

    } else if (argc == 3) {
        std::cout << "\nLoading point clouds from files..." << std::endl;

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

    } else {
        printUsage(argv[0]);
        return -1;
    }

    // Downsample for faster processing
    if (source->size() > 10000 || target->size() > 10000) {
        std::cout << "\nDownsampling for feature extraction..." << std::endl;

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
    // Demonstrate global registration
    // ====================================

    demonstrateGlobalRegistration(source, target, ground_truth);

    // ====================================
    // Explain outlier robustness
    // ====================================

    demonstrateOutlierRobustness();

    // ====================================
    // Summary
    // ====================================

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "When to use TEASER++:" << std::endl;
    std::cout << "  1. Loop closure detection and verification" << std::endl;
    std::cout << "  2. Relocalization after tracking loss" << std::endl;
    std::cout << "  3. Initial alignment before ICP" << std::endl;
    std::cout << "  4. When correspondences contain many outliers" << std::endl;
    std::cout << std::endl;
    std::cout << "Typical pipeline:" << std::endl;
    std::cout << "  1. Extract features (FPFH, SHOT, etc.)" << std::endl;
    std::cout << "  2. Find putative correspondences" << std::endl;
    std::cout << "  3. Run TEASER++ for robust transformation" << std::endl;
    std::cout << "  4. Refine with ICP/GICP" << std::endl;
    std::cout << std::endl;

    return 0;
}
