/**
 * @file icp_point_to_plane.cpp
 * @brief Point-to-Plane ICP registration using PCL with normal estimation
 *
 * This example demonstrates:
 * - Normal estimation using pcl::NormalEstimationOMP
 * - Point-to-plane ICP using pcl::IterativeClosestPointWithNormals
 * - Comparison between point-to-point and point-to-plane ICP
 * - Stepping both methods side by side, one iteration per keystroke
 *
 * Point-to-plane ICP minimizes the distance to the tangent plane at each point.
 * That lets correspondences slide along the surface rather than being pinned
 * point to point, so it converges in markedly fewer iterations wherever the
 * surface is locally smooth - which includes the bunny, even though the model is
 * nowhere near planar overall.
 *
 * The Stanford bunny (data/bun_zipper_res3.ply) is the target, and the source is
 * the same model displaced by a known transform, so both methods are scored
 * against the exact answer. The demo takes no arguments and always opens the
 * viewer; an X display is required.
 *
 * Usage: ./icp_point_to_plane
 */

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include "demo_common.hpp"
#include "demo_viz.hpp"

using PointT = demo::PointT;
using PointNT = pcl::PointNormal;
using PointCloudT = demo::CloudT;
using PointCloudNT = pcl::PointCloud<PointNT>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;

/**
 * @brief Compute normals for a point cloud
 *
 * K-nearest-neighbour search is used rather than a radius: the bunny is a
 * decimated mesh with ~1.9k vertices, and a radius small enough to capture its
 * curvature leaves isolated points with too few neighbours, which yields NaN
 * normals that IterativeClosestPointWithNormals cannot use.
 */
NormalCloud::Ptr computeNormals(const PointCloudT::Ptr& cloud, int k_neighbors = 20)
{
    NormalCloud::Ptr normals(new NormalCloud);

    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setKSearch(k_neighbors);

    // Use KdTree for efficient nearest neighbor search
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);

    ne.compute(*normals);

    return normals;
}

/**
 * @brief Create PointNormal cloud by concatenating points and normals
 */
PointCloudNT::Ptr createPointNormalCloud(const PointCloudT::Ptr& cloud,
                                           const NormalCloud::Ptr& normals)
{
    PointCloudNT::Ptr cloud_with_normals(new PointCloudNT);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    return cloud_with_normals;
}

/**
 * @brief Count points whose normal could not be estimated
 */
size_t countInvalidNormals(const NormalCloud& normals)
{
    size_t invalid = 0;
    for (const auto& n : normals)
    {
        if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) ||
            !std::isfinite(n.normal_z))
        {
            ++invalid;
        }
    }
    return invalid;
}

/**
 * @brief Run point-to-point ICP for comparison
 */
double runPointToPointICP(const PointCloudT::Ptr& source,
                           const PointCloudT::Ptr& target,
                           double max_correspondence_distance,
                           Eigen::Matrix4f& result_transform,
                           double& execution_time_ms)
{
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(max_correspondence_distance);

    PointCloudT::Ptr aligned(new PointCloudT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    execution_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result_transform = icp.getFinalTransformation();

    return icp.getFitnessScore();
}

/**
 * @brief Run point-to-plane ICP
 */
double runPointToPlaneICP(const PointCloudNT::Ptr& source_with_normals,
                           const PointCloudNT::Ptr& target_with_normals,
                           double max_correspondence_distance,
                           Eigen::Matrix4f& result_transform,
                           double& execution_time_ms)
{
    pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
    icp.setInputSource(source_with_normals);
    icp.setInputTarget(target_with_normals);
    icp.setMaximumIterations(30);  // Typically needs fewer iterations
    icp.setTransformationEpsilon(1e-10);
    icp.setEuclideanFitnessEpsilon(1e-8);
    icp.setMaxCorrespondenceDistance(max_correspondence_distance);

    PointCloudNT::Ptr aligned(new PointCloudNT);

    auto start = std::chrono::high_resolution_clock::now();
    icp.align(*aligned);
    auto end = std::chrono::high_resolution_clock::now();

    execution_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    result_transform = icp.getFinalTransformation();

    return icp.getFitnessScore();
}

int main()
{
    std::cout << "=== Point-to-Plane ICP Example ===\n\n";

    // The demo takes no arguments: it always registers the Stanford bunny
    // against a displaced copy of itself.
    const std::string model = demo::findDataFile(demo::kBunnyFile);

    if (model.empty())
    {
        std::cerr << "Error: Could not find data/" << demo::kBunnyFile << ".\n";
        std::cerr << "Run from the project root or from the build/ directory.\n";
        return -1;
    }

    PointCloudT::Ptr target_cloud = demo::loadCloud(model);
    if (!target_cloud)
    {
        std::cerr << "Error: Could not load " << model << "\n";
        return -1;
    }
    std::cout << "Loaded " << target_cloud->size() << " points from " << model << "\n";

    // Centre the model so the injected rotation turns it about its own axis
    target_cloud = demo::centerCloud(*target_cloud);

    // ============================================
    // Model scale
    // ============================================
    const double scale = demo::bboxDiagonal(*target_cloud);
    const double voxel_size = scale * 0.005;
    const double max_correspondence_distance = scale * 0.1;

    std::cout << "\nModel scale (bbox diagonal): " << std::fixed << std::setprecision(4)
              << scale << " m\n";

    // Build the source cloud by displacing the target with a known transform,
    // so both methods can be scored against the exact answer
    // Large enough that the misalignment is visible in the viewer: at a few
    // percent of the model the clouds overlap too closely to watch converge.
    const float shift = static_cast<float>(scale * 0.08);
    const float angle = 15.0f * M_PI / 180.0f;

    const Eigen::Matrix4f ground_truth =
        demo::makeTransform(shift, shift * 0.5f, shift * 0.2f, 0.01f, 0.02f, angle);

    PointCloudT::Ptr source_cloud(new PointCloudT);
    pcl::transformPointCloud(*target_cloud, *source_cloud, ground_truth);

    std::cout << "Source cloud: " << source_cloud->size() << " points\n";
    std::cout << "Applied transformation: t=(" << std::setprecision(4) << shift << ", "
              << shift * 0.5f << ", " << shift * 0.2f << ") m, rz=15deg\n";

    // Downsample
    std::cout << "\nDownsampling clouds with voxel size " << std::setprecision(4)
              << voxel_size << " m...\n";

    PointCloudT::Ptr source_filtered = demo::voxelDownsample(*source_cloud, voxel_size);
    PointCloudT::Ptr target_filtered = demo::voxelDownsample(*target_cloud, voxel_size);

    std::cout << "Source after filtering: " << source_filtered->size() << " points\n";
    std::cout << "Target after filtering: " << target_filtered->size() << " points\n";

    // ============================================
    // Compute normals
    // ============================================
    std::cout << "\n--- Computing Normals ---\n";

    auto start_normals = std::chrono::high_resolution_clock::now();

    NormalCloud::Ptr source_normals = computeNormals(source_filtered);
    NormalCloud::Ptr target_normals = computeNormals(target_filtered);

    auto end_normals = std::chrono::high_resolution_clock::now();
    auto normal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_normals - start_normals);

    std::cout << "Normal estimation time: " << normal_time.count() << " ms (k=20 neighbours)\n";
    std::cout << "Source normals: " << source_normals->size()
              << " (" << countInvalidNormals(*source_normals) << " invalid)\n";
    std::cout << "Target normals: " << target_normals->size()
              << " (" << countInvalidNormals(*target_normals) << " invalid)\n";

    // Create PointNormal clouds
    PointCloudNT::Ptr source_with_normals = createPointNormalCloud(source_filtered, source_normals);
    PointCloudNT::Ptr target_with_normals = createPointNormalCloud(target_filtered, target_normals);

    // ============================================
    // Run Point-to-Point ICP (for comparison)
    // ============================================
    std::cout << "\n--- Point-to-Point ICP ---\n";

    Eigen::Matrix4f transform_p2p;
    double time_p2p;
    double fitness_p2p = runPointToPointICP(source_filtered, target_filtered,
                                             max_correspondence_distance,
                                             transform_p2p, time_p2p);

    std::cout << "Fitness score: " << fitness_p2p << "\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << time_p2p << " ms\n";
    demo::printTransformation(transform_p2p, "Point-to-Point Transformation");

    // ============================================
    // Run Point-to-Plane ICP
    // ============================================
    std::cout << "\n--- Point-to-Plane ICP ---\n";

    Eigen::Matrix4f transform_p2plane;
    double time_p2plane;
    double fitness_p2plane = runPointToPlaneICP(source_with_normals, target_with_normals,
                                                 max_correspondence_distance,
                                                 transform_p2plane, time_p2plane);

    std::cout << "Fitness score: " << fitness_p2plane << "\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << time_p2plane << " ms\n";
    demo::printTransformation(transform_p2plane, "Point-to-Plane Transformation");

    // ============================================
    // Comparison
    // ============================================
    std::cout << "\n=== Comparison ===\n";
    std::cout << std::left << std::setw(25) << "Metric"
              << std::setw(20) << "Point-to-Point"
              << std::setw(20) << "Point-to-Plane" << "\n";
    std::cout << std::string(65, '-') << "\n";
    std::cout << std::left << std::setw(25) << "Fitness Score (MSE)"
              << std::setw(20) << std::scientific << std::setprecision(4) << fitness_p2p
              << std::setw(20) << fitness_p2plane << "\n";
    std::cout << std::left << std::setw(25) << "Execution Time (ms)"
              << std::setw(20) << std::fixed << std::setprecision(2) << time_p2p
              << std::setw(20) << time_p2plane << "\n";
    std::cout << std::left << std::setw(25) << "Normal Estimation"
              << std::setw(20) << "Not needed"
              << std::setw(20) << (std::to_string(normal_time.count()) + " ms") << "\n";

    const Eigen::Matrix4f gt = ground_truth.inverse();
    const demo::PoseError err_p2p = demo::poseError(transform_p2p, gt);
    const demo::PoseError err_p2plane = demo::poseError(transform_p2plane, gt);

    std::cout << std::left << std::setw(25) << "Rotation Error (deg)"
              << std::setw(20) << std::fixed << std::setprecision(4) << err_p2p.rotation_deg
              << std::setw(20) << err_p2plane.rotation_deg << "\n";
    std::cout << std::left << std::setw(25) << "Translation Error (m)"
              << std::setw(20) << std::setprecision(6) << err_p2p.translation_m
              << std::setw(20) << err_p2plane.translation_m << "\n";

    // On clean data both methods reach the same answer once they are given
    // enough iterations, so the final errors above are identical and say
    // nothing. The number that separates the two methods is how many iterations
    // each one needed to get there. Counted on throwaway copies, because the
    // steppers rewrite the clouds they are handed.
    PointCloudT::Ptr scratch_p2p(new PointCloudT(*source_filtered));
    PointCloudT::Ptr scratch_display(new PointCloudT(*source_filtered));
    PointCloudNT::Ptr scratch_p2plane(new PointCloudNT(*source_with_normals));

    const int iters_p2p = demo::countIterationsToConverge(
        demo::makePointToPointStep(target_filtered, scratch_p2p,
                                   max_correspondence_distance, scale));
    const int iters_p2plane = demo::countIterationsToConverge(
        demo::makePointToPlaneStep(target_with_normals, scratch_p2plane,
                                   scratch_display, max_correspondence_distance, scale));

    std::cout << std::left << std::setw(25) << "Iterations to converge"
              << std::setw(20) << iters_p2p
              << std::setw(20) << iters_p2plane << "\n";

    std::cout << "\nConclusion:\n";
    std::cout << "  Point-to-plane converged in " << iters_p2plane
              << " iterations against " << iters_p2p << " for point-to-point,\n";
    std::cout << "  on a model that is nowhere near planar. The advantage does not come\n";
    std::cout << "  from the surface being flat overall, but from each normal's\n";
    std::cout << "  neighbourhood being locally smooth: minimising distance to the tangent\n";
    std::cout << "  plane lets correspondences slide along the surface instead of being\n";
    std::cout << "  pinned point to point, which is what costs point-to-point its\n";
    std::cout << "  iterations. The price is estimating normals up front.\n";

    // ============================================
    // Step both methods interactively, side by side
    // ============================================
    if (!demo::hasDisplay())
    {
        demo::reportMissingDisplay();
        return 0;
    }

    // Both tracks start from the same displaced source and advance together, so
    // the difference in convergence rate is visible directly
    PointCloudT::Ptr step_p2p(new PointCloudT(*source_filtered));
    PointCloudT::Ptr step_p2plane(new PointCloudT(*source_filtered));
    PointCloudNT::Ptr step_p2plane_n(new PointCloudNT(*source_with_normals));

    std::vector<demo::Track> tracks;
    tracks.push_back({"point-to-point", "p2point", 255, 255, 0, step_p2p,
                      demo::makePointToPointStep(target_filtered, step_p2p,
                                                 max_correspondence_distance, scale),
                      0, -1, 0.0, Eigen::Matrix4f::Identity()});
    tracks.push_back({"point-to-plane", "p2plane", 255, 0, 255, step_p2plane,
                      demo::makePointToPlaneStep(target_with_normals, step_p2plane_n,
                                                 step_p2plane, max_correspondence_distance,
                                                 scale),
                      0, -1, 0.0, Eigen::Matrix4f::Identity()});

    demo::runStepViewer("Point-to-Point vs Point-to-Plane ICP - press any key to step",
                        target_filtered, tracks, scale);

    std::cout << "\n=== Done ===\n";
    return 0;
}
