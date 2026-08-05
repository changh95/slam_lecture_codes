/**
 * @file icp_point_to_plane.cpp
 * @brief Point-to-Plane ICP registration using PCL with normal estimation
 *
 * This example demonstrates:
 * - Normal estimation using pcl::NormalEstimationOMP
 * - Point-to-plane ICP using pcl::IterativeClosestPointWithNormals
 * - Comparison between point-to-point and point-to-plane ICP
 *
 * Point-to-plane ICP minimizes the distance to the tangent plane at each point,
 * which typically converges faster than point-to-point ICP, especially for
 * planar surfaces.
 *
 * By default the Stanford bunny (data/bun_zipper_res3.ply) is used as the
 * target, and the source is the same model displaced by a known transform, so
 * both methods can be scored against the exact answer.
 *
 * Usage: ./icp_point_to_plane                       # Stanford bunny (default)
 *        ./icp_point_to_plane source.pcd target.pcd # your own pair
 *        ./icp_point_to_plane --generate            # synthetic indoor box
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

using PointT = demo::PointT;
using PointNT = pcl::PointNormal;
using PointCloudT = demo::CloudT;
using PointCloudNT = pcl::PointCloud<PointNT>;
using NormalCloud = pcl::PointCloud<pcl::Normal>;

/**
 * @brief Generate a sample point cloud with planar regions
 */
PointCloudT::Ptr generatePlanarCloud(int num_points = 5000)
{
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->points.reserve(num_points);

    // Generate points on multiple planes (simulating indoor environment)
    int points_per_plane = num_points / 5;

    // Floor (z = 0)
    for (int i = 0; i < points_per_plane; ++i)
    {
        PointT p;
        p.x = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.y = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.z = 0.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        cloud->points.push_back(p);
    }

    // Left wall (x = -2)
    for (int i = 0; i < points_per_plane; ++i)
    {
        PointT p;
        p.x = -2.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        p.y = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
        cloud->points.push_back(p);
    }

    // Right wall (x = 2)
    for (int i = 0; i < points_per_plane; ++i)
    {
        PointT p;
        p.x = 2.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        p.y = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
        cloud->points.push_back(p);
    }

    // Back wall (y = -2)
    for (int i = 0; i < points_per_plane; ++i)
    {
        PointT p;
        p.x = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.y = -2.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
        cloud->points.push_back(p);
    }

    // Front wall (y = 2)
    for (int i = 0; i < points_per_plane; ++i)
    {
        PointT p;
        p.x = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f;
        p.y = 2.0f + 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        p.z = static_cast<float>(rand()) / RAND_MAX * 2.0f;
        cloud->points.push_back(p);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

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

int main(int argc, char** argv)
{
    std::cout << "=== Point-to-Plane ICP Example ===\n\n";

    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);

    bool generate_mode = false;
    bool help_mode = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--generate" || arg == "-g")
        {
            generate_mode = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            help_mode = true;
        }
        else if (arg[0] != '-')
        {
            files.push_back(arg);
        }
    }

    if (help_mode)
    {
        std::cout << "Usage: " << argv[0] << "                        (Stanford bunny)\n";
        std::cout << "       " << argv[0] << " source.pcd target.pcd\n";
        std::cout << "       " << argv[0] << " --generate\n";
        std::cout << "\nOptions:\n";
        std::cout << "  (no arguments)          Use data/" << demo::kBunnyFile
                  << " with a known transform\n";
        std::cout << "  source, target          Input clouds (.ply, .pcd, or KITTI .bin)\n";
        std::cout << "  --generate, -g          Generate a synthetic indoor box instead\n";
        return 0;
    }

    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();
    bool have_ground_truth = false;

    if (generate_mode)
    {
        std::cout << "Generating planar point clouds (simulating indoor environment)...\n";

        target_cloud = generatePlanarCloud(10000);
        std::cout << "Target cloud: " << target_cloud->size() << " points\n";
    }
    else if (files.size() >= 2)
    {
        std::cout << "Loading point clouds...\n";

        source_cloud = demo::loadCloud(files[0]);
        if (!source_cloud)
        {
            std::cerr << "Error: Could not load source cloud: " << files[0] << "\n";
            return -1;
        }
        std::cout << "Source: " << source_cloud->size() << " points from " << files[0] << "\n";

        target_cloud = demo::loadCloud(files[1]);
        if (!target_cloud)
        {
            std::cerr << "Error: Could not load target cloud: " << files[1] << "\n";
            return -1;
        }
        std::cout << "Target: " << target_cloud->size() << " points from " << files[1] << "\n";
    }
    else
    {
        const std::string model = files.empty() ? demo::findDataFile(demo::kBunnyFile) : files[0];

        if (model.empty())
        {
            std::cerr << "Error: Could not find data/" << demo::kBunnyFile << ".\n";
            std::cerr << "Run from the project root or build/ directory, "
                         "or pass a cloud file explicitly.\n";
            return -1;
        }

        target_cloud = demo::loadCloud(model);
        if (!target_cloud)
        {
            std::cerr << "Error: Could not load " << model << "\n";
            return -1;
        }
        std::cout << "Loaded " << target_cloud->size() << " points from " << model << "\n";

        target_cloud = demo::centerCloud(*target_cloud);
    }

    // ============================================
    // Model scale
    // ============================================
    const double scale = demo::bboxDiagonal(*target_cloud);
    const double voxel_size = scale * 0.005;
    const double max_correspondence_distance = scale * 0.1;

    std::cout << "\nModel scale (bbox diagonal): " << std::fixed << std::setprecision(4)
              << scale << " m\n";

    if (source_cloud->empty())
    {
        const float shift = static_cast<float>(scale * 0.04);
        const float angle = 5.0f * M_PI / 180.0f;

        ground_truth = demo::makeTransform(shift, shift * 0.5f, shift * 0.2f,
                                           0.01f, 0.02f, angle);
        have_ground_truth = true;

        pcl::transformPointCloud(*target_cloud, *source_cloud, ground_truth);

        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: t=(" << std::setprecision(4) << shift << ", "
                  << shift * 0.5f << ", " << shift * 0.2f << ") m, rz=5deg\n";
    }

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

    if (have_ground_truth)
    {
        const Eigen::Matrix4f gt = ground_truth.inverse();
        const demo::PoseError err_p2p = demo::poseError(transform_p2p, gt);
        const demo::PoseError err_p2plane = demo::poseError(transform_p2plane, gt);

        std::cout << std::left << std::setw(25) << "Rotation Error (deg)"
                  << std::setw(20) << std::fixed << std::setprecision(4) << err_p2p.rotation_deg
                  << std::setw(20) << err_p2plane.rotation_deg << "\n";
        std::cout << std::left << std::setw(25) << "Translation Error (m)"
                  << std::setw(20) << std::setprecision(6) << err_p2p.translation_m
                  << std::setw(20) << err_p2plane.translation_m << "\n";
    }

    std::cout << "\nConclusion:\n";
    if (fitness_p2plane < fitness_p2p)
    {
        std::cout << "  Point-to-plane ICP achieved better alignment (lower fitness score).\n";
    }
    else
    {
        std::cout << "  Point-to-point ICP achieved comparable or better alignment.\n";
    }

    std::cout << "  Point-to-plane ICP typically converges faster for planar surfaces,\n";
    std::cout << "  but requires additional time for normal estimation.\n";

    std::cout << "\n=== Done ===\n";
    return 0;
}
