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
 * Usage: ./icp_point_to_plane source.pcd target.pcd [--generate]
 */

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Dense>

// Type aliases
using PointT = pcl::PointXYZ;
using PointNT = pcl::PointNormal;
using PointCloudT = pcl::PointCloud<PointT>;
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
 * @brief Transform a point cloud
 */
PointCloudT::Ptr transformCloud(const PointCloudT::Ptr& cloud,
                                  float tx, float ty, float tz,
                                  float rx, float ry, float rz)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
    transform.translation() << tx, ty, tz;

    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*cloud, *transformed, transform);
    return transformed;
}

/**
 * @brief Compute normals for a point cloud
 */
NormalCloud::Ptr computeNormals(const PointCloudT::Ptr& cloud, double radius = 0.1)
{
    NormalCloud::Ptr normals(new NormalCloud);

    pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(radius);

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
 * @brief Print transformation matrix
 */
void printTransformation(const Eigen::Matrix4f& T, const std::string& name)
{
    std::cout << "\n" << name << ":\n";
    std::cout << "  Rotation matrix:\n";
    for (int i = 0; i < 3; ++i)
    {
        std::cout << "    [";
        for (int j = 0; j < 3; ++j)
        {
            std::cout << std::fixed << std::setprecision(6) << std::setw(10) << T(i, j);
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "  Translation: ["
              << T(0, 3) << ", " << T(1, 3) << ", " << T(2, 3) << "]\n";
}

/**
 * @brief Run point-to-point ICP for comparison
 */
double runPointToPointICP(const PointCloudT::Ptr& source,
                           const PointCloudT::Ptr& target,
                           Eigen::Matrix4f& result_transform,
                           double& execution_time_ms)
{
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.5);

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
                           Eigen::Matrix4f& result_transform,
                           double& execution_time_ms)
{
    pcl::IterativeClosestPointWithNormals<PointNT, PointNT> icp;
    icp.setInputSource(source_with_normals);
    icp.setInputTarget(target_with_normals);
    icp.setMaximumIterations(30);  // Typically needs fewer iterations
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.5);

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

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--generate" || arg == "-g")
        {
            generate_mode = true;
        }
    }

    if (generate_mode)
    {
        std::cout << "Generating planar point clouds (simulating indoor environment)...\n";

        target_cloud = generatePlanarCloud(10000);
        std::cout << "Target cloud: " << target_cloud->size() << " points\n";

        // Apply transformation: translate and rotate
        float angle = 3.0f * M_PI / 180.0f;  // 3 degrees
        source_cloud = transformCloud(target_cloud, 0.15f, 0.08f, 0.03f, 0.01f, 0.02f, angle);
        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: tx=0.15, ty=0.08, tz=0.03, rz=3deg\n";
    }
    else if (argc >= 3)
    {
        std::string source_file = argv[1];
        std::string target_file = argv[2];

        std::cout << "Loading point clouds...\n";

        if (pcl::io::loadPCDFile<PointT>(source_file, *source_cloud) == -1)
        {
            std::cerr << "Error: Could not load source cloud: " << source_file << "\n";
            return -1;
        }
        std::cout << "Source: " << source_cloud->size() << " points from " << source_file << "\n";

        if (pcl::io::loadPCDFile<PointT>(target_file, *target_cloud) == -1)
        {
            std::cerr << "Error: Could not load target cloud: " << target_file << "\n";
            return -1;
        }
        std::cout << "Target: " << target_cloud->size() << " points from " << target_file << "\n";
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " source.pcd target.pcd\n";
        std::cout << "       " << argv[0] << " --generate\n";
        return 0;
    }

    // Downsample
    std::cout << "\nDownsampling clouds...\n";
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.02f, 0.02f, 0.02f);

    PointCloudT::Ptr source_filtered(new PointCloudT);
    PointCloudT::Ptr target_filtered(new PointCloudT);

    voxel.setInputCloud(source_cloud);
    voxel.filter(*source_filtered);

    voxel.setInputCloud(target_cloud);
    voxel.filter(*target_filtered);

    std::cout << "Source after filtering: " << source_filtered->size() << " points\n";
    std::cout << "Target after filtering: " << target_filtered->size() << " points\n";

    // ============================================
    // Compute normals
    // ============================================
    std::cout << "\n--- Computing Normals ---\n";

    auto start_normals = std::chrono::high_resolution_clock::now();

    NormalCloud::Ptr source_normals = computeNormals(source_filtered, 0.1);
    NormalCloud::Ptr target_normals = computeNormals(target_filtered, 0.1);

    auto end_normals = std::chrono::high_resolution_clock::now();
    auto normal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_normals - start_normals);

    std::cout << "Normal estimation time: " << normal_time.count() << " ms\n";
    std::cout << "Source normals: " << source_normals->size() << "\n";
    std::cout << "Target normals: " << target_normals->size() << "\n";

    // Create PointNormal clouds
    PointCloudNT::Ptr source_with_normals = createPointNormalCloud(source_filtered, source_normals);
    PointCloudNT::Ptr target_with_normals = createPointNormalCloud(target_filtered, target_normals);

    // ============================================
    // Run Point-to-Point ICP (for comparison)
    // ============================================
    std::cout << "\n--- Point-to-Point ICP ---\n";

    Eigen::Matrix4f transform_p2p;
    double time_p2p;
    double fitness_p2p = runPointToPointICP(source_filtered, target_filtered, transform_p2p, time_p2p);

    std::cout << "Fitness score: " << fitness_p2p << "\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << time_p2p << " ms\n";
    printTransformation(transform_p2p, "Point-to-Point Transformation");

    // ============================================
    // Run Point-to-Plane ICP
    // ============================================
    std::cout << "\n--- Point-to-Plane ICP ---\n";

    Eigen::Matrix4f transform_p2plane;
    double time_p2plane;
    double fitness_p2plane = runPointToPlaneICP(source_with_normals, target_with_normals,
                                                  transform_p2plane, time_p2plane);

    std::cout << "Fitness score: " << fitness_p2plane << "\n";
    std::cout << "Execution time: " << std::fixed << std::setprecision(2) << time_p2plane << " ms\n";
    printTransformation(transform_p2plane, "Point-to-Plane Transformation");

    // ============================================
    // Comparison
    // ============================================
    std::cout << "\n=== Comparison ===\n";
    std::cout << std::left << std::setw(25) << "Metric"
              << std::setw(20) << "Point-to-Point"
              << std::setw(20) << "Point-to-Plane" << "\n";
    std::cout << std::string(65, '-') << "\n";
    std::cout << std::left << std::setw(25) << "Fitness Score (MSE)"
              << std::setw(20) << std::scientific << fitness_p2p
              << std::setw(20) << fitness_p2plane << "\n";
    std::cout << std::left << std::setw(25) << "Execution Time (ms)"
              << std::setw(20) << std::fixed << std::setprecision(2) << time_p2p
              << std::setw(20) << time_p2plane << "\n";
    std::cout << std::left << std::setw(25) << "Normal Estimation"
              << std::setw(20) << "Not needed"
              << std::setw(20) << (std::to_string(normal_time.count()) + " ms") << "\n";

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
