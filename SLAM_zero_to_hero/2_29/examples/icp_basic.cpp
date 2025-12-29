/**
 * @file icp_basic.cpp
 * @brief Basic Point-to-Point ICP (Iterative Closest Point) registration using PCL
 *
 * This example demonstrates:
 * - Loading source and target point clouds
 * - Setting up pcl::IterativeClosestPoint
 * - Configuring ICP parameters (iterations, epsilon, correspondence distance)
 * - Running ICP alignment and getting results
 *
 * Usage: ./icp_basic source.pcd target.pcd [--generate]
 *        --generate: Generate sample point clouds for testing
 */

#include <iostream>
#include <string>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Dense>

// Type aliases for convenience
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

/**
 * @brief Generate a sample point cloud (a simple bunny-like shape)
 */
PointCloudT::Ptr generateSampleCloud(int num_points = 5000)
{
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->points.reserve(num_points);

    // Generate points on a half-sphere with some random variation
    for (int i = 0; i < num_points; ++i)
    {
        double theta = static_cast<double>(rand()) / RAND_MAX * M_PI;        // 0 to PI
        double phi = static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;    // 0 to 2*PI
        double r = 1.0 + 0.1 * (static_cast<double>(rand()) / RAND_MAX - 0.5);

        PointT p;
        p.x = static_cast<float>(r * sin(theta) * cos(phi));
        p.y = static_cast<float>(r * sin(theta) * sin(phi));
        p.z = static_cast<float>(r * cos(theta));
        cloud->points.push_back(p);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * @brief Apply a transformation to create a transformed copy of the cloud
 */
PointCloudT::Ptr transformCloud(const PointCloudT::Ptr& cloud,
                                  float tx, float ty, float tz,
                                  float rx, float ry, float rz)
{
    // Create transformation matrix
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    // Rotation (in radians)
    transform.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));

    // Translation
    transform.translation() << tx, ty, tz;

    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*cloud, *transformed, transform);

    return transformed;
}

/**
 * @brief Print transformation matrix in a readable format
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

int main(int argc, char** argv)
{
    std::cout << "=== Basic Point-to-Point ICP Example ===\n\n";

    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);

    // Check for generate flag or load from files
    bool generate_mode = false;
    std::string source_file, target_file;

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
        std::cout << "Generating sample point clouds...\n";

        // Generate target cloud
        target_cloud = generateSampleCloud(5000);
        std::cout << "Target cloud: " << target_cloud->size() << " points\n";

        // Create source by transforming target
        // Known transformation: translate (0.1, 0.05, 0.02) and rotate 5 degrees around Z
        float angle = 5.0f * M_PI / 180.0f;  // 5 degrees in radians
        source_cloud = transformCloud(target_cloud, 0.1f, 0.05f, 0.02f, 0.0f, 0.0f, angle);
        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: tx=0.1, ty=0.05, tz=0.02, rz=5deg\n";
    }
    else if (argc >= 3)
    {
        source_file = argv[1];
        target_file = argv[2];

        std::cout << "Loading point clouds from files...\n";

        // Load source cloud
        if (pcl::io::loadPCDFile<PointT>(source_file, *source_cloud) == -1)
        {
            std::cerr << "Error: Could not load source cloud: " << source_file << "\n";
            return -1;
        }
        std::cout << "Source cloud: " << source_cloud->size() << " points from " << source_file << "\n";

        // Load target cloud
        if (pcl::io::loadPCDFile<PointT>(target_file, *target_cloud) == -1)
        {
            std::cerr << "Error: Could not load target cloud: " << target_file << "\n";
            return -1;
        }
        std::cout << "Target cloud: " << target_cloud->size() << " points from " << target_file << "\n";
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " source.pcd target.pcd\n";
        std::cout << "       " << argv[0] << " --generate\n";
        std::cout << "\nOptions:\n";
        std::cout << "  source.pcd, target.pcd  Input point cloud files\n";
        std::cout << "  --generate, -g          Generate sample point clouds for testing\n";
        return 0;
    }

    // Optional: Downsample clouds for faster processing
    std::cout << "\nDownsampling clouds with voxel size 0.01...\n";
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.01f, 0.01f, 0.01f);

    PointCloudT::Ptr source_filtered(new PointCloudT);
    PointCloudT::Ptr target_filtered(new PointCloudT);

    voxel.setInputCloud(source_cloud);
    voxel.filter(*source_filtered);

    voxel.setInputCloud(target_cloud);
    voxel.filter(*target_filtered);

    std::cout << "Source after filtering: " << source_filtered->size() << " points\n";
    std::cout << "Target after filtering: " << target_filtered->size() << " points\n";

    // ============================================
    // Setup ICP
    // ============================================
    std::cout << "\n--- Setting up Point-to-Point ICP ---\n";

    pcl::IterativeClosestPoint<PointT, PointT> icp;

    // Set source and target clouds
    icp.setInputSource(source_filtered);
    icp.setInputTarget(target_filtered);

    // Set ICP parameters
    icp.setMaximumIterations(50);           // Maximum number of iterations
    icp.setTransformationEpsilon(1e-8);     // Transformation epsilon (convergence criteria)
    icp.setEuclideanFitnessEpsilon(1e-6);   // Euclidean fitness epsilon (MSE convergence)
    icp.setMaxCorrespondenceDistance(0.5);  // Maximum correspondence distance

    std::cout << "ICP Parameters:\n";
    std::cout << "  Max iterations: 50\n";
    std::cout << "  Transformation epsilon: 1e-8\n";
    std::cout << "  Euclidean fitness epsilon: 1e-6\n";
    std::cout << "  Max correspondence distance: 0.5\n";

    // ============================================
    // Run ICP alignment
    // ============================================
    std::cout << "\nRunning ICP alignment...\n";

    PointCloudT::Ptr aligned_cloud(new PointCloudT);

    auto start_time = std::chrono::high_resolution_clock::now();
    icp.align(*aligned_cloud);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // ============================================
    // Check results
    // ============================================
    std::cout << "\n--- ICP Results ---\n";
    std::cout << "Converged: " << (icp.hasConverged() ? "Yes" : "No") << "\n";
    std::cout << "Fitness score (MSE): " << icp.getFitnessScore() << "\n";
    std::cout << "Execution time: " << duration.count() << " ms\n";

    if (icp.hasConverged())
    {
        Eigen::Matrix4f transformation = icp.getFinalTransformation();
        printTransformation(transformation, "Estimated Transformation (Source -> Target)");

        // Compute inverse to get Target -> Source transformation
        Eigen::Matrix4f inverse_transformation = transformation.inverse();
        printTransformation(inverse_transformation, "Inverse Transformation (Target -> Source)");

        // Interpret the fitness score
        double fitness = icp.getFitnessScore();
        std::cout << "\nAlignment quality: ";
        if (fitness < 0.001)
        {
            std::cout << "Excellent (MSE < 0.001)\n";
        }
        else if (fitness < 0.01)
        {
            std::cout << "Good (MSE < 0.01)\n";
        }
        else if (fitness < 0.1)
        {
            std::cout << "Acceptable (MSE < 0.1)\n";
        }
        else
        {
            std::cout << "Poor (MSE >= 0.1) - consider better initial guess or parameters\n";
        }

        // Save aligned cloud
        if (!generate_mode)
        {
            std::string output_file = "aligned_cloud.pcd";
            pcl::io::savePCDFileBinary(output_file, *aligned_cloud);
            std::cout << "\nAligned cloud saved to: " << output_file << "\n";
        }
    }
    else
    {
        std::cerr << "\nWarning: ICP did not converge!\n";
        std::cerr << "Suggestions:\n";
        std::cerr << "  - Provide a better initial guess\n";
        std::cerr << "  - Increase MaxCorrespondenceDistance\n";
        std::cerr << "  - Increase MaximumIterations\n";
        std::cerr << "  - Check if clouds have sufficient overlap\n";
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
