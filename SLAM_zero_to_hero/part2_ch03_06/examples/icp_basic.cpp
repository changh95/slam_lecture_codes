/**
 * @file icp_basic.cpp
 * @brief Basic Point-to-Point ICP (Iterative Closest Point) registration using PCL
 *
 * This example demonstrates:
 * - Loading source and target point clouds
 * - Setting up pcl::IterativeClosestPoint
 * - Configuring ICP parameters (iterations, epsilon, correspondence distance)
 * - Running ICP alignment and scoring the result against a known transform
 *
 * By default the Stanford bunny (data/bun_zipper_res3.ply) is used as the
 * target, and the source is the same model displaced by a known transform, so
 * the estimate can be compared against the exact answer.
 *
 * Usage: ./icp_basic                       # Stanford bunny (default)
 *        ./icp_basic source.pcd target.pcd # your own pair
 *        ./icp_basic --generate            # synthetic half-sphere
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

#include "demo_common.hpp"

using PointT = demo::PointT;
using PointCloudT = demo::CloudT;

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

int main(int argc, char** argv)
{
    std::cout << "=== Basic Point-to-Point ICP Example ===\n\n";

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
        std::cout << "  --generate, -g          Generate a synthetic half-sphere instead\n";
        return 0;
    }

    // The known transform applied to build the source cloud, when the demo
    // creates the pair itself. Filled in once the model scale is known.
    Eigen::Matrix4f ground_truth = Eigen::Matrix4f::Identity();
    bool have_ground_truth = false;

    if (generate_mode)
    {
        std::cout << "Generating sample point clouds...\n";

        target_cloud = generateSampleCloud(5000);
        std::cout << "Target cloud: " << target_cloud->size() << " points\n";
    }
    else if (files.size() >= 2)
    {
        std::cout << "Loading point clouds from files...\n";

        source_cloud = demo::loadCloud(files[0]);
        if (!source_cloud)
        {
            std::cerr << "Error: Could not load source cloud: " << files[0] << "\n";
            return -1;
        }
        std::cout << "Source cloud: " << source_cloud->size() << " points from " << files[0] << "\n";

        target_cloud = demo::loadCloud(files[1]);
        if (!target_cloud)
        {
            std::cerr << "Error: Could not load target cloud: " << files[1] << "\n";
            return -1;
        }
        std::cout << "Target cloud: " << target_cloud->size() << " points from " << files[1] << "\n";
    }
    else
    {
        // Default: the Stanford bunny, or any single cloud given on the command line
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

        // Centre the model so the injected rotation turns it about its own axis
        target_cloud = demo::centerCloud(*target_cloud);
    }

    // ============================================
    // Model scale
    // ============================================
    // The bunny is ~0.25 m across while a LiDAR scan spans ~100 m, so every
    // distance below is a fraction of the bounding-box diagonal rather than a
    // hard-coded value.
    const double scale = demo::bboxDiagonal(*target_cloud);
    const double voxel_size = scale * 0.005;
    const double max_correspondence_distance = scale * 0.1;

    std::cout << "\nModel scale (bbox diagonal): " << std::fixed << std::setprecision(4)
              << scale << " m\n";

    // Build the source cloud by displacing the target, when the demo owns the pair
    if (source_cloud->empty())
    {
        const float shift = static_cast<float>(scale * 0.04);
        const float angle = 8.0f * M_PI / 180.0f;

        ground_truth = demo::makeTransform(shift, shift * 0.5f, shift * 0.2f, 0.0f, 0.0f, angle);
        have_ground_truth = true;

        pcl::transformPointCloud(*target_cloud, *source_cloud, ground_truth);

        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: t=(" << std::setprecision(4) << shift << ", "
                  << shift * 0.5f << ", " << shift * 0.2f << ") m, rz=8deg\n";
    }

    // Optional: Downsample clouds for faster processing
    std::cout << "\nDownsampling clouds with voxel size " << std::setprecision(4)
              << voxel_size << " m...\n";

    PointCloudT::Ptr source_filtered = demo::voxelDownsample(*source_cloud, voxel_size);
    PointCloudT::Ptr target_filtered = demo::voxelDownsample(*target_cloud, voxel_size);

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
    icp.setTransformationEpsilon(1e-10);    // Transformation epsilon (convergence criteria)
    icp.setEuclideanFitnessEpsilon(1e-8);   // Euclidean fitness epsilon (MSE convergence)
    icp.setMaxCorrespondenceDistance(max_correspondence_distance);

    std::cout << "ICP Parameters:\n";
    std::cout << "  Max iterations: 50\n";
    std::cout << "  Transformation epsilon: 1e-10\n";
    std::cout << "  Euclidean fitness epsilon: 1e-8\n";
    std::cout << "  Max correspondence distance: " << std::setprecision(4)
              << max_correspondence_distance << " m (10% of model scale)\n";

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
        demo::printTransformation(transformation, "Estimated Transformation (Source -> Target)");

        if (have_ground_truth)
        {
            // ICP aligns source onto target, so it recovers the inverse of the
            // transform that produced the source.
            demo::printTransformation(ground_truth.inverse(),
                                      "Ground Truth Transformation (Source -> Target)");
            demo::printPoseError(transformation, ground_truth.inverse(), scale);
        }

        // Interpret the fitness score, relative to the model scale
        const double fitness = icp.getFitnessScore();
        const double relative = std::sqrt(fitness) / scale;
        std::cout << "\nAlignment quality: ";
        if (relative < 0.001)
        {
            std::cout << "Excellent (RMS < 0.1% of model size)\n";
        }
        else if (relative < 0.01)
        {
            std::cout << "Good (RMS < 1% of model size)\n";
        }
        else if (relative < 0.05)
        {
            std::cout << "Acceptable (RMS < 5% of model size)\n";
        }
        else
        {
            std::cout << "Poor (RMS >= 5% of model size) - "
                         "consider a better initial guess or parameters\n";
        }

        // Save aligned cloud
        const std::string output_file = "aligned_cloud.pcd";
        pcl::io::savePCDFileBinary(output_file, *aligned_cloud);
        std::cout << "\nAligned cloud saved to: " << output_file << "\n";
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
