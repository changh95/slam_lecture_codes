/**
 * @file icp_basic.cpp
 * @brief Basic Point-to-Point ICP (Iterative Closest Point) registration using PCL
 *
 * This example demonstrates:
 * - Loading source and target point clouds
 * - Setting up pcl::IterativeClosestPoint
 * - Configuring ICP parameters (iterations, epsilon, correspondence distance)
 * - Running ICP alignment and scoring the result against a known transform
 * - Stepping through the alignment interactively, one iteration per keystroke
 *
 * The Stanford bunny (data/bun_zipper_res3.ply) is the target, and the source is
 * the same model displaced by a known transform, so the estimate is compared
 * against the exact answer. The demo takes no arguments and always opens the
 * viewer; an X display is required.
 *
 * Usage: ./icp_basic
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
#include "demo_viz.hpp"

using PointT = demo::PointT;
using PointCloudT = demo::CloudT;

int main()
{
    std::cout << "=== Basic Point-to-Point ICP Example ===\n\n";

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
    // The bunny is ~0.25 m across while a LiDAR scan spans ~100 m, so every
    // distance below is a fraction of the bounding-box diagonal rather than a
    // hard-coded value.
    const double scale = demo::bboxDiagonal(*target_cloud);
    const double voxel_size = scale * 0.005;
    const double max_correspondence_distance = scale * 0.1;

    std::cout << "\nModel scale (bbox diagonal): " << std::fixed << std::setprecision(4)
              << scale << " m\n";

    // Build the source cloud by displacing the target with a known transform,
    // so the estimate can be scored against the exact answer
    const float shift = static_cast<float>(scale * 0.04);
    const float angle = 8.0f * M_PI / 180.0f;

    const Eigen::Matrix4f ground_truth =
        demo::makeTransform(shift, shift * 0.5f, shift * 0.2f, 0.0f, 0.0f, angle);

    PointCloudT::Ptr source_cloud(new PointCloudT);
    pcl::transformPointCloud(*target_cloud, *source_cloud, ground_truth);

    std::cout << "Source cloud: " << source_cloud->size() << " points\n";
    std::cout << "Applied transformation: t=(" << std::setprecision(4) << shift << ", "
              << shift * 0.5f << ", " << shift * 0.2f << ") m, rz=8deg\n";

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

        // ICP aligns source onto target, so it recovers the inverse of the
        // transform that produced the source.
        demo::printTransformation(ground_truth.inverse(),
                                  "Ground Truth Transformation (Source -> Target)");
        demo::printPoseError(transformation, ground_truth.inverse(), scale);

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

        // How many of the 50 iterations were actually needed. Run on a throwaway
        // copy, since the stepper rewrites the cloud it is given.
        PointCloudT::Ptr scratch(new PointCloudT(*source_filtered));
        const int needed = demo::countIterationsToConverge(
            demo::makePointToPointStep(target_filtered, scratch,
                                       max_correspondence_distance, scale));
        std::cout << "Iterations to converge: " << needed << "\n";

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

    // ============================================
    // Step through the alignment interactively
    // ============================================
    if (!demo::hasDisplay())
    {
        demo::reportMissingDisplay();
        return 0;
    }

    // The viewer starts from the displaced source and converges onto the target
    PointCloudT::Ptr stepping(new PointCloudT(*source_filtered));

    std::vector<demo::Track> tracks;
    tracks.push_back({"point-to-point", "p2point", 255, 255, 0, stepping,
                      demo::makePointToPointStep(target_filtered, stepping,
                                                 max_correspondence_distance, scale),
                      0, -1, 0.0, Eigen::Matrix4f::Identity()});

    demo::runStepViewer("Point-to-Point ICP - press any key to step",
                        target_filtered, tracks, scale);

    std::cout << "\n=== Done ===\n";
    return 0;
}
