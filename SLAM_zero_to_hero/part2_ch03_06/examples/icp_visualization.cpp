/**
 * @file icp_visualization.cpp
 * @brief Visualize ICP alignment process using PCL Visualizer
 *
 * This example demonstrates:
 * - Setting up PCL Visualizer for point cloud visualization
 * - Displaying source, target, and aligned clouds with different colors
 * - Interactive visualization with keyboard controls
 * - Step-by-step ICP visualization (optional)
 *
 * By default the Stanford bunny (data/bun_zipper_res3.ply) is used as the
 * target, and the source is the same model displaced by a known transform.
 * Camera distance, axis size and ICP distances all scale with the model, so the
 * same demo works for the bunny and for room-scale clouds.
 *
 * Usage: ./icp_visualization                       # Stanford bunny (default)
 *        ./icp_visualization --step                # one ICP iteration per keystroke
 *        ./icp_visualization source.pcd target.pcd # your own pair
 *        ./icp_visualization --generate            # synthetic torus
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

#include "demo_common.hpp"

using PointT = demo::PointT;
using PointCloudT = demo::CloudT;

/**
 * @brief Generate a sample point cloud (toroidal shape)
 */
PointCloudT::Ptr generateTorusCloud(int num_points = 5000)
{
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->points.reserve(num_points);

    float R = 1.0f;  // Major radius
    float r = 0.3f;  // Minor radius

    for (int i = 0; i < num_points; ++i)
    {
        float u = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;
        float v = static_cast<float>(rand()) / RAND_MAX * 2.0f * M_PI;

        // Add small noise
        float noise = 0.02f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

        PointT p;
        p.x = (R + (r + noise) * cos(v)) * cos(u);
        p.y = (R + (r + noise) * cos(v)) * sin(u);
        p.z = (r + noise) * sin(v);
        cloud->points.push_back(p);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * @brief Point the camera at the origin from a distance proportional to the model
 */
void setupCamera(pcl::visualization::PCLVisualizer& viewer, double scale)
{
    viewer.setCameraPosition(0.0, -2.0 * scale, 1.2 * scale,  // camera position
                             0.0, 0.0, 0.0,                    // look at the origin
                             0.0, 0.0, 1.0);                   // up is +Z
    viewer.setCameraClipDistances(0.01 * scale, 100.0 * scale);
}

/**
 * @brief Visualize ICP alignment
 */
void visualizeICP(const PointCloudT::Ptr& source,
                   const PointCloudT::Ptr& target,
                   const PointCloudT::Ptr& aligned,
                   const Eigen::Matrix4f& transformation,
                   double fitness_score,
                   double scale)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("ICP Visualization"));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);

    // Add coordinate system, sized relative to the model
    viewer->addCoordinateSystem(0.3 * scale, "coordinate");

    // Source cloud (green) - original position
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        source_color(source, 0, 255, 0);
    viewer->addPointCloud<PointT>(source, source_color, "source");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source");

    // Target cloud (blue)
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        target_color(target, 0, 100, 255);
    viewer->addPointCloud<PointT>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target");

    // Aligned cloud (red)
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        aligned_color(aligned, 255, 50, 50);
    viewer->addPointCloud<PointT>(aligned, aligned_color, "aligned");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "aligned");

    // Add text annotations
    viewer->addText("Green: Source (original)", 10, 80, 14, 0.0, 1.0, 0.0, "text_source");
    viewer->addText("Blue: Target", 10, 60, 14, 0.4, 0.4, 1.0, "text_target");
    viewer->addText("Red: Aligned (source after ICP)", 10, 40, 14, 1.0, 0.2, 0.2, "text_aligned");

    std::stringstream ss;
    ss << "Fitness Score (MSE): " << std::scientific << std::setprecision(4) << fitness_score;
    viewer->addText(ss.str(), 10, 20, 14, 1.0, 1.0, 1.0, "text_fitness");

    // Add transformation info
    std::stringstream ss_trans;
    ss_trans << "Translation: ["
             << std::fixed << std::setprecision(4)
             << transformation(0, 3) << ", "
             << transformation(1, 3) << ", "
             << transformation(2, 3) << "]";
    viewer->addText(ss_trans.str(), 10, 100, 12, 0.8, 0.8, 0.8, "text_trans");

    setupCamera(*viewer, scale);

    std::cout << "\n=== Visualization Controls ===\n";
    std::cout << "  Mouse: Rotate view\n";
    std::cout << "  Scroll: Zoom\n";
    std::cout << "  Shift+Mouse: Pan\n";
    std::cout << "  q: Quit\n";
    std::cout << "  r: Reset camera\n";
    std::cout << "  s: Save screenshot\n\n";

    // Spin viewer
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

/**
 * @brief Visualize ICP alignment step by step
 */
void visualizeICPStepByStep(const PointCloudT::Ptr& source,
                              const PointCloudT::Ptr& target,
                              double max_correspondence_distance,
                              double scale,
                              int max_iterations = 30)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("ICP Step-by-Step"));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->addCoordinateSystem(0.3 * scale, "coordinate");

    // Target cloud (blue) - static
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        target_color(target, 0, 100, 255);
    viewer->addPointCloud<PointT>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target");

    // Current aligned cloud - will be updated
    PointCloudT::Ptr current(new PointCloudT(*source));
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        current_color(current, 255, 255, 0);
    viewer->addPointCloud<PointT>(current, current_color, "current");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "current");

    viewer->addText("Blue: Target | Yellow: Current alignment", 10, 60, 14, 1.0, 1.0, 1.0, "text_legend");
    viewer->addText("Press any key for the next iteration, 'q' to quit", 10, 40, 14, 0.8, 0.8, 0.8, "text_help");

    setupCamera(*viewer, scale);

    Eigen::Matrix4f cumulative_transform = Eigen::Matrix4f::Identity();
    int iteration = 0;

    // Any keystroke advances one iteration. 'q' is excluded because that is the
    // viewer's own quit key, and the keys PCLVisualizer binds itself (r to reset
    // the camera, s/w for the render mode, g for the grid) still do their usual
    // job in addition to stepping.
    bool next_step = false;
    viewer->registerKeyboardCallback(
        [&next_step](const pcl::visualization::KeyboardEvent& event) {
            if (event.keyDown() && event.getKeySym() != "q")
            {
                next_step = true;
            }
        });

    std::cout << "\n=== Step-by-Step ICP ===\n";
    std::cout << "Press any key in the viewer window for the next iteration\n";
    std::cout << "Press 'q' to quit\n\n";

    bool limit_reported = false;

    // The viewer stays alive after the iteration limit instead of the window
    // disappearing, so the final alignment can still be inspected.
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);

        if (next_step)
        {
            next_step = false;

            if (iteration >= max_iterations)
            {
                if (!limit_reported)
                {
                    std::cout << "Reached the " << max_iterations
                              << " iteration limit - press 'q' to quit\n";
                    limit_reported = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            iteration++;

            // Run single ICP iteration
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setInputSource(current);
            icp.setInputTarget(target);
            icp.setMaximumIterations(1);  // Single iteration
            icp.setTransformationEpsilon(1e-12);
            icp.setMaxCorrespondenceDistance(max_correspondence_distance);

            PointCloudT::Ptr aligned(new PointCloudT);
            icp.align(*aligned);

            if (icp.hasConverged())
            {
                Eigen::Matrix4f step_transform = icp.getFinalTransformation();
                cumulative_transform = step_transform * cumulative_transform;

                *current = *aligned;

                // Update visualization
                viewer->updatePointCloud<PointT>(current, current_color, "current");

                // Update text
                std::stringstream ss;
                ss << "Iteration: " << iteration << " | Fitness: "
                   << std::scientific << std::setprecision(4) << icp.getFitnessScore();
                viewer->removeShape("text_iter");
                viewer->addText(ss.str(), 10, 20, 14, 1.0, 1.0, 0.0, "text_iter");

                std::cout << "Iteration " << iteration
                          << " - Fitness: " << icp.getFitnessScore() << "\n";
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "\nFinal cumulative transformation:\n" << cumulative_transform << "\n";
}

int main(int argc, char** argv)
{
    std::cout << "=== ICP Visualization Example ===\n\n";

    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);

    bool generate_mode = false;
    bool step_mode = false;
    bool help_mode = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--generate" || arg == "-g")
        {
            generate_mode = true;
        }
        else if (arg == "--step" || arg == "-s")
        {
            step_mode = true;
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
        std::cout << "Usage: " << argv[0] << " [--step]              (Stanford bunny)\n";
        std::cout << "       " << argv[0] << " source.pcd target.pcd [--step]\n";
        std::cout << "       " << argv[0] << " --generate [--step]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  (no arguments)          Use data/" << demo::kBunnyFile
                  << " with a known transform\n";
        std::cout << "  source, target          Input clouds (.ply, .pcd, or KITTI .bin)\n";
        std::cout << "  --generate, -g          Generate a synthetic torus instead\n";
        std::cout << "  --step, -s              Step through ICP one iteration per keystroke\n";
        return 0;
    }

    if (generate_mode)
    {
        std::cout << "Generating sample point clouds (torus shape)...\n";

        target_cloud = generateTorusCloud(8000);
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
        std::cout << "Source: " << source_cloud->size() << " points\n";

        target_cloud = demo::loadCloud(files[1]);
        if (!target_cloud)
        {
            std::cerr << "Error: Could not load target cloud: " << files[1] << "\n";
            return -1;
        }
        std::cout << "Target: " << target_cloud->size() << " points\n";
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
        // A larger offset than the other demos, so the convergence is visible
        const float shift = static_cast<float>(scale * 0.08);
        const float angle = 15.0f * M_PI / 180.0f;

        const Eigen::Matrix4f transform =
            demo::makeTransform(shift, shift * 0.5f, shift * 0.25f, 0.0f, 0.0f, angle);
        pcl::transformPointCloud(*target_cloud, *source_cloud, transform);

        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: t=(" << std::setprecision(4) << shift << ", "
                  << shift * 0.5f << ", " << shift * 0.25f << ") m, rz=15deg\n";
    }

    // Downsample for faster visualization
    std::cout << "\nDownsampling clouds with voxel size " << std::setprecision(4)
              << voxel_size << " m...\n";

    PointCloudT::Ptr source_filtered = demo::voxelDownsample(*source_cloud, voxel_size);
    PointCloudT::Ptr target_filtered = demo::voxelDownsample(*target_cloud, voxel_size);

    std::cout << "Source filtered: " << source_filtered->size() << " points\n";
    std::cout << "Target filtered: " << target_filtered->size() << " points\n";

    if (step_mode)
    {
        // Step-by-step visualization
        visualizeICPStepByStep(source_filtered, target_filtered,
                               max_correspondence_distance, scale, 50);
    }
    else
    {
        // Run full ICP first
        std::cout << "\nRunning ICP alignment...\n";

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(source_filtered);
        icp.setInputTarget(target_filtered);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-10);
        icp.setEuclideanFitnessEpsilon(1e-8);
        icp.setMaxCorrespondenceDistance(max_correspondence_distance);

        PointCloudT::Ptr aligned_cloud(new PointCloudT);
        icp.align(*aligned_cloud);

        if (icp.hasConverged())
        {
            Eigen::Matrix4f transformation = icp.getFinalTransformation();
            double fitness = icp.getFitnessScore();

            std::cout << "ICP converged!\n";
            std::cout << "Fitness score: " << fitness << "\n";
            std::cout << "Transformation:\n" << transformation << "\n";

            // Visualize
            visualizeICP(source_filtered, target_filtered, aligned_cloud,
                          transformation, fitness, scale);
        }
        else
        {
            std::cerr << "ICP did not converge!\n";
            return -1;
        }
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
