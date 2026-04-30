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
 * Usage: ./icp_visualization source.pcd target.pcd [--generate] [--step]
 *        --step: Show ICP alignment step by step
 */

#include <iostream>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>

// Type aliases
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

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
 * @brief Transform a point cloud
 */
PointCloudT::Ptr transformCloud(const PointCloudT::Ptr& cloud,
                                  const Eigen::Matrix4f& transform)
{
    PointCloudT::Ptr transformed(new PointCloudT);
    pcl::transformPointCloud(*cloud, *transformed, transform);
    return transformed;
}

/**
 * @brief Create a transformation matrix
 */
Eigen::Matrix4f createTransform(float tx, float ty, float tz,
                                  float rx, float ry, float rz)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
    transform.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
    transform.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
    transform.translation() << tx, ty, tz;
    return transform.matrix();
}

/**
 * @brief Visualize ICP alignment
 */
void visualizeICP(const PointCloudT::Ptr& source,
                   const PointCloudT::Ptr& target,
                   const PointCloudT::Ptr& aligned,
                   const Eigen::Matrix4f& transformation,
                   double fitness_score)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("ICP Visualization"));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);

    // Add coordinate system
    viewer->addCoordinateSystem(0.5, "coordinate");

    // Source cloud (green) - original position
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        source_color(source, 0, 255, 0);
    viewer->addPointCloud<PointT>(source, source_color, "source");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");

    // Target cloud (blue)
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        target_color(target, 0, 100, 255);
    viewer->addPointCloud<PointT>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    // Aligned cloud (red)
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        aligned_color(aligned, 255, 50, 50);
    viewer->addPointCloud<PointT>(aligned, aligned_color, "aligned");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned");

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
             << std::fixed << std::setprecision(3)
             << transformation(0, 3) << ", "
             << transformation(1, 3) << ", "
             << transformation(2, 3) << "]";
    viewer->addText(ss_trans.str(), 10, 100, 12, 0.8, 0.8, 0.8, "text_trans");

    // Camera settings
    viewer->setCameraPosition(0, -3, 2, 0, 0, 0, 0, 0, 1);

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
                              int max_iterations = 30)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("ICP Step-by-Step"));

    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->addCoordinateSystem(0.5, "coordinate");

    // Target cloud (blue) - static
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        target_color(target, 0, 100, 255);
    viewer->addPointCloud<PointT>(target, target_color, "target");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");

    // Current aligned cloud - will be updated
    PointCloudT::Ptr current(new PointCloudT(*source));
    pcl::visualization::PointCloudColorHandlerCustom<PointT>
        current_color(current, 255, 255, 0);
    viewer->addPointCloud<PointT>(current, current_color, "current");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "current");

    viewer->addText("Blue: Target | Yellow: Current alignment", 10, 60, 14, 1.0, 1.0, 1.0, "text_legend");
    viewer->addText("Press 'n' for next iteration, 'q' to quit", 10, 40, 14, 0.8, 0.8, 0.8, "text_help");

    viewer->setCameraPosition(0, -3, 2, 0, 0, 0, 0, 0, 1);

    Eigen::Matrix4f cumulative_transform = Eigen::Matrix4f::Identity();
    int iteration = 0;

    // Keyboard callback
    bool next_step = false;
    viewer->registerKeyboardCallback(
        [&next_step](const pcl::visualization::KeyboardEvent& event) {
            if (event.keyDown() && event.getKeySym() == "n")
            {
                next_step = true;
            }
        });

    std::cout << "\n=== Step-by-Step ICP ===\n";
    std::cout << "Press 'n' for next iteration\n";
    std::cout << "Press 'q' to quit\n\n";

    while (!viewer->wasStopped() && iteration < max_iterations)
    {
        viewer->spinOnce(100);

        if (next_step)
        {
            next_step = false;
            iteration++;

            // Run single ICP iteration
            pcl::IterativeClosestPoint<PointT, PointT> icp;
            icp.setInputSource(current);
            icp.setInputTarget(target);
            icp.setMaximumIterations(1);  // Single iteration
            icp.setTransformationEpsilon(1e-10);
            icp.setMaxCorrespondenceDistance(0.5);

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
    }

    if (generate_mode)
    {
        std::cout << "Generating sample point clouds (torus shape)...\n";

        target_cloud = generateTorusCloud(8000);
        std::cout << "Target cloud: " << target_cloud->size() << " points\n";

        // Create transformation
        float angle = 15.0f * M_PI / 180.0f;  // 15 degrees
        Eigen::Matrix4f transform = createTransform(0.2f, 0.1f, 0.05f, 0.0f, 0.0f, angle);

        source_cloud = transformCloud(target_cloud, transform);
        std::cout << "Source cloud: " << source_cloud->size() << " points\n";
        std::cout << "Applied transformation: tx=0.2, ty=0.1, tz=0.05, rz=15deg\n";
    }
    else if (argc >= 3)
    {
        std::string source_file, target_file;
        int file_idx = 0;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            if (arg[0] != '-')
            {
                if (file_idx == 0)
                {
                    source_file = arg;
                    file_idx++;
                }
                else if (file_idx == 1)
                {
                    target_file = arg;
                    file_idx++;
                }
            }
        }

        if (file_idx < 2)
        {
            std::cerr << "Error: Need source and target PCD files\n";
            return -1;
        }

        std::cout << "Loading point clouds...\n";

        if (pcl::io::loadPCDFile<PointT>(source_file, *source_cloud) == -1)
        {
            std::cerr << "Error: Could not load source cloud: " << source_file << "\n";
            return -1;
        }
        std::cout << "Source: " << source_cloud->size() << " points\n";

        if (pcl::io::loadPCDFile<PointT>(target_file, *target_cloud) == -1)
        {
            std::cerr << "Error: Could not load target cloud: " << target_file << "\n";
            return -1;
        }
        std::cout << "Target: " << target_cloud->size() << " points\n";
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " source.pcd target.pcd [--step]\n";
        std::cout << "       " << argv[0] << " --generate [--step]\n";
        std::cout << "\nOptions:\n";
        std::cout << "  --generate, -g  Generate sample point clouds\n";
        std::cout << "  --step, -s      Show step-by-step ICP alignment\n";
        return 0;
    }

    // Downsample for faster visualization
    std::cout << "\nDownsampling clouds...\n";
    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(0.02f, 0.02f, 0.02f);

    PointCloudT::Ptr source_filtered(new PointCloudT);
    PointCloudT::Ptr target_filtered(new PointCloudT);

    voxel.setInputCloud(source_cloud);
    voxel.filter(*source_filtered);

    voxel.setInputCloud(target_cloud);
    voxel.filter(*target_filtered);

    std::cout << "Source filtered: " << source_filtered->size() << " points\n";
    std::cout << "Target filtered: " << target_filtered->size() << " points\n";

    if (step_mode)
    {
        // Step-by-step visualization
        visualizeICPStepByStep(source_filtered, target_filtered, 50);
    }
    else
    {
        // Run full ICP first
        std::cout << "\nRunning ICP alignment...\n";

        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(source_filtered);
        icp.setInputTarget(target_filtered);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setMaxCorrespondenceDistance(0.5);

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
                          transformation, fitness);
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
