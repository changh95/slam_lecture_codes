/**
 * Point Cloud Visualization with PCL
 *
 * This example demonstrates:
 * 1. Basic PCLVisualizer setup
 * 2. Adding point clouds with different colors
 * 3. Visualizing normals
 * 4. Adding geometric primitives (spheres, lines, planes)
 * 5. Keyboard and mouse interaction
 *
 * Note: Visualization requires a display. In Docker, use VNC or X11 forwarding.
 */

#include <iostream>
#include <string>
#include <random>
#include <thread>
#include <chrono>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a colored point cloud (cube)
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateColoredCloud(int num_points) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointXYZRGB point;
        point.x = dist(rng);
        point.y = dist(rng);
        point.z = dist(rng);

        // Color based on position (gradient)
        point.r = static_cast<uint8_t>((point.x + 1.0f) * 127.5f);
        point.g = static_cast<uint8_t>((point.y + 1.0f) * 127.5f);
        point.b = static_cast<uint8_t>((point.z + 1.0f) * 127.5f);

        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Generate a sphere point cloud
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSphereCloud(int num_points, float radius) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI);
    uniform_real_distribution<float> phi_dist(0.0f, M_PI);

    for (int i = 0; i < num_points; ++i) {
        float theta = theta_dist(rng);
        float phi = phi_dist(rng);

        pcl::PointXYZ point;
        point.x = radius * sin(phi) * cos(theta);
        point.y = radius * sin(phi) * sin(theta);
        point.z = radius * cos(phi);

        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Generate plane point cloud with normals
 */
pcl::PointCloud<pcl::PointNormal>::Ptr generatePlaneWithNormals(int num_points) {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud(
        new pcl::PointCloud<pcl::PointNormal>);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    normal_distribution<float> noise(0.0f, 0.01f);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointNormal point;
        point.x = dist(rng);
        point.y = dist(rng);
        point.z = noise(rng);

        // Normal pointing up
        point.normal_x = 0.0f;
        point.normal_y = 0.0f;
        point.normal_z = 1.0f;

        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

// =============================================================================
// Visualization Demonstrations
// =============================================================================

void demoBasicVisualization() {
    cout << "=== Basic Visualization ===" << endl;
    cout << "Creating a simple point cloud viewer..." << endl;

    // Generate cloud
    auto cloud = generateColoredCloud(5000);

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Basic Visualization"));

    // Set background color
    viewer->setBackgroundColor(0.1, 0.1, 0.1);

    // Add point cloud
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "colored_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "colored_cloud");

    // Add coordinate system
    viewer->addCoordinateSystem(0.5);

    // Initialize camera
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 5, 0, 0, 0, 0, 1, 0);

    cout << "Displaying for 3 seconds..." << endl;
    cout << "  - Colored point cloud (5000 points)" << endl;
    cout << "  - Coordinate system at origin" << endl;

    // Spin for a few seconds (non-blocking demo)
    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);

        auto now = chrono::steady_clock::now();
        if (chrono::duration<double>(now - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void demoMultipleClouds() {
    cout << "=== Multiple Point Clouds ===" << endl;
    cout << "Displaying multiple clouds with different colors..." << endl;

    // Generate multiple clouds at different positions
    auto cloud1 = generateSphereCloud(1000, 0.5);
    auto cloud2 = generateSphereCloud(1000, 0.5);
    auto cloud3 = generateSphereCloud(1000, 0.5);

    // Translate clouds
    for (auto& p : cloud1->points) { p.x -= 1.5; }
    for (auto& p : cloud2->points) { /* center */ }
    for (auto& p : cloud3->points) { p.x += 1.5; }

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Multiple Clouds"));

    viewer->setBackgroundColor(0, 0, 0);

    // Add clouds with different solid colors
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        red(cloud1, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        green(cloud2, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        blue(cloud3, 0, 0, 255);

    viewer->addPointCloud<pcl::PointXYZ>(cloud1, red, "cloud1");
    viewer->addPointCloud<pcl::PointXYZ>(cloud2, green, "cloud2");
    viewer->addPointCloud<pcl::PointXYZ>(cloud3, blue, "cloud3");

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud1");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud3");

    viewer->addCoordinateSystem(0.5);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 6, 0, 0, 0, 0, 1, 0);

    cout << "  - Red sphere (left)" << endl;
    cout << "  - Green sphere (center)" << endl;
    cout << "  - Blue sphere (right)" << endl;

    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void demoNormalVisualization() {
    cout << "=== Normal Visualization ===" << endl;
    cout << "Displaying point cloud with surface normals..." << endl;

    // Generate plane with normals
    auto cloud = generatePlaneWithNormals(500);

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Normal Visualization"));

    viewer->setBackgroundColor(0.2, 0.2, 0.2);

    // Add point cloud
    viewer->addPointCloud<pcl::PointNormal>(cloud, "plane");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "plane");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.8, 0.0, "plane");

    // Add normals (display every 10th normal, length 0.1)
    viewer->addPointCloudNormals<pcl::PointNormal>(cloud, 10, 0.1f, "normals");

    viewer->addCoordinateSystem(0.3);
    viewer->initCameraParameters();
    viewer->setCameraPosition(2, 2, 2, 0, 0, 0, 0, 0, 1);

    cout << "  - Green plane with normal vectors" << endl;

    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void demoGeometricPrimitives() {
    cout << "=== Geometric Primitives ===" << endl;
    cout << "Adding spheres, lines, and text to viewer..." << endl;

    // Generate base cloud
    auto cloud = generateSphereCloud(1000, 1.0);

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Geometric Primitives"));

    viewer->setBackgroundColor(0.05, 0.05, 0.05);

    // Add point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        white(cloud, 200, 200, 200);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, white, "sphere_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sphere_cloud");

    // Add sphere marker
    pcl::PointXYZ center(0, 0, 0);
    viewer->addSphere(center, 0.1, 1.0, 0.0, 0.0, "center_sphere");

    // Add line
    pcl::PointXYZ p1(-1.5, 0, 0);
    pcl::PointXYZ p2(1.5, 0, 0);
    viewer->addLine(p1, p2, 0.0, 1.0, 1.0, "x_line");

    // Add text at position
    viewer->addText3D("Origin", center, 0.1, 1.0, 1.0, 0.0, "origin_text");

    // Add 2D text
    viewer->addText("PCL Visualization Demo", 10, 10, 20, 1.0, 1.0, 1.0, "title");

    viewer->addCoordinateSystem(0.5);
    viewer->initCameraParameters();
    viewer->setCameraPosition(3, 3, 3, 0, 0, 0, 0, 0, 1);

    cout << "  - White sphere point cloud" << endl;
    cout << "  - Red sphere at origin" << endl;
    cout << "  - Cyan line along X axis" << endl;
    cout << "  - 3D text label" << endl;

    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void demoColorMaps() {
    cout << "=== Color Maps ===" << endl;
    cout << "Coloring point cloud by Z coordinate..." << endl;

    // Generate sphere cloud
    auto cloud = generateSphereCloud(5000, 1.0);

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Color Maps"));

    viewer->setBackgroundColor(0, 0, 0);

    // Create intensity field from Z coordinate
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_intensity(
        new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto& p : cloud->points) {
        pcl::PointXYZI pi;
        pi.x = p.x;
        pi.y = p.y;
        pi.z = p.z;
        pi.intensity = p.z;  // Color by height
        cloud_intensity->points.push_back(pi);
    }
    cloud_intensity->width = cloud_intensity->points.size();
    cloud_intensity->height = 1;

    // Add with intensity color handler
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI>
        intensity_handler(cloud_intensity, "intensity");

    viewer->addPointCloud<pcl::PointXYZI>(cloud_intensity, intensity_handler, "intensity_cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "intensity_cloud");

    viewer->addCoordinateSystem(0.5);
    viewer->initCameraParameters();
    viewer->setCameraPosition(3, 0, 0, 0, 0, 0, 0, 0, 1);

    cout << "  - Sphere colored by Z coordinate (height)" << endl;

    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void demoViewports() {
    cout << "=== Multiple Viewports ===" << endl;
    cout << "Displaying same cloud with different colors in split view..." << endl;

    // Generate cloud
    auto cloud = generateSphereCloud(2000, 1.0);

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer(
        new pcl::visualization::PCLVisualizer("Multiple Viewports"));

    // Create two viewports
    int vp1 = 0, vp2 = 1;
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, vp1);  // Left half
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, vp2);  // Right half

    viewer->setBackgroundColor(0.1, 0.1, 0.1, vp1);
    viewer->setBackgroundColor(0.2, 0.2, 0.2, vp2);

    // Add cloud to viewport 1 (red)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        red(cloud, 255, 100, 100);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, red, "cloud_vp1", vp1);

    // Add cloud to viewport 2 (blue)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        blue(cloud, 100, 100, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, blue, "cloud_vp2", vp2);

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_vp1", vp1);
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_vp2", vp2);

    viewer->addCoordinateSystem(0.3, "coord1", vp1);
    viewer->addCoordinateSystem(0.3, "coord2", vp2);

    // Add text to viewports
    viewer->addText("Viewport 1 (Red)", 10, 10, 15, 1.0, 1.0, 1.0, "text_vp1", vp1);
    viewer->addText("Viewport 2 (Blue)", 10, 10, 15, 1.0, 1.0, 1.0, "text_vp2", vp2);

    viewer->initCameraParameters();

    cout << "  - Left viewport: Red cloud" << endl;
    cout << "  - Right viewport: Blue cloud" << endl;

    auto start = chrono::steady_clock::now();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        if (chrono::duration<double>(chrono::steady_clock::now() - start).count() > 3.0) {
            break;
        }
    }

    viewer->close();
    cout << endl;
}

void printUsage() {
    cout << "\n=== Visualization Controls ===" << endl;
    cout << "Mouse Controls:" << endl;
    cout << "  Left button drag  - Rotate camera" << endl;
    cout << "  Middle button     - Pan camera" << endl;
    cout << "  Scroll wheel      - Zoom in/out" << endl;
    cout << "  Right button drag - Zoom" << endl;
    cout << "\nKeyboard Controls:" << endl;
    cout << "  r - Reset camera view" << endl;
    cout << "  + / - - Increase/decrease point size" << endl;
    cout << "  g - Toggle coordinate system" << endl;
    cout << "  j - Save screenshot" << endl;
    cout << "  q - Close window" << endl;
    cout << endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    cout << "=== PCL Visualization Tutorial ===" << endl;
    cout << "Demonstrating PCL visualization capabilities\n" << endl;

    printUsage();

    // Check for display (headless mode)
    bool has_display = (getenv("DISPLAY") != nullptr);

    if (!has_display) {
        cout << "No display detected (DISPLAY not set)." << endl;
        cout << "Visualization requires X11 display or VNC." << endl;
        cout << "\nTo run with display:" << endl;
        cout << "  1. Use X11 forwarding: ssh -X user@host" << endl;
        cout << "  2. Or use VNC in Docker" << endl;
        cout << "  3. Or set DISPLAY=:0 if running locally" << endl;

        // Generate sample clouds and save for later viewing
        cout << "\nGenerating sample point clouds for offline viewing..." << endl;

        string output_dir = ".";
        if (argc > 1) {
            output_dir = argv[1];
        }

        auto colored_cloud = generateColoredCloud(5000);
        pcl::io::savePCDFileBinary(output_dir + "/viz_colored.pcd", *colored_cloud);

        auto sphere_cloud = generateSphereCloud(2000, 1.0);
        pcl::io::savePCDFileBinary(output_dir + "/viz_sphere.pcd", *sphere_cloud);

        auto plane_normals = generatePlaneWithNormals(500);
        pcl::io::savePCDFileBinary(output_dir + "/viz_plane_normals.pcd", *plane_normals);

        cout << "Saved sample clouds to " << output_dir << endl;
        cout << "Use pcl_viewer to visualize: pcl_viewer viz_colored.pcd" << endl;

        return 0;
    }

    // Run visualization demos
    try {
        demoBasicVisualization();
        demoMultipleClouds();
        demoNormalVisualization();
        demoGeometricPrimitives();
        demoColorMaps();
        demoViewports();
    } catch (const exception& e) {
        cerr << "Visualization error: " << e.what() << endl;
        return 1;
    }

    cout << "=== Visualization Demo Complete ===" << endl;
    return 0;
}
