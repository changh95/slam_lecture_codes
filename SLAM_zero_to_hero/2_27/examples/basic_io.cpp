/**
 * Basic Point Cloud I/O with PCL
 *
 * This example demonstrates:
 * 1. Creating point clouds programmatically
 * 2. Reading and writing PCD files (ASCII and binary)
 * 3. Reading and writing PLY files
 * 4. Working with different point types (XYZ, XYZRGB, XYZI)
 */

#include <iostream>
#include <string>
#include <vector>
#include <random>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a synthetic point cloud (random points in a cube)
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSyntheticCloud(int num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cloud->width = num_points;
    cloud->height = 1;  // Unorganized cloud
    cloud->is_dense = true;
    cloud->points.resize(num_points);

    for (auto& point : cloud->points) {
        point.x = dist(rng);
        point.y = dist(rng);
        point.z = dist(rng);
    }

    return cloud;
}

/**
 * Generate a colored point cloud
 */
pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateColoredCloud(int num_points) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    mt19937 rng(42);
    uniform_real_distribution<float> pos_dist(-1.0f, 1.0f);

    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->points.resize(num_points);

    for (auto& point : cloud->points) {
        point.x = pos_dist(rng);
        point.y = pos_dist(rng);
        point.z = pos_dist(rng);

        // Color based on position
        point.r = static_cast<uint8_t>((point.x + 1.0f) * 127.5f);
        point.g = static_cast<uint8_t>((point.y + 1.0f) * 127.5f);
        point.b = static_cast<uint8_t>((point.z + 1.0f) * 127.5f);
    }

    return cloud;
}

/**
 * Print cloud information
 */
template<typename PointT>
void printCloudInfo(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                    const string& name) {
    cout << "Cloud '" << name << "':" << endl;
    cout << "  Width:    " << cloud->width << endl;
    cout << "  Height:   " << cloud->height << endl;
    cout << "  Points:   " << cloud->points.size() << endl;
    cout << "  Is dense: " << (cloud->is_dense ? "yes" : "no") << endl;

    if (!cloud->points.empty()) {
        cout << "  First point: ("
             << cloud->points[0].x << ", "
             << cloud->points[0].y << ", "
             << cloud->points[0].z << ")" << endl;
    }
    cout << endl;
}

// =============================================================================
// Demo Functions
// =============================================================================

void demoCreateCloud() {
    cout << "=== Creating Point Cloud Programmatically ===" << endl;

    // Method 1: Using width/height
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(
        new pcl::PointCloud<pcl::PointXYZ>);

    cloud1->width = 5;
    cloud1->height = 1;
    cloud1->points.resize(cloud1->width * cloud1->height);

    for (size_t i = 0; i < cloud1->points.size(); ++i) {
        cloud1->points[i].x = static_cast<float>(i) * 0.1f;
        cloud1->points[i].y = 0.0f;
        cloud1->points[i].z = 0.0f;
    }

    printCloudInfo<pcl::PointXYZ>(cloud1, "cloud1 (manual)");

    // Method 2: Push back points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(
        new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; i < 5; ++i) {
        pcl::PointXYZ point;
        point.x = static_cast<float>(i);
        point.y = static_cast<float>(i) * 2.0f;
        point.z = static_cast<float>(i) * 3.0f;
        cloud2->points.push_back(point);
    }
    cloud2->width = cloud2->points.size();
    cloud2->height = 1;

    printCloudInfo<pcl::PointXYZ>(cloud2, "cloud2 (push_back)");
}

void demoPCDIO(const string& output_dir) {
    cout << "=== PCD File I/O ===" << endl;

    // Generate synthetic cloud
    auto cloud = generateSyntheticCloud(1000);
    printCloudInfo<pcl::PointXYZ>(cloud, "generated cloud");

    // Save as ASCII PCD
    string ascii_file = output_dir + "/cloud_ascii.pcd";
    pcl::io::savePCDFileASCII(ascii_file, *cloud);
    cout << "Saved ASCII PCD: " << ascii_file << endl;

    // Save as binary PCD (more efficient)
    string binary_file = output_dir + "/cloud_binary.pcd";
    pcl::io::savePCDFileBinary(binary_file, *cloud);
    cout << "Saved binary PCD: " << binary_file << endl;

    // Save as compressed binary PCD
    string compressed_file = output_dir + "/cloud_compressed.pcd";
    pcl::io::savePCDFileBinaryCompressed(compressed_file, *cloud);
    cout << "Saved compressed PCD: " << compressed_file << endl;

    // Load PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr loaded_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(binary_file, *loaded_cloud) == -1) {
        cerr << "Error: Couldn't read " << binary_file << endl;
        return;
    }

    printCloudInfo<pcl::PointXYZ>(loaded_cloud, "loaded cloud");
    cout << endl;
}

void demoPLYIO(const string& output_dir) {
    cout << "=== PLY File I/O ===" << endl;

    // Generate colored cloud for PLY (supports colors well)
    auto cloud = generateColoredCloud(1000);

    // Save as PLY
    string ply_file = output_dir + "/cloud_colored.ply";
    pcl::io::savePLYFile(ply_file, *cloud);
    cout << "Saved PLY: " << ply_file << endl;

    // Save as binary PLY
    string ply_binary_file = output_dir + "/cloud_colored_binary.ply";
    pcl::io::savePLYFileBinary(ply_binary_file, *cloud);
    cout << "Saved binary PLY: " << ply_binary_file << endl;

    // Load PLY file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr loaded_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(ply_file, *loaded_cloud) == -1) {
        cerr << "Error: Couldn't read " << ply_file << endl;
        return;
    }

    cout << "Loaded PLY with " << loaded_cloud->points.size() << " points" << endl;

    // Print first colored point
    if (!loaded_cloud->points.empty()) {
        const auto& p = loaded_cloud->points[0];
        cout << "First point: ("
             << p.x << ", " << p.y << ", " << p.z
             << ") RGB: ("
             << static_cast<int>(p.r) << ", "
             << static_cast<int>(p.g) << ", "
             << static_cast<int>(p.b) << ")" << endl;
    }
    cout << endl;
}

void demoPointTypes() {
    cout << "=== Different Point Types ===" << endl;

    // PointXYZ - Basic 3D point
    {
        pcl::PointXYZ point;
        point.x = 1.0f;
        point.y = 2.0f;
        point.z = 3.0f;
        cout << "PointXYZ: (" << point.x << ", " << point.y << ", " << point.z << ")" << endl;
    }

    // PointXYZI - Point with intensity (LiDAR)
    {
        pcl::PointXYZI point;
        point.x = 1.0f;
        point.y = 2.0f;
        point.z = 3.0f;
        point.intensity = 100.0f;
        cout << "PointXYZI: (" << point.x << ", " << point.y << ", " << point.z
             << ") intensity: " << point.intensity << endl;
    }

    // PointXYZRGB - Point with color
    {
        pcl::PointXYZRGB point;
        point.x = 1.0f;
        point.y = 2.0f;
        point.z = 3.0f;
        point.r = 255;
        point.g = 128;
        point.b = 0;
        cout << "PointXYZRGB: (" << point.x << ", " << point.y << ", " << point.z
             << ") RGB: (" << static_cast<int>(point.r) << ", "
             << static_cast<int>(point.g) << ", "
             << static_cast<int>(point.b) << ")" << endl;
    }

    // PointNormal - Point with normal vector
    {
        pcl::PointNormal point;
        point.x = 1.0f;
        point.y = 2.0f;
        point.z = 3.0f;
        point.normal_x = 0.0f;
        point.normal_y = 0.0f;
        point.normal_z = 1.0f;
        point.curvature = 0.1f;
        cout << "PointNormal: (" << point.x << ", " << point.y << ", " << point.z
             << ") normal: (" << point.normal_x << ", "
             << point.normal_y << ", "
             << point.normal_z << ")" << endl;
    }

    cout << endl;
}

void demoCloudOperations() {
    cout << "=== Cloud Operations ===" << endl;

    // Generate two clouds
    auto cloud1 = generateSyntheticCloud(500);
    auto cloud2 = generateSyntheticCloud(300);

    cout << "Cloud1: " << cloud1->points.size() << " points" << endl;
    cout << "Cloud2: " << cloud2->points.size() << " points" << endl;

    // Concatenate clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr combined(
        new pcl::PointCloud<pcl::PointXYZ>);
    *combined = *cloud1 + *cloud2;
    cout << "Combined: " << combined->points.size() << " points" << endl;

    // Get a subset of points
    pcl::PointCloud<pcl::PointXYZ>::Ptr subset(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < 100 && i < cloud1->points.size(); ++i) {
        subset->points.push_back(cloud1->points[i]);
    }
    subset->width = subset->points.size();
    subset->height = 1;
    cout << "Subset: " << subset->points.size() << " points" << endl;

    cout << endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    cout << "=== PCL Basic I/O Tutorial ===" << endl;
    cout << "Reading and writing point cloud files\n" << endl;

    // Determine output directory
    string output_dir = ".";
    if (argc > 1) {
        output_dir = argv[1];
    }
    cout << "Output directory: " << output_dir << endl << endl;

    // Run demos
    demoPointTypes();
    demoCreateCloud();
    demoPCDIO(output_dir);
    demoPLYIO(output_dir);
    demoCloudOperations();

    cout << "=== Basic I/O Demo Complete ===" << endl;
    return 0;
}
