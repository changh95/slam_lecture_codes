/**
 * Surface Normal Estimation with PCL
 *
 * This example demonstrates:
 * 1. Basic normal estimation using NormalEstimation
 * 2. Parallel normal estimation using NormalEstimationOMP
 * 3. Using radius search vs K-nearest neighbors
 * 4. Visualizing normals on point clouds
 *
 * Surface normals are essential for:
 * - Point cloud registration (GICP, NDT)
 * - Surface reconstruction
 * - Feature extraction (FPFH, SHOT)
 */

#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a point cloud representing a plane with noise
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generatePlaneCloud(int num_points,
                                                         float noise_level = 0.01f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> xy_dist(-2.0f, 2.0f);
    normal_distribution<float> noise_dist(0.0f, noise_level);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointXYZ point;
        point.x = xy_dist(rng);
        point.y = xy_dist(rng);
        point.z = noise_dist(rng);  // Points on Z=0 plane with noise
        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Generate a point cloud representing a sphere
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSphereCloud(int num_points,
                                                          float radius = 1.0f,
                                                          float noise_level = 0.01f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> theta_dist(0.0f, 2.0f * M_PI);
    uniform_real_distribution<float> phi_dist(0.0f, M_PI);
    normal_distribution<float> noise_dist(0.0f, noise_level);

    for (int i = 0; i < num_points; ++i) {
        float theta = theta_dist(rng);
        float phi = phi_dist(rng);
        float r = radius + noise_dist(rng);

        pcl::PointXYZ point;
        point.x = r * sin(phi) * cos(theta);
        point.y = r * sin(phi) * sin(theta);
        point.z = r * cos(phi);
        cloud->points.push_back(point);
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Generate a point cloud representing a box
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateBoxCloud(int points_per_face,
                                                       float size = 1.0f) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-size / 2, size / 2);

    // Generate points on each face of the box
    // Face 1: Z = +size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(dist(rng), dist(rng), size / 2));
    }
    // Face 2: Z = -size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(dist(rng), dist(rng), -size / 2));
    }
    // Face 3: X = +size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(size / 2, dist(rng), dist(rng)));
    }
    // Face 4: X = -size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(-size / 2, dist(rng), dist(rng)));
    }
    // Face 5: Y = +size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(dist(rng), size / 2, dist(rng)));
    }
    // Face 6: Y = -size/2
    for (int i = 0; i < points_per_face; ++i) {
        cloud->points.push_back(pcl::PointXYZ(dist(rng), -size / 2, dist(rng)));
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Time a function and return elapsed milliseconds
 */
template<typename Func>
double timeFunction(Func&& func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

/**
 * Compute normal statistics
 */
void printNormalStats(const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
    int valid_count = 0;
    float avg_curvature = 0.0f;

    for (const auto& n : normals->points) {
        if (!isnan(n.normal_x) && !isnan(n.normal_y) && !isnan(n.normal_z)) {
            valid_count++;
            avg_curvature += n.curvature;
        }
    }

    if (valid_count > 0) {
        avg_curvature /= valid_count;
    }

    cout << "  Valid normals: " << valid_count << "/" << normals->points.size() << endl;
    cout << "  Average curvature: " << avg_curvature << endl;
}

// =============================================================================
// Normal Estimation Demonstrations
// =============================================================================

void demoBasicNormalEstimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                const string& output_dir) {
    cout << "=== Basic Normal Estimation ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>);

    // Create KdTree for efficient neighbor search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    // Normal estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.1);  // Search radius of 10cm

    double time_ms = timeFunction([&]() {
        ne.compute(*normals);
    });

    cout << "  Method: Radius search (0.1m)" << endl;
    cout << "  Time: " << time_ms << " ms" << endl;
    printNormalStats(normals);

    // Save normals with cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    pcl::io::savePCDFileBinary(output_dir + "/cloud_with_normals.pcd", *cloud_with_normals);

    cout << endl;
}

void demoKSearchNormalEstimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                  const string& output_dir) {
    cout << "=== K-Search Normal Estimation ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // Compare different K values
    vector<int> k_values = {10, 20, 50};

    for (int k : k_values) {
        pcl::PointCloud<pcl::Normal>::Ptr normals(
            new pcl::PointCloud<pcl::Normal>);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(k);  // Use K nearest neighbors

        double time_ms = timeFunction([&]() {
            ne.compute(*normals);
        });

        cout << "  K = " << k << ": " << time_ms << " ms" << endl;
        printNormalStats(normals);
    }

    cout << endl;
}

void demoOMPNormalEstimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                              const string& output_dir) {
    cout << "=== OpenMP Parallel Normal Estimation ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // Compare single-threaded vs multi-threaded
    vector<int> thread_counts = {1, 2, 4};

    for (int num_threads : thread_counts) {
        pcl::PointCloud<pcl::Normal>::Ptr normals(
            new pcl::PointCloud<pcl::Normal>);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>);

        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne_omp;
        ne_omp.setNumberOfThreads(num_threads);
        ne_omp.setInputCloud(cloud);
        ne_omp.setSearchMethod(tree);
        ne_omp.setRadiusSearch(0.1);

        double time_ms = timeFunction([&]() {
            ne_omp.compute(*normals);
        });

        cout << "  Threads = " << num_threads << ": " << time_ms << " ms" << endl;
    }

    cout << endl;
}

void demoViewpointNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                           const string& output_dir) {
    cout << "=== Viewpoint-Consistent Normals ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.1);

    // Set viewpoint to orient normals consistently
    // Normals will point towards the viewpoint
    ne.setViewPoint(0.0f, 0.0f, 5.0f);  // Viewpoint above the origin

    ne.compute(*normals);

    // Count normals pointing up (positive Z)
    int up_count = 0;
    int down_count = 0;
    for (const auto& n : normals->points) {
        if (!isnan(n.normal_z)) {
            if (n.normal_z > 0) up_count++;
            else down_count++;
        }
    }

    cout << "  Viewpoint: (0, 0, 5)" << endl;
    cout << "  Normals pointing up (z > 0): " << up_count << endl;
    cout << "  Normals pointing down (z < 0): " << down_count << endl;

    // Save
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    pcl::io::savePCDFileBinary(output_dir + "/normals_viewpoint.pcd", *cloud_with_normals);

    cout << endl;
}

void demoPlaneNormals(const string& output_dir) {
    cout << "=== Plane Normal Estimation ===" << endl;

    // Generate plane cloud (should have normals +-Z)
    auto cloud = generatePlaneCloud(5000, 0.01f);
    cout << "Generated plane cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(30);
    ne.setViewPoint(0.0f, 0.0f, 1.0f);  // View from above

    ne.compute(*normals);

    // Check that normals are approximately (0, 0, 1)
    float avg_nx = 0, avg_ny = 0, avg_nz = 0;
    int count = 0;
    for (const auto& n : normals->points) {
        if (!isnan(n.normal_x)) {
            avg_nx += n.normal_x;
            avg_ny += n.normal_y;
            avg_nz += n.normal_z;
            count++;
        }
    }
    avg_nx /= count;
    avg_ny /= count;
    avg_nz /= count;

    cout << "  Expected normal: (0, 0, 1)" << endl;
    cout << "  Average normal: (" << avg_nx << ", " << avg_ny << ", " << avg_nz << ")" << endl;

    // Save
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    pcl::io::savePCDFileBinary(output_dir + "/plane_normals.pcd", *cloud_with_normals);

    cout << endl;
}

void demoSphereNormals(const string& output_dir) {
    cout << "=== Sphere Normal Estimation ===" << endl;

    // Generate sphere cloud (normals should point radially outward)
    auto cloud = generateSphereCloud(5000, 1.0f, 0.01f);
    cout << "Generated sphere cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::Normal>::Ptr normals(
        new pcl::PointCloud<pcl::Normal>);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(30);
    ne.setViewPoint(0.0f, 0.0f, 0.0f);  // View from center (normals point outward)

    ne.compute(*normals);

    // Check that normals point radially outward
    float avg_dot = 0;
    int count = 0;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        const auto& p = cloud->points[i];
        const auto& n = normals->points[i];

        if (!isnan(n.normal_x)) {
            // Compute dot product with normalized position vector
            float len = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (len > 0.001f) {
                float dot = (p.x * n.normal_x + p.y * n.normal_y + p.z * n.normal_z) / len;
                avg_dot += abs(dot);  // abs because orientation might be flipped
                count++;
            }
        }
    }
    avg_dot /= count;

    cout << "  Normals aligned with radial direction (1.0 = perfect): " << avg_dot << endl;

    // Save
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(
        new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
    pcl::io::savePCDFileBinary(output_dir + "/sphere_normals.pcd", *cloud_with_normals);

    cout << endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    cout << "=== PCL Normal Estimation Tutorial ===" << endl;
    cout << "Computing surface normals for point clouds\n" << endl;

    // Determine output directory
    string output_dir = ".";
    if (argc > 1) {
        output_dir = argv[1];
    }
    cout << "Output directory: " << output_dir << endl << endl;

    // Generate test cloud
    cout << "Generating test point cloud (box)..." << endl;
    auto cloud = generateBoxCloud(1000, 2.0f);
    pcl::io::savePCDFileBinary(output_dir + "/box_input.pcd", *cloud);
    cout << endl;

    // Run demonstrations
    demoBasicNormalEstimation(cloud, output_dir);
    demoKSearchNormalEstimation(cloud, output_dir);
    demoOMPNormalEstimation(cloud, output_dir);
    demoViewpointNormals(cloud, output_dir);
    demoPlaneNormals(output_dir);
    demoSphereNormals(output_dir);

    cout << "=== Normal Estimation Demo Complete ===" << endl;
    return 0;
}
