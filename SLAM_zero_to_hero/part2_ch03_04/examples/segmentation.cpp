/**
 * Point Cloud Segmentation with PCL
 *
 * This example demonstrates:
 * 1. RANSAC plane segmentation (ground plane extraction)
 * 2. Euclidean cluster extraction (object detection)
 * 3. Region growing segmentation
 * 4. Multi-plane segmentation
 *
 * These techniques are essential for:
 * - Autonomous driving (ground/obstacle separation)
 * - Object detection and tracking
 * - Scene understanding in SLAM
 */

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

using namespace std;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a scene with ground plane and objects (clusters)
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSceneCloud(int ground_points,
                                                         int num_objects,
                                                         int points_per_object) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> ground_xy(-5.0f, 5.0f);
    normal_distribution<float> ground_noise(0.0f, 0.02f);
    uniform_real_distribution<float> obj_center(-3.0f, 3.0f);
    uniform_real_distribution<float> obj_height(0.3f, 1.5f);
    normal_distribution<float> obj_spread(0.0f, 0.2f);

    // Generate ground plane (z = 0 with noise)
    for (int i = 0; i < ground_points; ++i) {
        pcl::PointXYZ point;
        point.x = ground_xy(rng);
        point.y = ground_xy(rng);
        point.z = ground_noise(rng);
        cloud->points.push_back(point);
    }

    // Generate objects (clusters above ground)
    for (int obj = 0; obj < num_objects; ++obj) {
        float cx = obj_center(rng);
        float cy = obj_center(rng);
        float cz = obj_height(rng);

        for (int i = 0; i < points_per_object; ++i) {
            pcl::PointXYZ point;
            point.x = cx + obj_spread(rng);
            point.y = cy + obj_spread(rng);
            point.z = cz + obj_spread(rng);
            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Generate a multi-plane scene (walls and floor)
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateMultiPlaneCloud(int points_per_plane) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    mt19937 rng(42);
    uniform_real_distribution<float> coord_dist(-2.0f, 2.0f);
    normal_distribution<float> noise(0.0f, 0.02f);

    // Floor (z = 0)
    for (int i = 0; i < points_per_plane; ++i) {
        cloud->points.push_back(pcl::PointXYZ(
            coord_dist(rng), coord_dist(rng), noise(rng)));
    }

    // Wall 1 (x = 2)
    for (int i = 0; i < points_per_plane; ++i) {
        cloud->points.push_back(pcl::PointXYZ(
            2.0f + noise(rng), coord_dist(rng), coord_dist(rng) + 2.0f));
    }

    // Wall 2 (y = 2)
    for (int i = 0; i < points_per_plane; ++i) {
        cloud->points.push_back(pcl::PointXYZ(
            coord_dist(rng), 2.0f + noise(rng), coord_dist(rng) + 2.0f));
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

// =============================================================================
// Segmentation Demonstrations
// =============================================================================

void demoRANSACPlaneSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                   const string& output_dir) {
    cout << "=== RANSAC Plane Segmentation ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // Setup RANSAC plane segmentation
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.05);  // 5cm inlier threshold

    seg.setInputCloud(cloud);

    // Segment
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    double time_ms = timeFunction([&]() {
        seg.segment(*inliers, *coefficients);
    });

    if (inliers->indices.empty()) {
        cerr << "No plane found!" << endl;
        return;
    }

    cout << "  Plane equation: "
         << coefficients->values[0] << "x + "
         << coefficients->values[1] << "y + "
         << coefficients->values[2] << "z + "
         << coefficients->values[3] << " = 0" << endl;
    cout << "  Inliers: " << inliers->indices.size() << " points" << endl;
    cout << "  Time: " << time_ms << " ms" << endl;

    // Extract ground points
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(
        new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(false);
    extract.filter(*ground);

    // Extract non-ground points (obstacles)
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles(
        new pcl::PointCloud<pcl::PointXYZ>);
    extract.setNegative(true);
    extract.filter(*obstacles);

    cout << "  Ground points: " << ground->points.size() << endl;
    cout << "  Obstacle points: " << obstacles->points.size() << endl;

    // Save results
    pcl::io::savePCDFileBinary(output_dir + "/ground.pcd", *ground);
    pcl::io::savePCDFileBinary(output_dir + "/obstacles.pcd", *obstacles);

    cout << endl;
}

void demoEuclideanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               const string& output_dir) {
    cout << "=== Euclidean Cluster Extraction ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    // First, remove ground plane
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.05);
    seg.setInputCloud(cloud);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(
        new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_objects);

    cout << "  After ground removal: " << cloud_objects->points.size() << " points" << endl;

    // Create KdTree for clustering
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_objects);

    // Euclidean clustering
    vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);    // 30cm distance threshold
    ec.setMinClusterSize(50);       // Minimum points per cluster
    ec.setMaxClusterSize(25000);    // Maximum points per cluster
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_objects);

    double time_ms = timeFunction([&]() {
        ec.extract(cluster_indices);
    });

    cout << "  Found " << cluster_indices.size() << " clusters (" << time_ms << " ms)" << endl;

    // Extract and save each cluster
    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(
            new pcl::PointCloud<pcl::PointXYZ>);

        for (const auto& idx : indices.indices) {
            cluster->points.push_back(cloud_objects->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        // Compute cluster centroid
        float cx = 0, cy = 0, cz = 0;
        for (const auto& p : cluster->points) {
            cx += p.x;
            cy += p.y;
            cz += p.z;
        }
        cx /= cluster->points.size();
        cy /= cluster->points.size();
        cz /= cluster->points.size();

        cout << "  Cluster " << cluster_id << ": "
             << cluster->points.size() << " points, centroid: ("
             << cx << ", " << cy << ", " << cz << ")" << endl;

        string filename = output_dir + "/cluster_" + to_string(cluster_id) + ".pcd";
        pcl::io::savePCDFileBinary(filename, *cluster);

        cluster_id++;
    }

    cout << endl;
}

void demoMultiPlaneSegmentation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                  const string& output_dir) {
    cout << "=== Multi-Plane Segmentation ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remaining(
        new pcl::PointCloud<pcl::PointXYZ>(*cloud));

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.05);

    pcl::ExtractIndices<pcl::PointXYZ> extract;

    int plane_id = 0;
    int min_points = 500;  // Minimum points for a valid plane

    while (cloud_remaining->points.size() > static_cast<size_t>(min_points)) {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setInputCloud(cloud_remaining);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() < static_cast<size_t>(min_points)) {
            break;
        }

        // Extract plane
        pcl::PointCloud<pcl::PointXYZ>::Ptr plane(
            new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(cloud_remaining);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);

        cout << "  Plane " << plane_id << ": "
             << plane->points.size() << " points, equation: "
             << coefficients->values[0] << "x + "
             << coefficients->values[1] << "y + "
             << coefficients->values[2] << "z + "
             << coefficients->values[3] << " = 0" << endl;

        string filename = output_dir + "/plane_" + to_string(plane_id) + ".pcd";
        pcl::io::savePCDFileBinary(filename, *plane);

        // Remove plane from remaining cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(
            new pcl::PointCloud<pcl::PointXYZ>);
        extract.setNegative(true);
        extract.filter(*cloud_temp);
        cloud_remaining = cloud_temp;

        plane_id++;
    }

    cout << "  Remaining points: " << cloud_remaining->points.size() << endl;

    if (!cloud_remaining->points.empty()) {
        pcl::io::savePCDFileBinary(output_dir + "/remaining.pcd", *cloud_remaining);
    }

    cout << endl;
}

void demoClusteringParameters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                const string& output_dir) {
    cout << "=== Clustering Parameter Comparison ===" << endl;

    // Remove ground first
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.05);
    seg.setInputCloud(cloud);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.segment(*inliers, *coefficients);

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(
        new pcl::PointCloud<pcl::PointXYZ>);
    extract.filter(*cloud_objects);

    cout << "  Objects cloud: " << cloud_objects->points.size() << " points" << endl;

    // Test different tolerance values
    vector<float> tolerances = {0.1f, 0.3f, 0.5f, 1.0f};

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_objects);

    for (float tol : tolerances) {
        vector<pcl::PointIndices> cluster_indices;

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(tol);
        ec.setMinClusterSize(20);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_objects);
        ec.extract(cluster_indices);

        cout << "  Tolerance " << tol << "m: " << cluster_indices.size() << " clusters" << endl;
    }

    cout << endl;
}

void demoSegmentationPipeline(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                const string& output_dir) {
    cout << "=== Complete Segmentation Pipeline ===" << endl;
    cout << "Input cloud: " << cloud->points.size() << " points" << endl;

    auto pipeline_start = chrono::high_resolution_clock::now();

    // Step 1: Downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(0.05f, 0.05f, 0.05f);
        voxel.filter(*cloud_downsampled);
        cout << "  1. Downsampled: " << cloud_downsampled->points.size() << " points" << endl;
    }

    // Step 2: Ground segmentation
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles(
        new pcl::PointCloud<pcl::PointXYZ>);
    {
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(1000);
        seg.setDistanceThreshold(0.05);
        seg.setInputCloud(cloud_downsampled);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        seg.segment(*inliers, *coefficients);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_downsampled);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*ground);
        extract.setNegative(true);
        extract.filter(*obstacles);

        cout << "  2. Ground: " << ground->points.size()
             << ", Obstacles: " << obstacles->points.size() << endl;
    }

    // Step 3: Cluster obstacles
    vector<pcl::PointIndices> cluster_indices;
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(obstacles);

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.3);
        ec.setMinClusterSize(20);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(obstacles);
        ec.extract(cluster_indices);

        cout << "  3. Found " << cluster_indices.size() << " object clusters" << endl;
    }

    // Step 4: Compute bounding boxes for each cluster
    cout << "  4. Object bounding boxes:" << endl;
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        float min_x = numeric_limits<float>::max();
        float min_y = numeric_limits<float>::max();
        float min_z = numeric_limits<float>::max();
        float max_x = numeric_limits<float>::lowest();
        float max_y = numeric_limits<float>::lowest();
        float max_z = numeric_limits<float>::lowest();

        for (const auto& idx : cluster_indices[i].indices) {
            const auto& p = obstacles->points[idx];
            min_x = min(min_x, p.x);
            min_y = min(min_y, p.y);
            min_z = min(min_z, p.z);
            max_x = max(max_x, p.x);
            max_y = max(max_y, p.y);
            max_z = max(max_z, p.z);
        }

        float size_x = max_x - min_x;
        float size_y = max_y - min_y;
        float size_z = max_z - min_z;

        cout << "     Object " << i << ": "
             << cluster_indices[i].indices.size() << " pts, size: ("
             << size_x << " x " << size_y << " x " << size_z << ") m" << endl;
    }

    auto pipeline_end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration<double, milli>(pipeline_end - pipeline_start).count();

    cout << "  Total pipeline time: " << total_time << " ms" << endl;

    // Save results
    pcl::io::savePCDFileBinary(output_dir + "/pipeline_ground.pcd", *ground);
    pcl::io::savePCDFileBinary(output_dir + "/pipeline_obstacles.pcd", *obstacles);

    cout << endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    cout << "=== PCL Segmentation Tutorial ===" << endl;
    cout << "Plane segmentation and clustering\n" << endl;

    // Determine output directory
    string output_dir = ".";
    if (argc > 1) {
        output_dir = argv[1];
    }
    cout << "Output directory: " << output_dir << endl << endl;

    // Generate scene with ground and objects
    cout << "Generating scene with ground plane and objects..." << endl;
    auto scene = generateSceneCloud(5000, 5, 200);
    pcl::io::savePCDFileBinary(output_dir + "/scene_input.pcd", *scene);
    cout << "Scene: " << scene->points.size() << " points" << endl << endl;

    // Run segmentation demonstrations
    demoRANSACPlaneSegmentation(scene, output_dir);
    demoEuclideanClustering(scene, output_dir);
    demoClusteringParameters(scene, output_dir);
    demoSegmentationPipeline(scene, output_dir);

    // Multi-plane segmentation demo
    cout << "Generating multi-plane scene..." << endl;
    auto multi_plane = generateMultiPlaneCloud(1000);
    pcl::io::savePCDFileBinary(output_dir + "/multiplane_input.pcd", *multi_plane);
    cout << endl;

    demoMultiPlaneSegmentation(multi_plane, output_dir);

    cout << "=== Segmentation Demo Complete ===" << endl;
    return 0;
}
