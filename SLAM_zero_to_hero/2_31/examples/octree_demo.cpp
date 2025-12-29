/**
 * PCL Octree Demo: Spatial Data Structure for Point Clouds
 *
 * This example demonstrates how to use PCL's Octree for:
 * 1. Building an octree from point cloud data
 * 2. Voxel search - finding points in the same voxel
 * 3. K-nearest neighbor search
 * 4. Radius search
 * 5. Octree statistics and structure analysis
 *
 * An octree recursively divides 3D space into 8 octants, providing
 * adaptive resolution based on data density. This is fundamental
 * for efficient spatial queries in SLAM and robotics applications.
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/common/common.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

/**
 * Generate a synthetic point cloud for demonstration
 * Creates a scene with random points and some clustered regions
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSyntheticCloud(int num_points = 10000) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Random points in a 20m x 20m x 5m space
    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-1.0f, 4.0f);

    // Add random scattered points
    for (int i = 0; i < num_points / 2; ++i) {
        pcl::PointXYZ pt;
        pt.x = dist_x(gen);
        pt.y = dist_y(gen);
        pt.z = dist_z(gen);
        cloud->push_back(pt);
    }

    // Add clustered regions (simulating objects)
    std::vector<std::tuple<float, float, float>> cluster_centers = {
        {5.0f, 5.0f, 1.0f},
        {-5.0f, 3.0f, 0.5f},
        {0.0f, -7.0f, 2.0f},
        {3.0f, -3.0f, 1.5f}
    };

    std::normal_distribution<float> cluster_dist(0.0f, 0.5f);

    for (const auto& center : cluster_centers) {
        for (int i = 0; i < num_points / 8; ++i) {
            pcl::PointXYZ pt;
            pt.x = std::get<0>(center) + cluster_dist(gen);
            pt.y = std::get<1>(center) + cluster_dist(gen);
            pt.z = std::get<2>(center) + cluster_dist(gen);
            cloud->push_back(pt);
        }
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Load point cloud from PCD file
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
        std::cerr << "Error: Could not load " << filename << std::endl;
        return nullptr;
    }

    std::cout << "Loaded " << cloud->size() << " points from " << filename << std::endl;
    return cloud;
}

int main(int argc, char* argv[]) {
    std::cout << "=== PCL Octree Demo ===" << std::endl;
    std::cout << std::endl;

    // Load or generate point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    if (argc > 1) {
        cloud = loadPointCloud(argv[1]);
        if (!cloud) {
            return -1;
        }
    } else {
        std::cout << "No input file provided. Generating synthetic point cloud..." << std::endl;
        cloud = generateSyntheticCloud(10000);
        std::cout << "Generated " << cloud->size() << " points" << std::endl;
    }
    std::cout << std::endl;

    // Get point cloud bounds
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    std::cout << "Point cloud bounds:" << std::endl;
    std::cout << "  X: [" << min_pt.x << ", " << max_pt.x << "]" << std::endl;
    std::cout << "  Y: [" << min_pt.y << ", " << max_pt.y << "]" << std::endl;
    std::cout << "  Z: [" << min_pt.z << ", " << max_pt.z << "]" << std::endl;
    std::cout << std::endl;

    // Create octree with specified resolution
    float resolution = 0.5f;  // 50cm voxels
    std::cout << "Creating octree with resolution: " << resolution << "m" << std::endl;

    auto start_build = std::chrono::high_resolution_clock::now();

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();

    auto end_build = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_build - start_build).count();

    std::cout << "  Build time: " << build_time << " us" << std::endl;
    std::cout << "  Leaf count: " << octree.getLeafCount() << std::endl;
    std::cout << "  Branch count: " << octree.getBranchCount() << std::endl;
    std::cout << "  Tree depth: " << octree.getTreeDepth() << std::endl;
    std::cout << std::endl;

    // Define search point
    pcl::PointXYZ search_point;
    search_point.x = 5.0f;
    search_point.y = 5.0f;
    search_point.z = 1.0f;

    std::cout << "Search point: (" << search_point.x << ", "
              << search_point.y << ", " << search_point.z << ")" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 1. Voxel Search
    // ============================================================
    std::cout << "--- 1. Voxel Search ---" << std::endl;
    std::cout << "Finding all points within the same voxel as search point..." << std::endl;

    std::vector<int> voxel_indices;

    auto start_voxel = std::chrono::high_resolution_clock::now();
    bool found = octree.voxelSearch(search_point, voxel_indices);
    auto end_voxel = std::chrono::high_resolution_clock::now();

    if (found) {
        std::cout << "  Found " << voxel_indices.size() << " points in voxel" << std::endl;
        std::cout << "  First 5 points:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), voxel_indices.size()); ++i) {
            const auto& pt = (*cloud)[voxel_indices[i]];
            std::cout << "    [" << i << "] (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
        }
    } else {
        std::cout << "  No points found in voxel" << std::endl;
    }

    auto voxel_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_voxel - start_voxel).count();
    std::cout << "  Query time: " << voxel_time << " ns" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 2. K-Nearest Neighbor Search
    // ============================================================
    std::cout << "--- 2. K-Nearest Neighbor Search ---" << std::endl;
    int k = 10;
    std::cout << "Finding " << k << " nearest neighbors..." << std::endl;

    std::vector<int> knn_indices(k);
    std::vector<float> knn_distances(k);

    auto start_knn = std::chrono::high_resolution_clock::now();
    int num_found = octree.nearestKSearch(search_point, k, knn_indices, knn_distances);
    auto end_knn = std::chrono::high_resolution_clock::now();

    std::cout << "  Found " << num_found << " neighbors:" << std::endl;
    for (int i = 0; i < num_found; ++i) {
        const auto& pt = (*cloud)[knn_indices[i]];
        std::cout << "    [" << i << "] (" << pt.x << ", " << pt.y << ", " << pt.z
                  << ") - distance: " << std::sqrt(knn_distances[i]) << "m" << std::endl;
    }

    auto knn_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_knn - start_knn).count();
    std::cout << "  Query time: " << knn_time << " ns" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 3. Radius Search
    // ============================================================
    std::cout << "--- 3. Radius Search ---" << std::endl;
    float radius = 1.0f;
    std::cout << "Finding all points within " << radius << "m radius..." << std::endl;

    std::vector<int> radius_indices;
    std::vector<float> radius_distances;

    auto start_radius = std::chrono::high_resolution_clock::now();
    int num_radius = octree.radiusSearch(search_point, radius, radius_indices, radius_distances);
    auto end_radius = std::chrono::high_resolution_clock::now();

    std::cout << "  Found " << num_radius << " points" << std::endl;
    std::cout << "  First 10 points:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), radius_indices.size()); ++i) {
        const auto& pt = (*cloud)[radius_indices[i]];
        std::cout << "    [" << i << "] (" << pt.x << ", " << pt.y << ", " << pt.z
                  << ") - distance: " << std::sqrt(radius_distances[i]) << "m" << std::endl;
    }

    auto radius_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end_radius - start_radius).count();
    std::cout << "  Query time: " << radius_time << " ns" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 4. Check if point is within bounds
    // ============================================================
    std::cout << "--- 4. Point Existence Check ---" << std::endl;
    pcl::PointXYZ test_point;
    test_point.x = search_point.x;
    test_point.y = search_point.y;
    test_point.z = search_point.z;

    bool exists = octree.isVoxelOccupiedAtPoint(test_point);
    std::cout << "  Point (" << test_point.x << ", " << test_point.y << ", " << test_point.z << ")"
              << " voxel occupied: " << (exists ? "yes" : "no") << std::endl;

    test_point.x = 100.0f;  // Far outside
    exists = octree.isVoxelOccupiedAtPoint(test_point);
    std::cout << "  Point (" << test_point.x << ", " << test_point.y << ", " << test_point.z << ")"
              << " voxel occupied: " << (exists ? "yes" : "no") << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 5. Iterate over occupied voxels
    // ============================================================
    std::cout << "--- 5. Occupied Voxel Centers ---" << std::endl;
    std::cout << "First 10 occupied voxel centers:" << std::endl;

    int voxel_count = 0;
    for (auto it = octree.begin(); it != octree.end(); ++it) {
        if (octree.isVoxelOccupiedAtPoint(it.getCurrentOctreeKey())) {
            pcl::PointXYZ voxel_center;
            octree.getVoxelBounds(it, voxel_center.x, voxel_center.y, voxel_center.z);

            if (voxel_count < 10) {
                std::cout << "  Voxel " << voxel_count << ": center approximately at ("
                          << voxel_center.x << ", " << voxel_center.y << ", " << voxel_center.z << ")"
                          << std::endl;
            }
            voxel_count++;
        }
    }
    std::cout << "  Total occupied voxels: " << octree.getLeafCount() << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 6. Performance Comparison: Different Resolutions
    // ============================================================
    std::cout << "--- 6. Resolution Comparison ---" << std::endl;
    std::vector<float> resolutions = {0.1f, 0.25f, 0.5f, 1.0f, 2.0f};

    std::cout << "Resolution | Leaves   | Branches | Depth | Build(us) | KNN(ns)" << std::endl;
    std::cout << "-----------|----------|----------|-------|-----------|--------" << std::endl;

    for (float res : resolutions) {
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> test_octree(res);

        auto t1 = std::chrono::high_resolution_clock::now();
        test_octree.setInputCloud(cloud);
        test_octree.addPointsFromInputCloud();
        auto t2 = std::chrono::high_resolution_clock::now();

        std::vector<int> test_indices(10);
        std::vector<float> test_distances(10);
        auto t3 = std::chrono::high_resolution_clock::now();
        test_octree.nearestKSearch(search_point, 10, test_indices, test_distances);
        auto t4 = std::chrono::high_resolution_clock::now();

        auto build_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto knn_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();

        printf("  %6.2fm  | %8lu | %8lu | %5d | %9ld | %6ld\n",
               res,
               static_cast<unsigned long>(test_octree.getLeafCount()),
               static_cast<unsigned long>(test_octree.getBranchCount()),
               test_octree.getTreeDepth(),
               build_us,
               knn_ns);
    }
    std::cout << std::endl;

    std::cout << "=== Octree Demo Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Key takeaways:" << std::endl;
    std::cout << "  - Octrees provide O(log n) query time for spatial searches" << std::endl;
    std::cout << "  - Resolution affects memory usage and query precision" << std::endl;
    std::cout << "  - Higher resolution = more leaves, more precise, more memory" << std::endl;
    std::cout << "  - Use for: nearest neighbor search, collision detection, spatial indexing" << std::endl;

    return 0;
}
