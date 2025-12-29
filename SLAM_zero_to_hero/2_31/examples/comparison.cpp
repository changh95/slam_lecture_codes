/**
 * Spatial Data Structures Comparison
 *
 * This example compares three spatial data structures:
 * 1. PCL Octree - Hierarchical spatial partitioning
 * 2. OctoMap - Probabilistic occupancy octree
 * 3. Bonxai - Hash-based high-performance voxel grid
 *
 * Metrics compared:
 * - Insertion time
 * - Query time (point lookup)
 * - Memory usage
 * - Feature capabilities
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/common/common.h>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include "bonxai/bonxai.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

/**
 * Generate synthetic point cloud
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateCloud(int num_points) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-1.0f, 4.0f);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointXYZ pt;
        pt.x = dist(gen);
        pt.y = dist(gen);
        pt.z = dist_z(gen);
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Timing helper
 */
class Timer {
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * Test PCL Octree
 */
void testPCLOctree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    double resolution,
                    const std::vector<pcl::PointXYZ>& query_points,
                    double& insert_time_us,
                    double& query_time_ns,
                    size_t& memory_bytes,
                    size_t& num_leaves) {
    Timer timer;

    // Insertion
    timer.start();
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    insert_time_us = timer.stop();

    // Query (voxel search)
    timer.start();
    for (const auto& pt : query_points) {
        std::vector<int> indices;
        octree.voxelSearch(pt, indices);
    }
    double total_query_us = timer.stop();
    query_time_ns = (total_query_us * 1000.0) / query_points.size();

    // Statistics
    num_leaves = octree.getLeafCount();

    // Estimate memory (rough approximation)
    // PCL octree doesn't provide direct memory usage
    memory_bytes = num_leaves * sizeof(void*) * 8 +  // Leaf pointers
                   octree.getBranchCount() * sizeof(void*) * 8;  // Branch nodes
}

/**
 * Test OctoMap
 */
void testOctoMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                  double resolution,
                  const std::vector<pcl::PointXYZ>& query_points,
                  double& insert_time_us,
                  double& query_time_ns,
                  size_t& memory_bytes,
                  size_t& num_leaves) {
    Timer timer;

    // Insertion
    timer.start();
    octomap::OcTree tree(resolution);
    for (const auto& pt : cloud->points) {
        tree.updateNode(octomap::point3d(pt.x, pt.y, pt.z), true);
    }
    tree.updateInnerOccupancy();
    insert_time_us = timer.stop();

    // Query
    timer.start();
    for (const auto& pt : query_points) {
        octomap::OcTreeNode* node = tree.search(octomap::point3d(pt.x, pt.y, pt.z));
        (void)node;  // Prevent optimization
    }
    double total_query_us = timer.stop();
    query_time_ns = (total_query_us * 1000.0) / query_points.size();

    // Statistics
    num_leaves = tree.getNumLeafNodes();
    memory_bytes = tree.memoryUsage();
}

/**
 * Test Bonxai
 */
void testBonxai(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                 double resolution,
                 const std::vector<pcl::PointXYZ>& query_points,
                 double& insert_time_us,
                 double& query_time_ns,
                 size_t& memory_bytes,
                 size_t& num_leaves) {
    Timer timer;

    // Insertion
    timer.start();
    Bonxai::VoxelGrid<float> grid(resolution);
    {
        auto accessor = grid.createAccessor();
        for (const auto& pt : cloud->points) {
            auto coord = grid.posToCoord(pt.x, pt.y, pt.z);
            accessor.setValue(coord, 1.0f);
        }
    }
    insert_time_us = timer.stop();

    // Query
    timer.start();
    {
        auto accessor = grid.createConstAccessor();
        for (const auto& pt : query_points) {
            auto coord = grid.posToCoord(pt.x, pt.y, pt.z);
            const float* value = accessor.value(coord);
            (void)value;  // Prevent optimization
        }
    }
    double total_query_us = timer.stop();
    query_time_ns = (total_query_us * 1000.0) / query_points.size();

    // Count active voxels
    num_leaves = 0;
    grid.forEachCell([&num_leaves](const float&, const Bonxai::CoordT&) {
        num_leaves++;
    });

    // Estimate memory (rough approximation)
    memory_bytes = num_leaves * sizeof(float) + num_leaves * sizeof(Bonxai::CoordT);
}

/**
 * Format bytes to human-readable string
 */
std::string formatBytes(size_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    } else {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Spatial Data Structures Comparison ===" << std::endl;
    std::cout << std::endl;

    // Test parameters
    std::vector<int> point_counts = {10000, 50000, 100000, 500000};
    double resolution = 0.1;  // 10cm

    if (argc > 1) {
        resolution = std::atof(argv[1]);
    }

    std::cout << "Resolution: " << resolution << "m" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // Performance Comparison
    // ============================================================
    std::cout << "--- Performance Comparison ---" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(10) << "Points"
              << " | " << std::setw(28) << "PCL Octree"
              << " | " << std::setw(28) << "OctoMap"
              << " | " << std::setw(28) << "Bonxai" << std::endl;
    std::cout << std::setw(10) << ""
              << " | " << std::setw(10) << "Insert(us)"
              << std::setw(10) << "Query(ns)"
              << std::setw(10) << "Leaves"
              << " | " << std::setw(10) << "Insert(us)"
              << std::setw(10) << "Query(ns)"
              << std::setw(10) << "Leaves"
              << " | " << std::setw(10) << "Insert(us)"
              << std::setw(10) << "Query(ns)"
              << std::setw(10) << "Leaves" << std::endl;
    std::cout << std::string(120, '-') << std::endl;

    for (int num_points : point_counts) {
        // Generate test data
        auto cloud = generateCloud(num_points);

        // Generate query points
        std::vector<pcl::PointXYZ> query_points;
        std::random_device rd;
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (int i = 0; i < 10000; ++i) {
            pcl::PointXYZ pt;
            pt.x = dist(gen);
            pt.y = dist(gen);
            pt.z = dist(gen);
            query_points.push_back(pt);
        }

        // Test each method
        double pcl_insert, pcl_query, octo_insert, octo_query, bonxai_insert, bonxai_query;
        size_t pcl_mem, octo_mem, bonxai_mem;
        size_t pcl_leaves, octo_leaves, bonxai_leaves;

        testPCLOctree(cloud, resolution, query_points,
                       pcl_insert, pcl_query, pcl_mem, pcl_leaves);
        testOctoMap(cloud, resolution, query_points,
                     octo_insert, octo_query, octo_mem, octo_leaves);
        testBonxai(cloud, resolution, query_points,
                    bonxai_insert, bonxai_query, bonxai_mem, bonxai_leaves);

        std::cout << std::setw(10) << num_points
                  << " | " << std::setw(10) << std::fixed << std::setprecision(0) << pcl_insert
                  << std::setw(10) << std::fixed << std::setprecision(0) << pcl_query
                  << std::setw(10) << pcl_leaves
                  << " | " << std::setw(10) << std::fixed << std::setprecision(0) << octo_insert
                  << std::setw(10) << std::fixed << std::setprecision(0) << octo_query
                  << std::setw(10) << octo_leaves
                  << " | " << std::setw(10) << std::fixed << std::setprecision(0) << bonxai_insert
                  << std::setw(10) << std::fixed << std::setprecision(0) << bonxai_query
                  << std::setw(10) << bonxai_leaves << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // Memory Comparison
    // ============================================================
    std::cout << "--- Memory Usage (500K points) ---" << std::endl;

    auto cloud = generateCloud(500000);
    std::vector<pcl::PointXYZ> query_points;
    std::random_device rd;
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < 1000; ++i) {
        pcl::PointXYZ pt;
        pt.x = dist(gen);
        pt.y = dist(gen);
        pt.z = dist(gen);
        query_points.push_back(pt);
    }

    double dummy1, dummy2;
    size_t pcl_mem, octo_mem, bonxai_mem;
    size_t pcl_leaves, octo_leaves, bonxai_leaves;

    testPCLOctree(cloud, resolution, query_points, dummy1, dummy2, pcl_mem, pcl_leaves);
    testOctoMap(cloud, resolution, query_points, dummy1, dummy2, octo_mem, octo_leaves);
    testBonxai(cloud, resolution, query_points, dummy1, dummy2, bonxai_mem, bonxai_leaves);

    std::cout << "  PCL Octree: " << formatBytes(pcl_mem) << " (estimated)" << std::endl;
    std::cout << "  OctoMap:    " << formatBytes(octo_mem) << " (actual)" << std::endl;
    std::cout << "  Bonxai:     " << formatBytes(bonxai_mem) << " (estimated)" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // Feature Comparison
    // ============================================================
    std::cout << "--- Feature Comparison ---" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(25) << "Feature"
              << std::setw(15) << "PCL Octree"
              << std::setw(15) << "OctoMap"
              << std::setw(15) << "Bonxai" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << std::setw(25) << "Probabilistic"
              << std::setw(15) << "No"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Custom" << std::endl;

    std::cout << std::setw(25) << "Multi-resolution"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Yes"
              << std::setw(15) << "No" << std::endl;

    std::cout << std::setw(25) << "Ray casting"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Custom" << std::endl;

    std::cout << std::setw(25) << "K-NN search"
              << std::setw(15) << "Yes"
              << std::setw(15) << "No"
              << std::setw(15) << "No" << std::endl;

    std::cout << std::setw(25) << "Radius search"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Limited"
              << std::setw(15) << "No" << std::endl;

    std::cout << std::setw(25) << "Serialization"
              << std::setw(15) << "Limited"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Yes" << std::endl;

    std::cout << std::setw(25) << "Header-only"
              << std::setw(15) << "No"
              << std::setw(15) << "No"
              << std::setw(15) << "Yes" << std::endl;

    std::cout << std::setw(25) << "Custom data types"
              << std::setw(15) << "Yes"
              << std::setw(15) << "Limited"
              << std::setw(15) << "Yes" << std::endl;

    std::cout << std::endl;

    // ============================================================
    // Resolution Impact
    // ============================================================
    std::cout << "--- Resolution Impact (100K points) ---" << std::endl;
    std::cout << std::endl;

    cloud = generateCloud(100000);
    std::vector<double> resolutions = {0.05, 0.1, 0.2, 0.5, 1.0};

    std::cout << std::setw(10) << "Resolution"
              << " | " << std::setw(20) << "PCL Leaves"
              << " | " << std::setw(20) << "OctoMap Leaves"
              << " | " << std::setw(20) << "Bonxai Voxels" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    for (double res : resolutions) {
        testPCLOctree(cloud, res, query_points, dummy1, dummy2, pcl_mem, pcl_leaves);
        testOctoMap(cloud, res, query_points, dummy1, dummy2, octo_mem, octo_leaves);
        testBonxai(cloud, res, query_points, dummy1, dummy2, bonxai_mem, bonxai_leaves);

        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << res << "m"
                  << " | " << std::setw(20) << pcl_leaves
                  << " | " << std::setw(20) << octo_leaves
                  << " | " << std::setw(20) << bonxai_leaves << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // Summary
    // ============================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << std::endl;
    std::cout << "PCL Octree:" << std::endl;
    std::cout << "  - Best for: Spatial queries (KNN, radius), point cloud processing" << std::endl;
    std::cout << "  - Integrates with PCL ecosystem" << std::endl;
    std::cout << "  - Fast insertion, good query performance" << std::endl;
    std::cout << std::endl;

    std::cout << "OctoMap:" << std::endl;
    std::cout << "  - Best for: Navigation, probabilistic mapping, path planning" << std::endl;
    std::cout << "  - Built-in occupancy probability updates" << std::endl;
    std::cout << "  - Multi-resolution queries for hierarchical planning" << std::endl;
    std::cout << "  - Slower insertion but memory-efficient" << std::endl;
    std::cout << std::endl;

    std::cout << "Bonxai:" << std::endl;
    std::cout << "  - Best for: Real-time applications, large-scale maps" << std::endl;
    std::cout << "  - Fastest insertion and query times" << std::endl;
    std::cout << "  - Header-only, easy integration" << std::endl;
    std::cout << "  - Custom data types with minimal overhead" << std::endl;
    std::cout << "  - No built-in multi-resolution (trade-off for speed)" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Comparison Complete ===" << std::endl;

    return 0;
}
