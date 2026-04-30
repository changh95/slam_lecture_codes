/**
 * OctoMap Demo: Probabilistic 3D Occupancy Mapping
 *
 * This example demonstrates how to use OctoMap for:
 * 1. Creating an occupancy octree with probabilistic updates
 * 2. Inserting point clouds with ray casting (marks free space)
 * 3. Querying occupancy probabilities
 * 4. Multi-resolution queries
 * 5. Sensor model configuration
 * 6. Saving and loading maps
 *
 * OctoMap extends octrees with probabilistic occupancy modeling,
 * making it ideal for navigation and mapping in robotics.
 */

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

/**
 * Generate a synthetic point cloud simulating a LiDAR scan
 * Creates a scene with walls, obstacles, and ground plane
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateLidarScan(
    const octomap::point3d& sensor_origin,
    int num_rays = 1000)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> noise(-0.02f, 0.02f);  // 2cm noise

    // Simulate rays hitting various surfaces
    for (int i = 0; i < num_rays; ++i) {
        float azimuth = static_cast<float>(i) / num_rays * 2.0f * M_PI;
        float elevation = -0.2f + 0.4f * std::sin(azimuth * 3);

        // Ray direction
        float dx = std::cos(azimuth) * std::cos(elevation);
        float dy = std::sin(azimuth) * std::cos(elevation);
        float dz = std::sin(elevation);

        // Simulate different surfaces
        float range;

        // Ground plane (z = -0.5)
        if (dz < -0.1f) {
            range = (-0.5f - sensor_origin.z()) / dz;
        }
        // Walls at various distances
        else if (std::abs(dx) > std::abs(dy)) {
            range = (dx > 0 ? 8.0f : 5.0f) / std::abs(dx);
        } else {
            range = (dy > 0 ? 10.0f : 6.0f) / std::abs(dy);
        }

        // Add some obstacles
        if (azimuth > 0.5f && azimuth < 1.0f) {
            range = std::min(range, 3.0f);
        }
        if (azimuth > 2.5f && azimuth < 3.0f) {
            range = std::min(range, 4.0f);
        }

        // Limit max range
        range = std::min(range, 15.0f);

        pcl::PointXYZ pt;
        pt.x = sensor_origin.x() + range * dx + noise(gen);
        pt.y = sensor_origin.y() + range * dy + noise(gen);
        pt.z = sensor_origin.z() + range * dz + noise(gen);
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

int main(int argc, char* argv[]) {
    std::cout << "=== OctoMap Probabilistic Mapping Demo ===" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 1. Create OctoMap with resolution
    // ============================================================
    double resolution = 0.1;  // 10cm voxels
    std::cout << "Creating OctoMap with resolution: " << resolution << "m" << std::endl;

    octomap::OcTree tree(resolution);

    // Configure sensor model
    tree.setProbHit(0.7);      // P(occupied | hit) - probability when sensor sees obstacle
    tree.setProbMiss(0.4);     // P(occupied | miss) - probability for free space
    tree.setClampingThresMin(0.12);  // Minimum occupancy probability
    tree.setClampingThresMax(0.97);  // Maximum occupancy probability

    std::cout << "Sensor model:" << std::endl;
    std::cout << "  P(hit):  " << tree.getProbHit() << std::endl;
    std::cout << "  P(miss): " << tree.getProbMiss() << std::endl;
    std::cout << "  Clamp min: " << tree.getClampingThresMin() << std::endl;
    std::cout << "  Clamp max: " << tree.getClampingThresMax() << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 2. Insert point clouds from multiple sensor poses
    // ============================================================
    std::cout << "--- Inserting point clouds from multiple poses ---" << std::endl;

    std::vector<octomap::point3d> sensor_poses = {
        octomap::point3d(0.0, 0.0, 0.5),
        octomap::point3d(2.0, 0.0, 0.5),
        octomap::point3d(2.0, 2.0, 0.5),
        octomap::point3d(0.0, 2.0, 0.5),
        octomap::point3d(1.0, 1.0, 0.5)
    };

    auto start_insert = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < sensor_poses.size(); ++i) {
        const auto& origin = sensor_poses[i];

        // Generate simulated LiDAR scan
        auto cloud = generateLidarScan(origin, 500);

        // Convert PCL cloud to OctoMap pointcloud
        octomap::Pointcloud octo_cloud;
        for (const auto& pt : cloud->points) {
            octo_cloud.push_back(pt.x, pt.y, pt.z);
        }

        // Insert with ray casting
        // This marks:
        // - Endpoint voxels as OCCUPIED
        // - All voxels along the ray as FREE
        tree.insertPointCloud(octo_cloud, origin);

        std::cout << "  Pose " << i << ": origin=(" << origin.x() << ", "
                  << origin.y() << ", " << origin.z() << "), "
                  << cloud->size() << " points" << std::endl;
    }

    auto end_insert = std::chrono::high_resolution_clock::now();
    auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_insert - start_insert).count();

    // Update inner occupancy for consistent tree
    tree.updateInnerOccupancy();

    std::cout << std::endl;
    std::cout << "Insertion complete:" << std::endl;
    std::cout << "  Total time: " << insert_time << " ms" << std::endl;
    std::cout << "  Tree depth: " << tree.getTreeDepth() << std::endl;
    std::cout << "  Num leaf nodes: " << tree.getNumLeafNodes() << std::endl;
    std::cout << "  Memory usage: " << tree.memoryUsage() / 1024.0 << " KB" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 3. Query occupancy at specific points
    // ============================================================
    std::cout << "--- Querying occupancy ---" << std::endl;

    std::vector<octomap::point3d> query_points = {
        octomap::point3d(0.0, 0.0, 0.5),    // Should be free (sensor location)
        octomap::point3d(8.0, 0.0, 0.5),    // Should be occupied (wall)
        octomap::point3d(4.0, 0.0, 0.5),    // Should be free (middle of room)
        octomap::point3d(0.0, 0.0, -0.5),   // Ground plane
        octomap::point3d(100.0, 100.0, 0.0) // Unknown (never observed)
    };

    for (const auto& query : query_points) {
        octomap::OcTreeNode* node = tree.search(query);

        std::cout << "  Point (" << query.x() << ", " << query.y() << ", " << query.z() << "): ";

        if (node == nullptr) {
            std::cout << "UNKNOWN (not observed)" << std::endl;
        } else {
            double occupancy = node->getOccupancy();
            std::cout << "occupancy=" << occupancy;

            if (tree.isNodeOccupied(node)) {
                std::cout << " [OCCUPIED]";
            } else {
                std::cout << " [FREE]";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // ============================================================
    // 4. Multi-resolution queries
    // ============================================================
    std::cout << "--- Multi-resolution query ---" << std::endl;
    std::cout << "Querying same point at different tree depths:" << std::endl;

    octomap::point3d multi_res_query(4.0, 0.0, 0.0);
    std::cout << "Query point: (" << multi_res_query.x() << ", "
              << multi_res_query.y() << ", " << multi_res_query.z() << ")" << std::endl;

    for (unsigned int depth = tree.getTreeDepth(); depth >= tree.getTreeDepth() - 4; --depth) {
        double node_resolution = tree.getResolution() *
            std::pow(2, tree.getTreeDepth() - depth);

        octomap::OcTreeNode* node = tree.search(multi_res_query, depth);

        std::cout << "  Depth " << depth << " (res=" << node_resolution << "m): ";
        if (node) {
            std::cout << "occupancy=" << node->getOccupancy() << std::endl;
        } else {
            std::cout << "not found" << std::endl;
        }
    }
    std::cout << std::endl;

    // ============================================================
    // 5. Iterate over occupied/free voxels
    // ============================================================
    std::cout << "--- Voxel statistics ---" << std::endl;

    int num_occupied = 0;
    int num_free = 0;
    octomap::point3d occupied_min(1e9, 1e9, 1e9);
    octomap::point3d occupied_max(-1e9, -1e9, -1e9);

    for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
        if (tree.isNodeOccupied(*it)) {
            num_occupied++;

            octomap::point3d coord = it.getCoordinate();
            occupied_min.x() = std::min(occupied_min.x(), coord.x());
            occupied_min.y() = std::min(occupied_min.y(), coord.y());
            occupied_min.z() = std::min(occupied_min.z(), coord.z());
            occupied_max.x() = std::max(occupied_max.x(), coord.x());
            occupied_max.y() = std::max(occupied_max.y(), coord.y());
            occupied_max.z() = std::max(occupied_max.z(), coord.z());
        } else {
            num_free++;
        }
    }

    std::cout << "  Occupied voxels: " << num_occupied << std::endl;
    std::cout << "  Free voxels: " << num_free << std::endl;
    std::cout << "  Occupied bounds:" << std::endl;
    std::cout << "    Min: (" << occupied_min.x() << ", " << occupied_min.y()
              << ", " << occupied_min.z() << ")" << std::endl;
    std::cout << "    Max: (" << occupied_max.x() << ", " << occupied_max.y()
              << ", " << occupied_max.z() << ")" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 6. Ray casting for collision checking
    // ============================================================
    std::cout << "--- Ray casting ---" << std::endl;

    octomap::point3d ray_origin(0.0, 0.0, 0.5);
    octomap::point3d ray_direction(1.0, 0.0, 0.0);  // Looking along +X

    octomap::point3d hit_point;
    bool hit = tree.castRay(ray_origin, ray_direction, hit_point, true, 20.0);

    std::cout << "  Ray from (" << ray_origin.x() << ", " << ray_origin.y()
              << ", " << ray_origin.z() << ") in direction ("
              << ray_direction.x() << ", " << ray_direction.y() << ", "
              << ray_direction.z() << "):" << std::endl;

    if (hit) {
        std::cout << "  Hit obstacle at: (" << hit_point.x() << ", "
                  << hit_point.y() << ", " << hit_point.z() << ")" << std::endl;
        double distance = (hit_point - ray_origin).norm();
        std::cout << "  Distance: " << distance << "m" << std::endl;
    } else {
        std::cout << "  No obstacle hit within max range" << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 7. Save map to file
    // ============================================================
    std::cout << "--- Saving map ---" << std::endl;

    std::string filename_bt = "octomap_demo.bt";     // Binary format (compact)
    std::string filename_ot = "octomap_demo.ot";     // Full format (with probabilities)

    tree.writeBinary(filename_bt);
    std::cout << "  Saved binary map: " << filename_bt << std::endl;

    tree.write(filename_ot);
    std::cout << "  Saved full map: " << filename_ot << std::endl;

    // Get file sizes
    std::ifstream bt_file(filename_bt, std::ios::binary | std::ios::ate);
    std::ifstream ot_file(filename_ot, std::ios::binary | std::ios::ate);
    std::cout << "  Binary size: " << bt_file.tellg() / 1024.0 << " KB" << std::endl;
    std::cout << "  Full size: " << ot_file.tellg() / 1024.0 << " KB" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 8. Load map and verify
    // ============================================================
    std::cout << "--- Loading map ---" << std::endl;

    octomap::OcTree loaded_tree(filename_bt);
    std::cout << "  Loaded " << filename_bt << std::endl;
    std::cout << "  Leaf nodes: " << loaded_tree.getNumLeafNodes() << std::endl;
    std::cout << "  Memory: " << loaded_tree.memoryUsage() / 1024.0 << " KB" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 9. Probabilistic update demonstration
    // ============================================================
    std::cout << "--- Probabilistic update demo ---" << std::endl;

    octomap::OcTree prob_tree(0.1);
    octomap::point3d test_point(0.0, 0.0, 0.0);

    std::cout << "  Repeatedly observing point as occupied:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        prob_tree.updateNode(test_point, true);  // Observed as occupied
        octomap::OcTreeNode* node = prob_tree.search(test_point);
        if (node) {
            std::cout << "    Update " << (i+1) << ": occupancy = "
                      << node->getOccupancy() << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "  Now observing as free:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        prob_tree.updateNode(test_point, false);  // Observed as free
        octomap::OcTreeNode* node = prob_tree.search(test_point);
        if (node) {
            std::cout << "    Update " << (i+1) << ": occupancy = "
                      << node->getOccupancy() << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "=== OctoMap Demo Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Key takeaways:" << std::endl;
    std::cout << "  - OctoMap provides probabilistic occupancy mapping" << std::endl;
    std::cout << "  - Ray casting marks free space between sensor and obstacle" << std::endl;
    std::cout << "  - Probabilities converge with repeated observations" << std::endl;
    std::cout << "  - Multi-resolution queries enable coarse-to-fine planning" << std::endl;
    std::cout << "  - Compact binary format for storage and transmission" << std::endl;

    return 0;
}
