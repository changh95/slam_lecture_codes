/**
 * OctoMap Navigation Demo: Path Planning with Ray Casting
 *
 * This example demonstrates how to use OctoMap for navigation:
 * 1. Loading an OctoMap from file or creating a test environment
 * 2. Collision checking using ray casting
 * 3. Path validation (checking if a path is collision-free)
 * 4. Robot footprint collision checking
 * 5. Simple path planning concepts
 *
 * These techniques are fundamental for autonomous navigation
 * in robotics and SLAM applications.
 */

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

/**
 * Create a test environment with walls and obstacles
 */
void createTestEnvironment(octomap::OcTree& tree) {
    std::cout << "Creating test environment..." << std::endl;

    // Create walls (boundary of 10m x 10m room)
    double wall_height = 2.0;
    double resolution = tree.getResolution();

    // North wall (y = 5)
    for (double x = -5.0; x <= 5.0; x += resolution) {
        for (double z = 0.0; z <= wall_height; z += resolution) {
            tree.updateNode(octomap::point3d(x, 5.0, z), true);
        }
    }

    // South wall (y = -5)
    for (double x = -5.0; x <= 5.0; x += resolution) {
        for (double z = 0.0; z <= wall_height; z += resolution) {
            tree.updateNode(octomap::point3d(x, -5.0, z), true);
        }
    }

    // East wall (x = 5)
    for (double y = -5.0; y <= 5.0; y += resolution) {
        for (double z = 0.0; z <= wall_height; z += resolution) {
            tree.updateNode(octomap::point3d(5.0, y, z), true);
        }
    }

    // West wall (x = -5)
    for (double y = -5.0; y <= 5.0; y += resolution) {
        for (double z = 0.0; z <= wall_height; z += resolution) {
            tree.updateNode(octomap::point3d(-5.0, y, z), true);
        }
    }

    // Add some obstacles
    // Obstacle 1: Box at (2, 2)
    for (double x = 1.5; x <= 2.5; x += resolution) {
        for (double y = 1.5; y <= 2.5; y += resolution) {
            for (double z = 0.0; z <= 1.5; z += resolution) {
                tree.updateNode(octomap::point3d(x, y, z), true);
            }
        }
    }

    // Obstacle 2: Pillar at (-2, 0)
    for (double x = -2.3; x <= -1.7; x += resolution) {
        for (double y = -0.3; y <= 0.3; y += resolution) {
            for (double z = 0.0; z <= 2.0; z += resolution) {
                tree.updateNode(octomap::point3d(x, y, z), true);
            }
        }
    }

    // Obstacle 3: L-shaped barrier
    for (double x = 0.0; x <= 2.0; x += resolution) {
        for (double z = 0.0; z <= 1.0; z += resolution) {
            tree.updateNode(octomap::point3d(x, -2.0, z), true);
        }
    }
    for (double y = -2.0; y <= 0.0; y += resolution) {
        for (double z = 0.0; z <= 1.0; z += resolution) {
            tree.updateNode(octomap::point3d(2.0, y, z), true);
        }
    }

    // Ground plane
    for (double x = -5.0; x <= 5.0; x += resolution * 2) {
        for (double y = -5.0; y <= 5.0; y += resolution * 2) {
            tree.updateNode(octomap::point3d(x, y, -0.1), true);
        }
    }

    tree.updateInnerOccupancy();

    std::cout << "  Created room with " << tree.getNumLeafNodes() << " voxels" << std::endl;
}

/**
 * Check if a single point collides with the map
 */
bool isCollision(const octomap::OcTree& tree, const octomap::point3d& point) {
    octomap::OcTreeNode* node = tree.search(point);
    return (node != nullptr && tree.isNodeOccupied(node));
}

/**
 * Check if a straight-line path is collision-free using ray casting
 */
bool isPathFree(const octomap::OcTree& tree,
                const octomap::point3d& start,
                const octomap::point3d& end) {
    octomap::KeyRay ray;
    tree.computeRayKeys(start, end, ray);

    for (const auto& key : ray) {
        octomap::OcTreeNode* node = tree.search(key);
        if (node != nullptr && tree.isNodeOccupied(node)) {
            return false;  // Collision detected
        }
    }
    return true;  // Path is free
}

/**
 * Check collision for a robot footprint (cylindrical)
 */
bool checkRobotCollision(const octomap::OcTree& tree,
                         double x, double y, double z,
                         double robot_radius, double robot_height,
                         int num_samples = 8) {
    // Check center
    if (isCollision(tree, octomap::point3d(x, y, z))) {
        return true;
    }

    // Check points on robot footprint
    for (int i = 0; i < num_samples; ++i) {
        double angle = 2.0 * M_PI * i / num_samples;
        double dx = robot_radius * std::cos(angle);
        double dy = robot_radius * std::sin(angle);

        // Check at multiple heights
        for (double dz = 0.0; dz <= robot_height; dz += tree.getResolution()) {
            if (isCollision(tree, octomap::point3d(x + dx, y + dy, z + dz))) {
                return true;
            }
        }
    }

    return false;
}

/**
 * Simple 3D point structure
 */
struct Point3D {
    double x, y, z;

    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double distanceTo(const Point3D& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

/**
 * Simple path validation
 */
bool validatePath(const octomap::OcTree& tree,
                  const std::vector<Point3D>& path,
                  double robot_radius,
                  double robot_height) {
    if (path.empty()) return false;

    // Check each waypoint
    for (size_t i = 0; i < path.size(); ++i) {
        if (checkRobotCollision(tree, path[i].x, path[i].y, path[i].z,
                                robot_radius, robot_height)) {
            std::cout << "  Collision at waypoint " << i << ": ("
                      << path[i].x << ", " << path[i].y << ", " << path[i].z << ")" << std::endl;
            return false;
        }
    }

    // Check path segments
    for (size_t i = 0; i < path.size() - 1; ++i) {
        octomap::point3d start(path[i].x, path[i].y, path[i].z);
        octomap::point3d end(path[i+1].x, path[i+1].y, path[i+1].z);

        if (!isPathFree(tree, start, end)) {
            std::cout << "  Collision on segment " << i << " -> " << (i+1) << std::endl;
            return false;
        }
    }

    return true;
}

/**
 * Find nearest free point (simple version)
 */
octomap::point3d findNearestFreePoint(const octomap::OcTree& tree,
                                       const octomap::point3d& start,
                                       double search_radius = 2.0) {
    double resolution = tree.getResolution();

    for (double r = resolution; r <= search_radius; r += resolution) {
        for (double theta = 0; theta < 2 * M_PI; theta += 0.5) {
            for (double phi = 0; phi < M_PI; phi += 0.5) {
                double x = start.x() + r * std::sin(phi) * std::cos(theta);
                double y = start.y() + r * std::sin(phi) * std::sin(theta);
                double z = start.z() + r * std::cos(phi);

                octomap::point3d candidate(x, y, z);
                if (!isCollision(tree, candidate)) {
                    return candidate;
                }
            }
        }
    }

    return start;  // Return original if nothing found
}

int main(int argc, char* argv[]) {
    std::cout << "=== OctoMap Navigation Demo ===" << std::endl;
    std::cout << std::endl;

    // Create or load OctoMap
    double resolution = 0.1;  // 10cm
    octomap::OcTree tree(resolution);

    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Loading map from: " << filename << std::endl;
        tree.readBinary(filename);
    } else {
        std::cout << "No map file provided. Creating test environment..." << std::endl;
        createTestEnvironment(tree);
    }

    std::cout << "Map loaded:" << std::endl;
    std::cout << "  Resolution: " << tree.getResolution() << "m" << std::endl;
    std::cout << "  Leaf nodes: " << tree.getNumLeafNodes() << std::endl;
    std::cout << std::endl;

    // Robot parameters
    double robot_radius = 0.3;  // 30cm radius
    double robot_height = 0.5;  // 50cm height

    std::cout << "Robot parameters:" << std::endl;
    std::cout << "  Radius: " << robot_radius << "m" << std::endl;
    std::cout << "  Height: " << robot_height << "m" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 1. Single point collision checking
    // ============================================================
    std::cout << "--- 1. Point Collision Checking ---" << std::endl;

    std::vector<octomap::point3d> test_points = {
        octomap::point3d(0.0, 0.0, 0.5),   // Should be free (center of room)
        octomap::point3d(2.0, 2.0, 0.5),   // Should be occupied (obstacle)
        octomap::point3d(5.0, 0.0, 0.5),   // Should be occupied (wall)
        octomap::point3d(-3.0, -3.0, 0.5)  // Should be free
    };

    for (const auto& pt : test_points) {
        bool collision = isCollision(tree, pt);
        std::cout << "  Point (" << pt.x() << ", " << pt.y() << ", " << pt.z() << "): "
                  << (collision ? "COLLISION" : "FREE") << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 2. Ray casting for path checking
    // ============================================================
    std::cout << "--- 2. Ray Casting Path Check ---" << std::endl;

    struct PathTest {
        octomap::point3d start;
        octomap::point3d end;
        std::string description;
    };

    std::vector<PathTest> path_tests = {
        {{0.0, 0.0, 0.5}, {3.0, 3.0, 0.5}, "Center to NE corner (through obstacle)"},
        {{0.0, 0.0, 0.5}, {-3.0, -3.0, 0.5}, "Center to SW corner (free)"},
        {{-4.0, 0.0, 0.5}, {4.0, 0.0, 0.5}, "West to East (hits pillar)"},
        {{0.0, 3.0, 0.5}, {0.0, -3.0, 0.5}, "North to South (hits barrier)"}
    };

    for (const auto& test : path_tests) {
        auto start_time = std::chrono::high_resolution_clock::now();
        bool free = isPathFree(tree, test.start, test.end);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();

        std::cout << "  " << test.description << ": "
                  << (free ? "FREE" : "BLOCKED")
                  << " (" << duration << " us)" << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 3. Robot footprint collision checking
    // ============================================================
    std::cout << "--- 3. Robot Footprint Collision ---" << std::endl;

    std::vector<Point3D> robot_positions = {
        {0.0, 0.0, 0.0},
        {1.8, 2.0, 0.0},   // Near obstacle
        {-2.0, 0.0, 0.0},  // Near pillar
        {-3.0, -3.0, 0.0}  // Free area
    };

    for (const auto& pos : robot_positions) {
        bool collision = checkRobotCollision(tree, pos.x, pos.y, pos.z,
                                              robot_radius, robot_height);
        std::cout << "  Robot at (" << pos.x << ", " << pos.y << ", " << pos.z << "): "
                  << (collision ? "COLLISION" : "SAFE") << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 4. Path validation
    // ============================================================
    std::cout << "--- 4. Path Validation ---" << std::endl;

    // Test path 1: Should be valid
    std::vector<Point3D> valid_path = {
        {-3.0, -3.0, 0.0},
        {-3.0, 0.0, 0.0},
        {-3.0, 3.0, 0.0},
        {0.0, 3.0, 0.0}
    };

    std::cout << "Path 1 (should be valid):" << std::endl;
    for (const auto& wp : valid_path) {
        std::cout << "  -> (" << wp.x << ", " << wp.y << ", " << wp.z << ")" << std::endl;
    }
    bool valid1 = validatePath(tree, valid_path, robot_radius, robot_height);
    std::cout << "  Result: " << (valid1 ? "VALID" : "INVALID") << std::endl;
    std::cout << std::endl;

    // Test path 2: Should be invalid (goes through obstacle)
    std::vector<Point3D> invalid_path = {
        {0.0, 0.0, 0.0},
        {2.0, 2.0, 0.0},   // Obstacle location
        {4.0, 4.0, 0.0}
    };

    std::cout << "Path 2 (goes through obstacle):" << std::endl;
    for (const auto& wp : invalid_path) {
        std::cout << "  -> (" << wp.x << ", " << wp.y << ", " << wp.z << ")" << std::endl;
    }
    bool valid2 = validatePath(tree, invalid_path, robot_radius, robot_height);
    std::cout << "  Result: " << (valid2 ? "VALID" : "INVALID") << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 5. Ray casting performance
    // ============================================================
    std::cout << "--- 5. Ray Casting Performance ---" << std::endl;

    int num_rays = 1000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-4.0, 4.0);

    auto start_perf = std::chrono::high_resolution_clock::now();

    int blocked_count = 0;
    for (int i = 0; i < num_rays; ++i) {
        octomap::point3d start(dist(gen), dist(gen), 0.5);
        octomap::point3d end(dist(gen), dist(gen), 0.5);

        if (!isPathFree(tree, start, end)) {
            blocked_count++;
        }
    }

    auto end_perf = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_perf - start_perf).count();

    std::cout << "  Tested " << num_rays << " random rays" << std::endl;
    std::cout << "  Total time: " << total_time << " us" << std::endl;
    std::cout << "  Average per ray: " << total_time / num_rays << " us" << std::endl;
    std::cout << "  Blocked: " << blocked_count << " ("
              << (100.0 * blocked_count / num_rays) << "%)" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 6. Multi-resolution collision checking
    // ============================================================
    std::cout << "--- 6. Multi-Resolution Collision Check ---" << std::endl;

    octomap::point3d coarse_query(0.0, 0.0, 0.5);
    std::cout << "Checking point (" << coarse_query.x() << ", "
              << coarse_query.y() << ", " << coarse_query.z()
              << ") at different resolutions:" << std::endl;

    for (unsigned int depth = tree.getTreeDepth(); depth >= tree.getTreeDepth() - 3; --depth) {
        double level_resolution = tree.getResolution() *
            std::pow(2, tree.getTreeDepth() - depth);

        octomap::OcTreeNode* node = tree.search(coarse_query, depth);

        std::cout << "  Depth " << depth << " (res=" << level_resolution << "m): ";
        if (node) {
            if (tree.isNodeOccupied(node)) {
                std::cout << "OCCUPIED (prob=" << node->getOccupancy() << ")" << std::endl;
            } else {
                std::cout << "FREE (prob=" << node->getOccupancy() << ")" << std::endl;
            }
        } else {
            std::cout << "UNKNOWN" << std::endl;
        }
    }
    std::cout << std::endl;

    // ============================================================
    // 7. Nearest free point search
    // ============================================================
    std::cout << "--- 7. Nearest Free Point ---" << std::endl;

    octomap::point3d obstacle_point(2.0, 2.0, 0.5);  // Inside obstacle
    octomap::point3d nearest = findNearestFreePoint(tree, obstacle_point);

    std::cout << "  Start point (in obstacle): (" << obstacle_point.x() << ", "
              << obstacle_point.y() << ", " << obstacle_point.z() << ")" << std::endl;
    std::cout << "  Nearest free point: (" << nearest.x() << ", "
              << nearest.y() << ", " << nearest.z() << ")" << std::endl;
    std::cout << "  Distance: " << (nearest - obstacle_point).norm() << "m" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Navigation Demo Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Key takeaways:" << std::endl;
    std::cout << "  - Ray casting enables efficient path validation" << std::endl;
    std::cout << "  - Robot footprint checking prevents collisions" << std::endl;
    std::cout << "  - Multi-resolution queries enable coarse-to-fine planning" << std::endl;
    std::cout << "  - OctoMap is well-suited for navigation applications" << std::endl;

    return 0;
}
