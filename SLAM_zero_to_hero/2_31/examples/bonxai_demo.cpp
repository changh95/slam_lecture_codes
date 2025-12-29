/**
 * Bonxai Demo: High-Performance Voxel Grid
 *
 * This example demonstrates how to use Bonxai for:
 * 1. Creating a voxel grid with custom resolution
 * 2. Setting and querying voxel values
 * 3. Iterating over active voxels
 * 4. Implementing probabilistic occupancy (like OctoMap)
 * 5. Performance comparison with traditional methods
 *
 * Bonxai is a header-only, high-performance alternative to OctoMap.
 * It uses hash-based storage for O(1) voxel access and is typically
 * 10-100x faster than OctoMap for many operations.
 */

#include "bonxai/bonxai.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

/**
 * Generate a synthetic point cloud for testing
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr generateSyntheticCloud(int num_points = 10000) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_y(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_z(-1.0f, 4.0f);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointXYZ pt;
        pt.x = dist_x(gen);
        pt.y = dist_y(gen);
        pt.z = dist_z(gen);
        cloud->push_back(pt);
    }

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

/**
 * Probabilistic voxel data structure
 * Stores log-odds for occupancy probability
 */
struct ProbabilisticVoxel {
    float log_odds;

    ProbabilisticVoxel() : log_odds(0.0f) {}

    float getOccupancy() const {
        return 1.0f / (1.0f + std::exp(-log_odds));
    }

    bool isOccupied(float threshold = 0.5f) const {
        return getOccupancy() > threshold;
    }
};

/**
 * Wrapper class for probabilistic Bonxai grid
 * Implements OctoMap-like probability updates
 */
class ProbabilisticBonxai {
public:
    ProbabilisticBonxai(double resolution)
        : grid_(resolution), prob_hit_(0.7), prob_miss_(0.4),
          log_odds_hit_(logOdds(prob_hit_)), log_odds_miss_(logOdds(prob_miss_)),
          clamp_min_(-2.0f), clamp_max_(3.5f) {}

    void updateOccupied(double x, double y, double z) {
        auto accessor = grid_.createAccessor();
        auto coord = grid_.posToCoord(x, y, z);

        ProbabilisticVoxel* voxel = accessor.value(coord, true);
        if (voxel) {
            voxel->log_odds = std::clamp(
                voxel->log_odds + log_odds_hit_, clamp_min_, clamp_max_);
        }
    }

    void updateFree(double x, double y, double z) {
        auto accessor = grid_.createAccessor();
        auto coord = grid_.posToCoord(x, y, z);

        ProbabilisticVoxel* voxel = accessor.value(coord, true);
        if (voxel) {
            voxel->log_odds = std::clamp(
                voxel->log_odds + log_odds_miss_, clamp_min_, clamp_max_);
        }
    }

    void insertRay(double ox, double oy, double oz,
                   double ex, double ey, double ez) {
        // Mark endpoint as occupied
        updateOccupied(ex, ey, ez);

        // March along ray and mark as free
        double dx = ex - ox;
        double dy = ey - oy;
        double dz = ez - oz;
        double length = std::sqrt(dx*dx + dy*dy + dz*dz);

        if (length < grid_.resolution()) return;

        dx /= length;
        dy /= length;
        dz /= length;

        for (double t = 0; t < length - grid_.resolution(); t += grid_.resolution()) {
            updateFree(ox + t*dx, oy + t*dy, oz + t*dz);
        }
    }

    float getOccupancy(double x, double y, double z) const {
        auto accessor = grid_.createConstAccessor();
        auto coord = grid_.posToCoord(x, y, z);

        const ProbabilisticVoxel* voxel = accessor.value(coord);
        if (voxel) {
            return voxel->getOccupancy();
        }
        return 0.5f;  // Unknown
    }

    bool isOccupied(double x, double y, double z) const {
        return getOccupancy(x, y, z) > 0.5f;
    }

    size_t countActiveVoxels() const {
        size_t count = 0;
        grid_.forEachCell([&count](const ProbabilisticVoxel&, const Bonxai::CoordT&) {
            count++;
        });
        return count;
    }

    double resolution() const { return grid_.resolution(); }

    Bonxai::VoxelGrid<ProbabilisticVoxel>& grid() { return grid_; }
    const Bonxai::VoxelGrid<ProbabilisticVoxel>& grid() const { return grid_; }

private:
    static float logOdds(double p) {
        return static_cast<float>(std::log(p / (1.0 - p)));
    }

    Bonxai::VoxelGrid<ProbabilisticVoxel> grid_;
    double prob_hit_, prob_miss_;
    float log_odds_hit_, log_odds_miss_;
    float clamp_min_, clamp_max_;
};

int main(int argc, char* argv[]) {
    std::cout << "=== Bonxai High-Performance Voxel Grid Demo ===" << std::endl;
    std::cout << std::endl;

    // Load or generate point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    if (argc > 1) {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1) {
            std::cerr << "Error loading " << argv[1] << std::endl;
            return -1;
        }
        std::cout << "Loaded " << cloud->size() << " points from " << argv[1] << std::endl;
    } else {
        std::cout << "No input file. Generating synthetic cloud..." << std::endl;
        cloud = generateSyntheticCloud(50000);
        std::cout << "Generated " << cloud->size() << " points" << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 1. Basic VoxelGrid Usage
    // ============================================================
    std::cout << "--- 1. Basic VoxelGrid Usage ---" << std::endl;

    double resolution = 0.1;  // 10cm voxels
    Bonxai::VoxelGrid<float> grid(resolution);

    std::cout << "Created VoxelGrid with resolution: " << resolution << "m" << std::endl;

    // Insert points
    auto start_insert = std::chrono::high_resolution_clock::now();

    {
        auto accessor = grid.createAccessor();
        for (const auto& pt : cloud->points) {
            auto coord = grid.posToCoord(pt.x, pt.y, pt.z);
            accessor.setValue(coord, 1.0f);  // Mark as occupied
        }
    }

    auto end_insert = std::chrono::high_resolution_clock::now();
    auto insert_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_insert - start_insert).count();

    std::cout << "Inserted " << cloud->size() << " points" << std::endl;
    std::cout << "  Insert time: " << insert_time << " us" << std::endl;
    std::cout << "  Time per point: " << insert_time * 1000.0 / cloud->size() << " ns" << std::endl;

    // Count active voxels
    size_t num_voxels = 0;
    grid.forEachCell([&num_voxels](const float&, const Bonxai::CoordT&) {
        num_voxels++;
    });
    std::cout << "  Active voxels: " << num_voxels << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 2. Voxel Queries
    // ============================================================
    std::cout << "--- 2. Voxel Queries ---" << std::endl;

    // Random query points
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> query_dist(-10.0, 10.0);

    int num_queries = 100000;
    int found_count = 0;

    auto start_query = std::chrono::high_resolution_clock::now();

    {
        auto accessor = grid.createConstAccessor();
        for (int i = 0; i < num_queries; ++i) {
            double x = query_dist(gen);
            double y = query_dist(gen);
            double z = query_dist(gen);

            auto coord = grid.posToCoord(x, y, z);
            const float* value = accessor.value(coord);
            if (value != nullptr) {
                found_count++;
            }
        }
    }

    auto end_query = std::chrono::high_resolution_clock::now();
    auto query_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_query - start_query).count();

    std::cout << "Performed " << num_queries << " random queries" << std::endl;
    std::cout << "  Query time: " << query_time << " us" << std::endl;
    std::cout << "  Time per query: " << query_time * 1000.0 / num_queries << " ns" << std::endl;
    std::cout << "  Hit rate: " << 100.0 * found_count / num_queries << "%" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 3. Iterate Over Voxels
    // ============================================================
    std::cout << "--- 3. Iterate Over Voxels ---" << std::endl;

    Bonxai::Point3D min_bound = {1e9, 1e9, 1e9};
    Bonxai::Point3D max_bound = {-1e9, -1e9, -1e9};
    float sum = 0.0f;

    auto start_iter = std::chrono::high_resolution_clock::now();

    grid.forEachCell([&](const float& value, const Bonxai::CoordT& coord) {
        auto pos = grid.coordToPos(coord);
        min_bound.x = std::min(min_bound.x, pos.x);
        min_bound.y = std::min(min_bound.y, pos.y);
        min_bound.z = std::min(min_bound.z, pos.z);
        max_bound.x = std::max(max_bound.x, pos.x);
        max_bound.y = std::max(max_bound.y, pos.y);
        max_bound.z = std::max(max_bound.z, pos.z);
        sum += value;
    });

    auto end_iter = std::chrono::high_resolution_clock::now();
    auto iter_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_iter - start_iter).count();

    std::cout << "Iterated over " << num_voxels << " voxels" << std::endl;
    std::cout << "  Iteration time: " << iter_time << " us" << std::endl;
    std::cout << "  Bounds: [" << min_bound.x << ", " << min_bound.y << ", " << min_bound.z
              << "] to [" << max_bound.x << ", " << max_bound.y << ", " << max_bound.z << "]" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 4. Custom Data Types
    // ============================================================
    std::cout << "--- 4. Custom Data Types ---" << std::endl;

    struct ColorVoxel {
        uint8_t r, g, b;
        uint16_t hit_count;
    };

    Bonxai::VoxelGrid<ColorVoxel> color_grid(0.1);

    {
        auto accessor = color_grid.createAccessor();

        // Add some colored voxels
        for (int i = 0; i < 1000; ++i) {
            auto coord = color_grid.posToCoord(i * 0.1, 0, 0);
            ColorVoxel voxel{
                static_cast<uint8_t>(i % 256),
                static_cast<uint8_t>((i * 2) % 256),
                static_cast<uint8_t>((i * 3) % 256),
                1
            };
            accessor.setValue(coord, voxel);
        }
    }

    size_t color_count = 0;
    color_grid.forEachCell([&color_count](const ColorVoxel&, const Bonxai::CoordT&) {
        color_count++;
    });

    std::cout << "Created " << color_count << " color voxels" << std::endl;
    std::cout << "  Voxel size: " << sizeof(ColorVoxel) << " bytes" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // 5. Probabilistic Occupancy Grid
    // ============================================================
    std::cout << "--- 5. Probabilistic Occupancy Grid ---" << std::endl;

    ProbabilisticBonxai prob_grid(0.1);

    // Simulate LiDAR scans
    Bonxai::Point3D sensor_origin = {0, 0, 0.5};
    int num_rays = 1000;

    auto start_prob = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_rays; ++i) {
        double azimuth = 2.0 * M_PI * i / num_rays;
        double elevation = -0.1 + 0.2 * std::sin(azimuth * 3);

        double range = 5.0 + 2.0 * std::sin(azimuth * 5);

        double ex = sensor_origin.x + range * std::cos(azimuth) * std::cos(elevation);
        double ey = sensor_origin.y + range * std::sin(azimuth) * std::cos(elevation);
        double ez = sensor_origin.z + range * std::sin(elevation);

        prob_grid.insertRay(sensor_origin.x, sensor_origin.y, sensor_origin.z,
                            ex, ey, ez);
    }

    auto end_prob = std::chrono::high_resolution_clock::now();
    auto prob_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_prob - start_prob).count();

    std::cout << "Inserted " << num_rays << " rays" << std::endl;
    std::cout << "  Time: " << prob_time << " ms" << std::endl;
    std::cout << "  Active voxels: " << prob_grid.countActiveVoxels() << std::endl;

    // Query some points
    std::cout << std::endl;
    std::cout << "Occupancy queries:" << std::endl;

    std::vector<std::tuple<double, double, double>> queries = {
        {0.0, 0.0, 0.5},   // Sensor location (should be free)
        {5.0, 0.0, 0.5},   // Near a hit point (should be occupied)
        {2.5, 0.0, 0.5},   // Midway (should be free)
        {10.0, 10.0, 0.5}  // Never observed
    };

    for (const auto& [x, y, z] : queries) {
        float occ = prob_grid.getOccupancy(x, y, z);
        std::cout << "  (" << x << ", " << y << ", " << z << "): "
                  << "prob=" << occ
                  << (occ > 0.5f ? " [OCCUPIED]" : (occ < 0.5f ? " [FREE]" : " [UNKNOWN]"))
                  << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 6. Resolution Comparison
    // ============================================================
    std::cout << "--- 6. Resolution Comparison ---" << std::endl;

    std::vector<double> resolutions = {0.05, 0.1, 0.2, 0.5, 1.0};

    std::cout << "Resolution | Voxels   | Insert(us) | Query(ns)" << std::endl;
    std::cout << "-----------|----------|------------|----------" << std::endl;

    for (double res : resolutions) {
        Bonxai::VoxelGrid<float> test_grid(res);

        // Insert
        auto t1 = std::chrono::high_resolution_clock::now();
        {
            auto accessor = test_grid.createAccessor();
            for (const auto& pt : cloud->points) {
                auto coord = test_grid.posToCoord(pt.x, pt.y, pt.z);
                accessor.setValue(coord, 1.0f);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();

        // Count voxels
        size_t voxel_count = 0;
        test_grid.forEachCell([&voxel_count](const float&, const Bonxai::CoordT&) {
            voxel_count++;
        });

        // Query
        auto t3 = std::chrono::high_resolution_clock::now();
        {
            auto accessor = test_grid.createConstAccessor();
            for (int i = 0; i < 10000; ++i) {
                auto coord = test_grid.posToCoord(query_dist(gen), query_dist(gen), query_dist(gen));
                accessor.value(coord);
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();

        auto insert_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto query_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 10000;

        printf("  %7.2fm | %8lu | %10ld | %9ld\n",
               res, static_cast<unsigned long>(voxel_count), insert_us, query_ns);
    }
    std::cout << std::endl;

    std::cout << "=== Bonxai Demo Complete ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Key takeaways:" << std::endl;
    std::cout << "  - Bonxai provides O(1) average voxel access" << std::endl;
    std::cout << "  - Header-only library, easy to integrate" << std::endl;
    std::cout << "  - Supports custom data types (not just float)" << std::endl;
    std::cout << "  - 10-100x faster than OctoMap for many operations" << std::endl;
    std::cout << "  - No built-in multi-resolution (unlike OctoMap)" << std::endl;
    std::cout << "  - Best for: real-time applications, large-scale maps" << std::endl;

    return 0;
}
