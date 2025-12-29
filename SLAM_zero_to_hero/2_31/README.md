# Octree, Octomap, and Bonxai: Spatial Data Structures for Mapping

This tutorial covers spatial data structures for 3D mapping: Octree (hierarchical spatial subdivision), OctoMap (probabilistic occupancy mapping), and Bonxai (modern high-performance voxel mapping).

---

## Overview

Efficient 3D map representations are crucial for SLAM and robotics:

| Library | Type | Key Feature | Best For |
|---------|------|-------------|----------|
| **Octree** | Hierarchical tree | Adaptive resolution | Spatial queries |
| **OctoMap** | Probabilistic octree | Occupancy probability | Navigation |
| **Bonxai** | Hash-based voxels | High performance | Large-scale maps |

---

## 1. Octree

An octree recursively divides 3D space into 8 octants, providing adaptive resolution based on data density.

### Structure

```
         Root
        /    \
       /      \
    [...]    [...]
     /|\      /|\
    8 children per node

Each node divides space into 8 equal octants:
    +---+---+
   /   /   /|
  +---+---+ |
 /   /   /| +
+---+---+ |/|
|   |   | + |
+---+---+/| +
|   |   | |/
+---+---+ +
```

### Properties

- **Space complexity**: O(n) where n is number of occupied cells
- **Query complexity**: O(log n) for point queries
- **Adaptive**: Fine resolution where needed, coarse elsewhere

### PCL Octree Code

```cpp
#include <pcl/octree/octree_search.h>

// Create octree with resolution
float resolution = 0.1f;  // 10cm voxels
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);

// Build from point cloud
octree.setInputCloud(cloud);
octree.addPointsFromInputCloud();

// Voxel search: find points in same voxel
std::vector<int> indices;
pcl::PointXYZ query(1.0, 2.0, 3.0);
if (octree.voxelSearch(query, indices)) {
    std::cout << "Found " << indices.size() << " points in voxel" << std::endl;
}

// Nearest neighbor search
int k = 10;
std::vector<int> k_indices(k);
std::vector<float> k_distances(k);
octree.nearestKSearch(query, k, k_indices, k_distances);

// Radius search
float radius = 0.5f;
std::vector<int> radius_indices;
std::vector<float> radius_distances;
octree.radiusSearch(query, radius, radius_indices, radius_distances);
```

---

## 2. OctoMap

OctoMap extends octrees with probabilistic occupancy modeling.

### Key Concepts

**Occupancy Probability**:
- Each voxel stores log-odds of occupancy
- Updated via sensor model
- Handles uncertainty and sensor noise

**Log-odds Update**:
```
L(n|z_{1:t}) = L(n|z_{1:t-1}) + L(n|z_t)

where L(n) = log(P(n) / (1 - P(n)))
```

### Advantages

- Probabilistic: Handles uncertainty
- Multi-resolution: Query at any level
- Memory-efficient: Only stores observed space
- Update-able: Integrates new measurements

### Code Example

```cpp
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

// Create octree with 10cm resolution
octomap::OcTree tree(0.1);

// Insert point cloud
for (const auto& p : cloud->points) {
    tree.updateNode(octomap::point3d(p.x, p.y, p.z), true);  // Occupied
}

// Insert free space (ray casting)
octomap::point3d sensor_origin(0, 0, 0);
for (const auto& p : cloud->points) {
    octomap::point3d end(p.x, p.y, p.z);
    tree.insertRay(sensor_origin, end);  // Marks ray as free, end as occupied
}

// Update internal structure
tree.updateInnerOccupancy();

// Query occupancy
octomap::point3d query(1.0, 2.0, 3.0);
octomap::OcTreeNode* node = tree.search(query);
if (node != nullptr) {
    double occupancy = node->getOccupancy();
    std::cout << "Occupancy: " << occupancy << std::endl;

    if (tree.isNodeOccupied(node)) {
        std::cout << "Node is occupied" << std::endl;
    }
}

// Save/load
tree.writeBinary("map.bt");
// tree.readBinary("map.bt");
```

### Sensor Model Integration

```cpp
// Set sensor model parameters
tree.setProbHit(0.7);     // P(occupied | hit)
tree.setProbMiss(0.4);    // P(occupied | miss)
tree.setClampingThresMin(0.12);  // Min probability
tree.setClampingThresMax(0.97);  // Max probability

// Insert scan with sensor pose
octomap::Pointcloud scan;
for (const auto& p : lidar_scan) {
    scan.push_back(p.x, p.y, p.z);
}

octomap::pose6d sensor_pose(tx, ty, tz, roll, pitch, yaw);
tree.insertPointCloud(scan, sensor_origin, sensor_pose);
```

### Ray Casting for Path Planning

```cpp
// Check if path is collision-free
bool isPathFree(const octomap::OcTree& tree,
                const octomap::point3d& start,
                const octomap::point3d& end) {

    octomap::KeyRay ray;
    tree.computeRayKeys(start, end, ray);

    for (const auto& key : ray) {
        octomap::OcTreeNode* node = tree.search(key);
        if (node != nullptr && tree.isNodeOccupied(node)) {
            return false;  // Collision
        }
    }
    return true;  // Free path
}
```

### Multi-Resolution Queries

```cpp
// Query at different depths
for (int depth = 0; depth <= tree.getTreeDepth(); depth++) {
    double resolution = tree.getResolution() * pow(2, tree.getTreeDepth() - depth);
    std::cout << "Depth " << depth << ": resolution = " << resolution << "m" << std::endl;
}

// Get occupied voxels at specific depth
int depth = 14;  // Lower depth = coarser
for (auto it = tree.begin_leafs(); it != tree.end_leafs(); ++it) {
    if (it.getDepth() == depth && tree.isNodeOccupied(*it)) {
        octomap::point3d coord = it.getCoordinate();
        double size = it.getSize();
        std::cout << "Occupied voxel at " << coord << " size " << size << std::endl;
    }
}
```

---

## 3. Bonxai

Bonxai is a modern, high-performance voxel library using hash-based storage.

### Key Features

- **10-100x faster** than OctoMap for many operations
- **Header-only**: Easy integration
- **Memory efficient**: Hash map storage
- **No tree overhead**: Direct voxel access

### Code Example

```cpp
#include "bonxai/bonxai.hpp"

// Create voxel grid with 10cm resolution
Bonxai::VoxelGrid<float> grid(0.1);

// Set voxel values
for (const auto& p : cloud->points) {
    auto accessor = grid.createAccessor();
    Bonxai::CoordT coord = grid.posToCoord({p.x, p.y, p.z});
    accessor.setValue(coord, 1.0f);  // Set occupancy
}

// Query voxel
Bonxai::CoordT query_coord = grid.posToCoord({1.0, 2.0, 3.0});
auto accessor = grid.createAccessor();
float* value = accessor.value(query_coord);
if (value != nullptr) {
    std::cout << "Voxel value: " << *value << std::endl;
}

// Iterate over all voxels
grid.forEachCell([](const Bonxai::CoordT& coord, float& value) {
    // Process each voxel
});
```

### Probabilistic Updates (Custom)

```cpp
// Implement log-odds update similar to OctoMap
class ProbabilisticBonxai {
public:
    ProbabilisticBonxai(double resolution)
        : grid_(resolution), prob_hit_(0.7), prob_miss_(0.4) {}

    void insertRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& end) {
        // Mark end point as occupied
        updateVoxel(end, true);

        // Ray march and mark as free
        Eigen::Vector3d direction = (end - origin).normalized();
        double length = (end - origin).norm();

        for (double t = grid_.resolution(); t < length; t += grid_.resolution()) {
            Eigen::Vector3d point = origin + t * direction;
            updateVoxel(point, false);
        }
    }

    void updateVoxel(const Eigen::Vector3d& point, bool hit) {
        auto accessor = grid_.createAccessor();
        Bonxai::CoordT coord = grid_.posToCoord({point.x(), point.y(), point.z()});

        float* log_odds = accessor.value(coord);
        if (log_odds == nullptr) {
            accessor.setValue(coord, 0.0f);
            log_odds = accessor.value(coord);
        }

        float update = hit ? logOdds(prob_hit_) : logOdds(prob_miss_);
        *log_odds = std::clamp(*log_odds + update, -2.0f, 3.5f);
    }

private:
    float logOdds(double p) { return std::log(p / (1.0 - p)); }

    Bonxai::VoxelGrid<float> grid_;
    double prob_hit_, prob_miss_;
};
```

---

## Comparison

### Performance (1M points)

| Operation | Octree (PCL) | OctoMap | Bonxai |
|-----------|--------------|---------|--------|
| Insertion | 50ms | 200ms | 20ms |
| Query | 0.1μs | 0.5μs | 0.05μs |
| Memory | Medium | Low | Low |

### Feature Comparison

| Feature | Octree | OctoMap | Bonxai |
|---------|--------|---------|--------|
| Probabilistic | No | Yes | Custom |
| Multi-resolution | Yes | Yes | No |
| Ray casting | Yes | Yes | Custom |
| Serialization | Yes | Yes | Yes |
| Header-only | No | No | Yes |

---

## Project Structure

```
2_31/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── 3rdparty/
│   └── bonxai/                 # Header-only Bonxai
├── data/
│   └── sample_clouds/
└── examples/
    ├── octree_demo.cpp         # PCL Octree usage
    ├── octomap_demo.cpp        # OctoMap mapping
    ├── octomap_navigation.cpp  # Path planning with OctoMap
    ├── bonxai_demo.cpp         # Bonxai voxel grid
    └── comparison.cpp          # Compare all methods
```

---

## How to Build

### Dependencies
- PCL 1.12+
- OctoMap 1.9+
- Bonxai (header-only, included)

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_31
```

---

## How to Run

```bash
# PCL Octree demo
./build/octree_demo cloud.pcd

# OctoMap mapping
./build/octomap_demo /path/to/lidar/sequence/

# OctoMap navigation
./build/octomap_navigation map.bt start_x start_y start_z end_x end_y end_z

# Bonxai demo
./build/bonxai_demo cloud.pcd

# Compare methods
./build/comparison cloud.pcd
```

---

## Integration with SLAM

### OctoMap Server (ROS)

```cpp
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

class OctomapMapper {
public:
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
        // Convert to PCL
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // Insert into OctoMap
        octomap::Pointcloud scan;
        for (const auto& p : cloud.points) {
            scan.push_back(p.x, p.y, p.z);
        }

        // Get sensor origin from TF
        octomap::point3d origin(sensor_pose_.x, sensor_pose_.y, sensor_pose_.z);
        tree_.insertPointCloud(scan, origin);

        // Publish map
        octomap_msgs::Octomap msg_out;
        octomap_msgs::binaryMapToMsg(tree_, msg_out);
        map_pub_.publish(msg_out);
    }

private:
    octomap::OcTree tree_{0.1};
    ros::Publisher map_pub_;
    geometry_msgs::Pose sensor_pose_;
};
```

### Using Maps for Navigation

```cpp
// Check if robot footprint collides with map
bool checkCollision(const octomap::OcTree& tree,
                    double x, double y, double z,
                    double robot_radius) {
    // Sample points on robot footprint
    for (double dx = -robot_radius; dx <= robot_radius; dx += 0.1) {
        for (double dy = -robot_radius; dy <= robot_radius; dy += 0.1) {
            if (dx*dx + dy*dy > robot_radius*robot_radius) continue;

            octomap::point3d query(x + dx, y + dy, z);
            octomap::OcTreeNode* node = tree.search(query);

            if (node && tree.isNodeOccupied(node)) {
                return true;  // Collision
            }
        }
    }
    return false;  // No collision
}
```

---

## Best Practices

1. **Resolution Selection**:
   - Indoor: 1-5cm
   - Outdoor: 10-50cm
   - Large-scale: 50cm-1m

2. **Memory Management**:
   - Use OctoMap for long-term storage
   - Use Bonxai for real-time processing

3. **Update Strategy**:
   - Batch updates when possible
   - Prune tree periodically in OctoMap

4. **Multi-resolution**:
   - Query at coarse level for planning
   - Fine level for precise collision checking

---

## References

- Hornung et al., "OctoMap: An Efficient Probabilistic 3D Mapping Framework", Autonomous Robots 2013
- [OctoMap GitHub](https://github.com/OctoMap/octomap)
- [Bonxai GitHub](https://github.com/facontidavide/Bonxai)
- [PCL Octree Tutorial](https://pcl.readthedocs.io/projects/tutorials/en/latest/octree.html)
- Meagher, "Geometric Modeling Using Octree Encoding", Computer Graphics 1982
