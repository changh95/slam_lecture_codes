# Introduction to Point Cloud Processing using PCL

This tutorial covers the fundamentals of point cloud processing and pre-processing techniques using the Point Cloud Library (PCL). These concepts are essential for LiDAR-based SLAM and 3D perception systems.

---

## Overview

A **point cloud** is a set of 3D points representing a surface or object. Each point typically contains:
- **Position**: (x, y, z) coordinates
- **Intensity**: Return signal strength (LiDAR)
- **Color**: RGB values (RGB-D cameras)
- **Normal**: Surface normal vector

### Applications in SLAM

| Application | Description |
|-------------|-------------|
| **LiDAR SLAM** | 3D mapping with laser scanners |
| **RGB-D SLAM** | Depth camera-based mapping |
| **Autonomous Driving** | Obstacle detection, localization |
| **Robotics** | Navigation, manipulation |

---

## Point Cloud Sources

| Sensor | Points/Frame | Range | Typical Use |
|--------|-------------|-------|-------------|
| Velodyne VLP-16 | ~30,000 | 100m | Mobile robots |
| Velodyne HDL-64E | ~130,000 | 120m | Autonomous driving |
| Ouster OS1-128 | ~262,000 | 120m | High-res mapping |
| Intel RealSense | ~300,000 | 10m | Indoor robotics |
| Kinect v2 | ~217,000 | 4.5m | Indoor, research |

---

## Project Structure

```
2_27/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_clouds/           # Sample point cloud files
└── examples/
    ├── basic_io.cpp             # Read/write point clouds
    ├── visualization.cpp        # PCL visualizer
    ├── filtering.cpp            # Downsampling, outlier removal
    ├── normal_estimation.cpp    # Compute surface normals
    └── segmentation.cpp         # Ground plane, clustering
```

---

## How to Build

### Dependencies
- PCL 1.12+
- Eigen3

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_27
```

---

## How to Run

### Local

```bash
# Basic I/O
./build/basic_io input.pcd output.pcd

# Visualization
./build/visualization cloud.pcd

# Filtering
./build/filtering cloud.pcd --voxel 0.1 --sor 1.0

# Normal estimation
./build/normal_estimation cloud.pcd --radius 0.1

# Segmentation
./build/segmentation cloud.pcd --ground --cluster
```

### Docker

```bash
docker run -it --rm \
    -v $(pwd)/data:/data \
    slam_zero_to_hero:2_27
```

---

## Key Concepts

### Point Types in PCL

```cpp
// Basic 3D point
pcl::PointXYZ point;
point.x = 1.0f;
point.y = 2.0f;
point.z = 3.0f;

// Point with intensity (LiDAR)
pcl::PointXYZI point_i;
point_i.intensity = 100.0f;

// Point with color (RGB-D)
pcl::PointXYZRGB point_rgb;
point_rgb.r = 255; point_rgb.g = 0; point_rgb.b = 0;

// Point with normal
pcl::PointNormal point_n;
point_n.normal_x = 0; point_n.normal_y = 0; point_n.normal_z = 1;
```

### Point Cloud Container

```cpp
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Create point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZ>);

// Set cloud properties
cloud->width = 640;
cloud->height = 480;  // Organized cloud (from depth camera)
cloud->is_dense = false;

// Or unorganized cloud
cloud->width = 10000;
cloud->height = 1;

// Add points
cloud->points.resize(cloud->width * cloud->height);
for (auto& p : cloud->points) {
    p.x = rand() / (float)RAND_MAX;
    p.y = rand() / (float)RAND_MAX;
    p.z = rand() / (float)RAND_MAX;
}
```

---

## Code Examples

### 1. Reading and Writing Point Clouds

```cpp
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// Read PCD file
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZ>);
if (pcl::io::loadPCDFile<pcl::PointXYZ>("input.pcd", *cloud) == -1) {
    PCL_ERROR("Couldn't read file\n");
    return -1;
}
std::cout << "Loaded " << cloud->points.size() << " points" << std::endl;

// Write PCD file
pcl::io::savePCDFileASCII("output.pcd", *cloud);
// Or binary for efficiency
pcl::io::savePCDFileBinary("output_binary.pcd", *cloud);

// Read/write PLY files
pcl::io::loadPLYFile("input.ply", *cloud);
pcl::io::savePLYFile("output.ply", *cloud);
```

### 2. Voxel Grid Downsampling

Reduces point density while preserving structure:

```cpp
#include <pcl/filters/voxel_grid.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
    new pcl::PointCloud<pcl::PointXYZ>);

pcl::VoxelGrid<pcl::PointXYZ> voxel;
voxel.setInputCloud(cloud);
voxel.setLeafSize(0.1f, 0.1f, 0.1f);  // 10cm voxels
voxel.filter(*cloud_filtered);

std::cout << "Before: " << cloud->points.size() << " points" << std::endl;
std::cout << "After: " << cloud_filtered->points.size() << " points" << std::endl;
```

### 3. Statistical Outlier Removal

Removes noise based on neighbor statistics:

```cpp
#include <pcl/filters/statistical_outlier_removal.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
    new pcl::PointCloud<pcl::PointXYZ>);

pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
sor.setInputCloud(cloud);
sor.setMeanK(50);              // Number of neighbors to analyze
sor.setStddevMulThresh(1.0);   // Std dev multiplier threshold
sor.filter(*cloud_filtered);

// Get removed points (outliers)
pcl::PointCloud<pcl::PointXYZ>::Ptr outliers(
    new pcl::PointCloud<pcl::PointXYZ>);
sor.setNegative(true);
sor.filter(*outliers);
```

### 4. Radius Outlier Removal

Removes points with few neighbors:

```cpp
#include <pcl/filters/radius_outlier_removal.h>

pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
ror.setInputCloud(cloud);
ror.setRadiusSearch(0.5);       // Search radius (meters)
ror.setMinNeighborsInRadius(5); // Minimum neighbors required
ror.filter(*cloud_filtered);
```

### 5. PassThrough Filter

Filters points outside a range:

```cpp
#include <pcl/filters/passthrough.h>

pcl::PassThrough<pcl::PointXYZ> pass;
pass.setInputCloud(cloud);
pass.setFilterFieldName("z");
pass.setFilterLimits(0.0, 2.0);  // Keep points with z in [0, 2]
pass.filter(*cloud_filtered);

// Remove points in range instead
pass.setNegative(true);
pass.filter(*cloud_outside);
```

### 6. Normal Estimation

Compute surface normals for each point:

```cpp
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>  // OpenMP version

pcl::PointCloud<pcl::Normal>::Ptr normals(
    new pcl::PointCloud<pcl::Normal>);

// Standard version
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
ne.setInputCloud(cloud);

// Use KdTree for neighbor search
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
    new pcl::search::KdTree<pcl::PointXYZ>);
ne.setSearchMethod(tree);

// Set search radius (or use setKSearch for K nearest neighbors)
ne.setRadiusSearch(0.1);
ne.compute(*normals);

// Or use OpenMP parallel version
pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne_omp;
ne_omp.setNumberOfThreads(4);
ne_omp.setInputCloud(cloud);
ne_omp.setRadiusSearch(0.1);
ne_omp.compute(*normals);
```

### 7. Ground Plane Segmentation (RANSAC)

```cpp
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// Setup RANSAC plane segmentation
pcl::SACSegmentation<pcl::PointXYZ> seg;
seg.setOptimizeCoefficients(true);
seg.setModelType(pcl::SACMODEL_PLANE);
seg.setMethodType(pcl::SAC_RANSAC);
seg.setMaxIterations(1000);
seg.setDistanceThreshold(0.1);  // 10cm inlier threshold

seg.setInputCloud(cloud);

// Segment
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
seg.segment(*inliers, *coefficients);

std::cout << "Plane coefficients: "
          << coefficients->values[0] << " "
          << coefficients->values[1] << " "
          << coefficients->values[2] << " "
          << coefficients->values[3] << std::endl;

// Extract ground points
pcl::ExtractIndices<pcl::PointXYZ> extract;
extract.setInputCloud(cloud);
extract.setIndices(inliers);

pcl::PointCloud<pcl::PointXYZ>::Ptr ground(
    new pcl::PointCloud<pcl::PointXYZ>);
extract.setNegative(false);
extract.filter(*ground);

// Extract non-ground points
pcl::PointCloud<pcl::PointXYZ>::Ptr obstacles(
    new pcl::PointCloud<pcl::PointXYZ>);
extract.setNegative(true);
extract.filter(*obstacles);
```

### 8. Euclidean Cluster Extraction

Group points into clusters:

```cpp
#include <pcl/segmentation/extract_clusters.h>

// Create KdTree for search
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
    new pcl::search::KdTree<pcl::PointXYZ>);
tree->setInputCloud(cloud);

// Euclidean clustering
std::vector<pcl::PointIndices> cluster_indices;
pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
ec.setClusterTolerance(0.5);    // 50cm distance threshold
ec.setMinClusterSize(100);      // Minimum points per cluster
ec.setMaxClusterSize(25000);    // Maximum points per cluster
ec.setSearchMethod(tree);
ec.setInputCloud(cloud);
ec.extract(cluster_indices);

std::cout << "Found " << cluster_indices.size() << " clusters" << std::endl;

// Extract each cluster
int cluster_id = 0;
for (const auto& indices : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : indices.indices) {
        cluster->points.push_back(cloud->points[idx]);
    }
    cluster->width = cluster->points.size();
    cluster->height = 1;
    cluster->is_dense = true;

    std::cout << "Cluster " << cluster_id++ << ": "
              << cluster->points.size() << " points" << std::endl;
}
```

### 9. Point Cloud Visualization

```cpp
#include <pcl/visualization/pcl_visualizer.h>

pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("3D Viewer"));

viewer->setBackgroundColor(0, 0, 0);
viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
viewer->addCoordinateSystem(1.0);
viewer->initCameraParameters();

while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
}
```

---

## Pre-processing Pipeline for SLAM

```
Raw Point Cloud
       |
       v
+-------------------+
| Range Filtering   |  Remove points too close/far
| (PassThrough)     |
+-------------------+
       |
       v
+-------------------+
| Downsampling      |  Reduce density (VoxelGrid)
| (VoxelGrid)       |
+-------------------+
       |
       v
+-------------------+
| Outlier Removal   |  Remove noise (SOR/ROR)
| (Statistical)     |
+-------------------+
       |
       v
+-------------------+
| Ground Removal    |  Separate ground/obstacles
| (RANSAC Plane)    |  (Optional for odometry)
+-------------------+
       |
       v
+-------------------+
| Normal Estimation |  Compute surface normals
| (for GICP/NDT)    |  (for some ICP variants)
+-------------------+
       |
       v
Processed Point Cloud
(Ready for Registration)
```

---

## Performance Considerations

### Voxel Size Selection

| Voxel Size | Points (typical) | Use Case |
|------------|-----------------|----------|
| 0.01m | High | High-precision mapping |
| 0.05m | Medium | Indoor SLAM |
| 0.1m | Medium | Outdoor mapping |
| 0.5m | Low | Long-range sensing |

### Computational Costs

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| VoxelGrid | O(n) | 5-10 ms |
| SOR | O(n * k) | 50-100 ms |
| Normal Est. | O(n * k) | 100-500 ms |
| RANSAC | O(n * iter) | 10-50 ms |
| Clustering | O(n log n) | 50-200 ms |

---

## Common Data Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PCD | .pcd | PCL native, ASCII/binary |
| PLY | .ply | Meshes + clouds |
| LAS | .las | LiDAR standard |
| XYZ | .xyz | Simple ASCII |
| BIN | .bin | KITTI format |

### KITTI Binary Format

```cpp
// Read KITTI binary point cloud
void loadKITTIBin(const std::string& file,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    std::ifstream in(file, std::ios::binary);
    while (in.good()) {
        float x, y, z, intensity;
        in.read((char*)&x, sizeof(float));
        in.read((char*)&y, sizeof(float));
        in.read((char*)&z, sizeof(float));
        in.read((char*)&intensity, sizeof(float));

        pcl::PointXYZI point;
        point.x = x; point.y = y; point.z = z;
        point.intensity = intensity;
        cloud->points.push_back(point);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
}
```

---

## References

- [PCL Documentation](https://pointclouds.org/documentation/)
- [PCL Tutorials](https://pcl.readthedocs.io/projects/tutorials/en/latest/)
- Rusu & Cousins, "3D is here: Point Cloud Library (PCL)"
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
