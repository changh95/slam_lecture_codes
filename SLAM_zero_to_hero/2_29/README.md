# Iterative Closest Point (ICP) using PCL

This tutorial covers the Iterative Closest Point (ICP) algorithm for point cloud registration using the Point Cloud Library (PCL). ICP is fundamental to LiDAR odometry and 3D mapping.

---

## Overview

**Iterative Closest Point (ICP)** is an algorithm for aligning two point clouds by iteratively minimizing the distance between corresponding points.

### Applications

- **LiDAR Odometry**: Estimate sensor motion between scans
- **3D Reconstruction**: Merge multiple scans into a map
- **Object Recognition**: Align model to observed points
- **Loop Closure**: Refine pose after detecting revisited area

---

## ICP Algorithm

### Basic Steps

```
Input: Source cloud P, Target cloud Q, Initial transform T0

1. Initialize T = T0
2. Repeat until convergence:
   a. For each point p in P:
      - Find closest point q in Q (correspondence)
   b. Compute optimal T that minimizes:
      E(T) = sum ||T(p) - q||^2
   c. Apply T to P: P' = T(P)
3. Return final transformation T
```

### Mathematical Formulation

**Objective Function (Point-to-Point)**:
```
E(R, t) = sum_i ||R * p_i + t - q_i||^2
```

**Closed-form Solution**:
1. Compute centroids: `p_bar = mean(p_i)`, `q_bar = mean(q_i)`
2. Build cross-covariance: `H = sum (p_i - p_bar)(q_i - q_bar)^T`
3. SVD: `H = U * S * V^T`
4. Rotation: `R = V * U^T`
5. Translation: `t = q_bar - R * p_bar`

---

## Project Structure

```
2_29/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_clouds/
└── examples/
    ├── icp_basic.cpp              # Point-to-point ICP
    ├── icp_point_to_plane.cpp     # Point-to-plane ICP
    ├── icp_visualization.cpp      # Visualize alignment
    └── lidar_odometry.cpp         # Sequential scan registration
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
docker build . -t slam_zero_to_hero:2_29
```

---

## How to Run

```bash
# Basic ICP
./build/icp_basic source.pcd target.pcd

# Point-to-plane ICP
./build/icp_point_to_plane source.pcd target.pcd

# Visualization
./build/icp_visualization source.pcd target.pcd

# LiDAR odometry on sequence
./build/lidar_odometry /path/to/kitti/velodyne/
```

---

## ICP Variants

### 1. Point-to-Point ICP

Standard formulation minimizing point distances:

```
E = sum ||R*p_i + t - q_i||^2
```

**Pros**: Simple, fast
**Cons**: Slow convergence, local minima

### 2. Point-to-Plane ICP

Minimizes distance to tangent plane:

```
E = sum ((R*p_i + t - q_i) · n_i)^2
```

Where `n_i` is the normal at `q_i`.

**Pros**: Faster convergence, better for planar surfaces
**Cons**: Requires normals, more computation per iteration

### 3. Comparison

| Aspect | Point-to-Point | Point-to-Plane |
|--------|----------------|----------------|
| Convergence | Slower | 10x faster |
| Per iteration | Faster | Slower |
| Requirements | Points only | Points + normals |
| Accuracy | Good | Better |

---

## Code Examples

### 1. Basic Point-to-Point ICP

```cpp
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>

// Load point clouds
pcl::PointCloud<pcl::PointXYZ>::Ptr source(
    new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr target(
    new pcl::PointCloud<pcl::PointXYZ>);
pcl::io::loadPCDFile("source.pcd", *source);
pcl::io::loadPCDFile("target.pcd", *target);

// Setup ICP
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
icp.setInputSource(source);
icp.setInputTarget(target);

// Set parameters
icp.setMaximumIterations(50);
icp.setTransformationEpsilon(1e-8);
icp.setEuclideanFitnessEpsilon(1e-6);
icp.setMaxCorrespondenceDistance(0.5);

// Align
pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
    new pcl::PointCloud<pcl::PointXYZ>);
icp.align(*aligned);

// Get results
if (icp.hasConverged()) {
    std::cout << "ICP converged!" << std::endl;
    std::cout << "Fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;
} else {
    std::cout << "ICP did not converge!" << std::endl;
}
```

### 2. Point-to-Plane ICP

```cpp
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d_omp.h>

// Compute normals for target cloud
pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(
    new pcl::PointCloud<pcl::PointNormal>);

pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
ne.setInputCloud(target);
ne.setRadiusSearch(0.1);

pcl::PointCloud<pcl::Normal>::Ptr normals(
    new pcl::PointCloud<pcl::Normal>);
ne.compute(*normals);

// Concatenate points and normals
pcl::concatenateFields(*target, *normals, *target_with_normals);

// Also for source
pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(
    new pcl::PointCloud<pcl::PointNormal>);
ne.setInputCloud(source);
ne.compute(*normals);
pcl::concatenateFields(*source, *normals, *source_with_normals);

// Point-to-plane ICP
pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
icp.setInputSource(source_with_normals);
icp.setInputTarget(target_with_normals);
icp.setMaximumIterations(30);
icp.setTransformationEpsilon(1e-8);

pcl::PointCloud<pcl::PointNormal>::Ptr aligned(
    new pcl::PointCloud<pcl::PointNormal>);
icp.align(*aligned);
```

### 3. ICP with Initial Guess

```cpp
// Provide initial transformation estimate
Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

// From IMU or previous odometry
initial_guess(0, 3) = 0.1;  // x translation
initial_guess(1, 3) = 0.0;  // y translation
initial_guess(2, 3) = 0.0;  // z translation

// Align with initial guess
icp.align(*aligned, initial_guess);
```

### 4. Sequential LiDAR Odometry

```cpp
class LidarOdometry {
public:
    LidarOdometry() {
        icp_.setMaximumIterations(30);
        icp_.setTransformationEpsilon(1e-6);
        icp_.setMaxCorrespondenceDistance(1.0);

        global_pose_ = Eigen::Matrix4f::Identity();
    }

    void processCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
        // Downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZ>);
        voxel_.setInputCloud(cloud);
        voxel_.setLeafSize(0.1f, 0.1f, 0.1f);
        voxel_.filter(*filtered);

        if (prev_cloud_ == nullptr) {
            prev_cloud_ = filtered;
            return;
        }

        // ICP alignment
        icp_.setInputSource(filtered);
        icp_.setInputTarget(prev_cloud_);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
            new pcl::PointCloud<pcl::PointXYZ>);
        icp_.align(*aligned);

        if (icp_.hasConverged() && icp_.getFitnessScore() < 0.5) {
            // Get relative transformation
            Eigen::Matrix4f delta = icp_.getFinalTransformation();

            // Update global pose
            global_pose_ = global_pose_ * delta;

            std::cout << "Position: "
                      << global_pose_(0, 3) << ", "
                      << global_pose_(1, 3) << ", "
                      << global_pose_(2, 3) << std::endl;
        }

        prev_cloud_ = filtered;
    }

    Eigen::Matrix4f getPose() const { return global_pose_; }

private:
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
    pcl::VoxelGrid<pcl::PointXYZ> voxel_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
    Eigen::Matrix4f global_pose_;
};
```

### 5. Visualization

```cpp
#include <pcl/visualization/pcl_visualizer.h>

void visualizeICP(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target,
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned) {

    pcl::visualization::PCLVisualizer viewer("ICP Visualization");
    viewer.setBackgroundColor(0, 0, 0);

    // Source (green)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        source_color(source, 0, 255, 0);
    viewer.addPointCloud<pcl::PointXYZ>(source, source_color, "source");

    // Target (blue)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        target_color(target, 0, 0, 255);
    viewer.addPointCloud<pcl::PointXYZ>(target, target_color, "target");

    // Aligned (red)
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
        aligned_color(aligned, 255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZ>(aligned, aligned_color, "aligned");

    viewer.addCoordinateSystem(1.0);
    viewer.spin();
}
```

---

## ICP Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `MaximumIterations` | Max optimization iterations | 30-100 |
| `TransformationEpsilon` | Convergence threshold (transform change) | 1e-6 to 1e-8 |
| `EuclideanFitnessEpsilon` | Convergence threshold (MSE change) | 1e-6 |
| `MaxCorrespondenceDistance` | Max distance for correspondences | 0.5-2.0m |

### Parameter Tuning Tips

1. **MaxCorrespondenceDistance**:
   - Too small: Misses correct correspondences
   - Too large: Includes wrong correspondences
   - Start large, decrease during iterations

2. **Initial Guess**:
   - Critical for convergence
   - Use IMU, wheel odometry, or constant velocity model

3. **Preprocessing**:
   - Downsample to reduce computation
   - Remove outliers for robustness

---

## Convergence Criteria

```cpp
// Check convergence
if (icp.hasConverged()) {
    double fitness = icp.getFitnessScore();

    if (fitness < 0.1) {
        // Good alignment
    } else if (fitness < 0.5) {
        // Acceptable
    } else {
        // Poor alignment - may need different initial guess
    }
}
```

---

## Limitations of Standard ICP

| Issue | Description | Solution |
|-------|-------------|----------|
| Local minima | Converges to wrong solution | Good initial guess |
| Outliers | Wrong correspondences | Robust variants (GICP) |
| Partial overlap | Only part of clouds match | Robust cost functions |
| Degenerate geometry | Featureless environments | Multi-sensor fusion |
| Computational cost | O(n*m) per iteration | KD-tree, downsampling |

---

## Tips for Robust ICP

1. **Good Initial Guess**: Use IMU or motion model
2. **Multi-resolution**: Coarse-to-fine approach
3. **Outlier Rejection**: Use robust estimators
4. **Correspondence Rejection**: Remove poor matches
5. **Preprocessing**: Downsample, remove noise

```cpp
// Multi-resolution ICP
void multiResolutionICP(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target,
    Eigen::Matrix4f& transform) {

    std::vector<float> resolutions = {1.0f, 0.5f, 0.1f};

    transform = Eigen::Matrix4f::Identity();

    for (float res : resolutions) {
        // Downsample
        pcl::PointCloud<pcl::PointXYZ>::Ptr src_down(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_down(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setLeafSize(res, res, res);

        voxel.setInputCloud(source);
        voxel.filter(*src_down);

        voxel.setInputCloud(target);
        voxel.filter(*tgt_down);

        // ICP at this resolution
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(src_down);
        icp.setInputTarget(tgt_down);
        icp.setMaximumIterations(20);
        icp.setMaxCorrespondenceDistance(res * 5);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
            new pcl::PointCloud<pcl::PointXYZ>);
        icp.align(*aligned, transform);

        if (icp.hasConverged()) {
            transform = icp.getFinalTransformation();
        }
    }
}
```

---

## References

- Besl & McKay, "A Method for Registration of 3-D Shapes", 1992
- Chen & Medioni, "Object Modelling by Registration of Multiple Range Images", 1992
- [PCL ICP Tutorial](https://pcl.readthedocs.io/projects/tutorials/en/latest/iterative_closest_point.html)
- Pomerleau et al., "A Review of Point Cloud Registration Algorithms for Mobile Robotics"
