# Advanced ICP Methods: GICP, NDT, TEASER++, and KISS-ICP

This tutorial covers advanced point cloud registration methods that address limitations of standard ICP: Generalized ICP (GICP), Normal Distributions Transform (NDT), TEASER++ (robust global registration), and KISS-ICP (simple modern LiDAR odometry).

---

## Overview

Standard ICP has limitations that advanced methods address:

| Method | Strength | Best For |
|--------|----------|----------|
| **GICP** | Probabilistic, handles uncertainty | Accurate local registration |
| **NDT** | Fast, grid-based | Real-time odometry |
| **TEASER++** | Globally optimal, outlier-robust | Initial alignment, loop closure |
| **KISS-ICP** | Simple, modern, real-time | Production LiDAR odometry |

---

## 1. Generalized ICP (GICP)

GICP combines point-to-point and point-to-plane ICP in a probabilistic framework.

### Mathematical Formulation

Models both source and target points as Gaussian distributions:
```
p_i ~ N(p_i, C_i^A)
q_i ~ N(q_i, C_i^B)

Minimize: sum d_i^T (C_i^B + R*C_i^A*R^T)^(-1) d_i
where d_i = q_i - (R*p_i + t)
```

### Covariance Models

- **Plane-to-plane**: Elongated along surface normal
- **Point-to-point**: Isotropic (standard ICP)
- **Point-to-plane**: Intermediate

### Code Example

```cpp
#include <pcl/registration/gicp.h>

pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
gicp.setInputSource(source);
gicp.setInputTarget(target);

// GICP-specific parameters
gicp.setMaximumIterations(50);
gicp.setTransformationEpsilon(1e-8);
gicp.setMaxCorrespondenceDistance(1.0);
gicp.setCorrespondenceRandomness(20);  // Number of neighbors for covariance
gicp.setMaximumOptimizerIterations(20);

pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
    new pcl::PointCloud<pcl::PointXYZ>);
gicp.align(*aligned);

std::cout << "GICP fitness: " << gicp.getFitnessScore() << std::endl;
std::cout << "Transform:\n" << gicp.getFinalTransformation() << std::endl;
```

---

## 2. Normal Distributions Transform (NDT)

NDT represents the target cloud as a grid of Gaussian distributions.

### Algorithm

1. Divide target space into cells (voxels)
2. For each cell, compute mean and covariance of points
3. For source points, maximize likelihood under target distributions

### Advantages

- Very fast (no per-point correspondences)
- Smooth cost function
- Works well with sparse data

### Code Example

```cpp
#include <pcl/registration/ndt.h>

pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
ndt.setInputSource(source);
ndt.setInputTarget(target);

// NDT-specific parameters
ndt.setResolution(1.0);           // Cell size (meters)
ndt.setStepSize(0.1);             // Newton optimization step size
ndt.setTransformationEpsilon(0.01);
ndt.setMaximumIterations(50);

// Provide initial guess (important for NDT)
Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
    new pcl::PointCloud<pcl::PointXYZ>);
ndt.align(*aligned, initial_guess);

std::cout << "NDT converged: " << ndt.hasConverged() << std::endl;
std::cout << "Fitness: " << ndt.getFitnessScore() << std::endl;
```

### NDT Parameter Tuning

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| Resolution | Cell size, affects smoothness | 0.5-2.0m |
| StepSize | Optimization step, affects speed | 0.05-0.5 |
| MaxIterations | Convergence limit | 30-100 |

---

## 3. TEASER++ (Truncated least squares Estimation And SEmidefinite Relaxation)

TEASER++ is a globally optimal, certifiably robust registration method.

### Key Features

- **Outlier-robust**: Works with >95% outliers
- **Certifiable**: Provides optimality guarantee
- **No initial guess**: Global registration

### When to Use

- Loop closure verification
- Relocalization
- Initial alignment for ICP

### Code Example

```cpp
#include <teaser/registration.h>

// Prepare correspondences
teaser::PointCloud src_cloud, tgt_cloud;
for (const auto& p : source->points) {
    src_cloud.push_back({p.x, p.y, p.z});
}
for (const auto& p : target->points) {
    tgt_cloud.push_back({p.x, p.y, p.z});
}

// Setup TEASER++
teaser::RobustRegistrationSolver::Params params;
params.noise_bound = 0.05;  // Noise threshold
params.cbar2 = 1.0;         // TLS parameter
params.estimate_scaling = false;
params.rotation_max_iterations = 100;
params.rotation_gnc_factor = 1.4;
params.rotation_estimation_algorithm =
    teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
params.rotation_cost_threshold = 0.005;

teaser::RobustRegistrationSolver solver(params);
solver.solve(src_cloud, tgt_cloud);

// Get result
auto solution = solver.getSolution();
Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
transform.block<3, 3>(0, 0) = solution.rotation;
transform.block<3, 1>(0, 3) = solution.translation;

std::cout << "TEASER++ valid: " << solution.valid << std::endl;
std::cout << "Transform:\n" << transform << std::endl;
```

---

## 4. KISS-ICP (Keep It Simple and Stupid ICP)

KISS-ICP is a modern, simple LiDAR odometry pipeline that achieves state-of-the-art results.

### Key Innovations

1. **Adaptive threshold**: Automatically adjusts correspondence threshold
2. **Voxel-based correspondences**: Fast approximate nearest neighbors
3. **Point-to-point ICP**: Simple but effective
4. **Motion compensation**: Handles sensor motion during scan

### Pipeline

```
Raw LiDAR Scan
      |
      v
+------------------+
| Motion Compen-   |  Deskew points using predicted motion
| sation           |
+------------------+
      |
      v
+------------------+
| Voxel Down-      |  Efficient representation
| sampling         |
+------------------+
      |
      v
+------------------+
| Adaptive ICP     |  Point-to-point with adaptive threshold
+------------------+
      |
      v
+------------------+
| Local Map        |  Maintain sliding window map
| Management       |
+------------------+
      |
      v
Pose Estimate
```

### Code Example (Using kiss-icp library)

```cpp
#include <kiss_icp/pipeline/KissICP.hpp>

// Configuration
kiss_icp::pipeline::KISSConfig config;
config.max_range = 100.0;
config.min_range = 5.0;
config.deskew = true;
config.voxel_size = 1.0;
config.max_points_per_voxel = 20;
config.initial_threshold = 2.0;

// Create pipeline
kiss_icp::pipeline::KissICP kiss_icp(config);

// Process frames
for (const auto& cloud : lidar_sequence) {
    // Convert to kiss-icp format
    std::vector<Eigen::Vector3d> points;
    for (const auto& p : cloud->points) {
        points.emplace_back(p.x, p.y, p.z);
    }

    // Register
    kiss_icp.RegisterFrame(points);

    // Get pose
    Sophus::SE3d pose = kiss_icp.pose();
    std::cout << "Position: "
              << pose.translation().transpose() << std::endl;
}
```

### Python Interface

```python
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.config import KISSConfig

config = KISSConfig()
config.data.max_range = 100.0
config.data.min_range = 5.0

pipeline = OdometryPipeline(config)

for cloud in lidar_sequence:
    pose = pipeline.process(cloud)
    print(f"Position: {pose[:3, 3]}")
```

---

## Method Comparison

### Accuracy Benchmarks (KITTI)

| Method | Translation Error | Rotation Error | FPS |
|--------|------------------|----------------|-----|
| ICP | 1.5% | 0.006 deg/m | 20 |
| GICP | 1.0% | 0.004 deg/m | 10 |
| NDT | 1.2% | 0.005 deg/m | 30 |
| KISS-ICP | 0.8% | 0.003 deg/m | 50+ |

### When to Use Each

| Scenario | Recommended Method |
|----------|-------------------|
| Real-time odometry | KISS-ICP or NDT |
| High accuracy needed | GICP |
| Unknown initial pose | TEASER++ + GICP |
| Loop closure | TEASER++ |
| Structured environments | NDT |
| Unstructured outdoor | KISS-ICP |

---

## Project Structure

```
2_30/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   └── sample_sequences/
└── examples/
    ├── gicp_demo.cpp              # GICP registration
    ├── ndt_demo.cpp               # NDT registration
    ├── teaser_demo.cpp            # TEASER++ global registration
    ├── kiss_icp_demo.cpp          # KISS-ICP odometry
    └── method_comparison.cpp      # Compare all methods
```

---

## How to Build

### Dependencies
- PCL 1.12+
- Eigen3
- TEASER++ (optional)
- KISS-ICP (optional)

### Local Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Docker Build

```bash
docker build . -t slam_zero_to_hero:2_30
```

---

## How to Run

```bash
# GICP
./build/gicp_demo source.pcd target.pcd

# NDT
./build/ndt_demo source.pcd target.pcd --resolution 1.0

# TEASER++
./build/teaser_demo source.pcd target.pcd

# KISS-ICP
./build/kiss_icp_demo /path/to/kitti/velodyne/

# Compare methods
./build/method_comparison source.pcd target.pcd
```

---

## Advanced: Custom Registration Pipeline

```cpp
class LidarOdometryAdvanced {
public:
    Eigen::Matrix4f registerClouds(
        pcl::PointCloud<pcl::PointXYZ>::Ptr source,
        pcl::PointCloud<pcl::PointXYZ>::Ptr target) {

        // 1. Coarse alignment with TEASER++ (if needed)
        Eigen::Matrix4f initial = Eigen::Matrix4f::Identity();
        if (!hasGoodInitialGuess()) {
            initial = runTEASER(source, target);
        }

        // 2. Fine alignment with GICP
        pcl::GeneralizedIterativeClosestPoint<
            pcl::PointXYZ, pcl::PointXYZ> gicp;
        gicp.setInputSource(source);
        gicp.setInputTarget(target);
        gicp.setMaximumIterations(30);

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(
            new pcl::PointCloud<pcl::PointXYZ>);
        gicp.align(*aligned, initial);

        return gicp.getFinalTransformation();
    }
};
```

---

## References

- Segal et al., "Generalized-ICP", RSS 2009
- Biber & Strasser, "The Normal Distributions Transform", IROS 2003
- Yang et al., "TEASER: Fast and Certifiable Point Cloud Registration", IEEE T-RO 2020
- Vizzo et al., "KISS-ICP: In Defense of Point-to-Point ICP", IEEE RA-L 2023
- [KISS-ICP GitHub](https://github.com/PRBonn/kiss-icp)
- [TEASER++ GitHub](https://github.com/MIT-SPARK/TEASER-plusplus)
