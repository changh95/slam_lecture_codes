# g2o: General Graph Optimization

This tutorial covers [g2o](https://github.com/RainerKuemmerle/g2o), a general framework for graph-based optimization commonly used in SLAM systems like ORB-SLAM, LSD-SLAM, and many others.

---

## What is g2o?

g2o (General Graph Optimization) is a C++ framework for optimizing graph-based nonlinear error functions. It's widely used for:

- **Pose Graph Optimization**: Optimize robot trajectory from odometry and loop closures
- **Bundle Adjustment**: Jointly optimize camera poses and 3D point positions
- **Sensor Calibration**: Optimize sensor parameters

---

## Key Concepts

### Graph Structure

```
Vertices (Parameters to optimize):
  - Robot poses (SE2, SE3)
  - 3D points (PointXYZ)
  - Camera intrinsics

Edges (Constraints/Measurements):
  - Odometry between poses
  - 2D observations of 3D points
  - Loop closure constraints
```

### Common Vertex Types

| Type | Description | DOF |
|------|-------------|-----|
| `VertexSE2` | 2D pose (x, y, theta) | 3 |
| `VertexSE3Expmap` | 3D pose (SE3) | 6 |
| `VertexPointXYZ` | 3D point | 3 |
| `VertexSBAPointXYZ` | 3D point for BA | 3 |

### Common Edge Types

| Type | Description |
|------|-------------|
| `EdgeSE2` | 2D pose-pose constraint |
| `EdgeSE3Expmap` | 3D pose-pose constraint |
| `EdgeProjectXYZ2UV` | 3D point to 2D image projection |

---

## How to Build

Local build:
```bash
mkdir build && cd build
cmake ..
make -j4
```

Docker build:
```bash
docker build . -t slam_zero_to_hero:3_13
```

---

## Examples

### 1. 2D Pose Graph Optimization

```cpp
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam2d/types_slam2d.h>

// Create optimizer
SparseOptimizer optimizer;

// Add pose vertex
VertexSE2* v = new VertexSE2();
v->setId(0);
v->setEstimate(SE2(0, 0, 0));
v->setFixed(true);  // Anchor first pose
optimizer.addVertex(v);

// Add odometry edge
EdgeSE2* e = new EdgeSE2();
e->setVertex(0, optimizer.vertex(0));
e->setVertex(1, optimizer.vertex(1));
e->setMeasurement(SE2(1.0, 0.0, 0.0));
e->setInformation(Eigen::Matrix3d::Identity() * 100);
optimizer.addEdge(e);

// Optimize
optimizer.initializeOptimization();
optimizer.optimize(20);
```

### 2. Bundle Adjustment

```cpp
#include <g2o/types/sba/types_six_dof_expmap.h>

// Camera vertex
VertexSE3Expmap* cam = new VertexSE3Expmap();
cam->setId(0);
cam->setEstimate(SE3Quat());
optimizer.addVertex(cam);

// Point vertex
VertexPointXYZ* point = new VertexPointXYZ();
point->setId(1);
point->setEstimate(Eigen::Vector3d(1, 2, 3));
point->setMarginalized(true);  // For Schur complement
optimizer.addVertex(point);

// Observation edge
EdgeProjectXYZ2UV* obs = new EdgeProjectXYZ2UV();
obs->setVertex(0, point);
obs->setVertex(1, cam);
obs->setMeasurement(Eigen::Vector2d(100, 200));
obs->setInformation(Eigen::Matrix2d::Identity());
optimizer.addEdge(obs);
```

---

## BAL Dataset Format

Bundle Adjustment in the Large (BAL) datasets:

```
<num_cameras> <num_points> <num_observations>
<camera_idx> <point_idx> <x> <y>    # observations
...
<rotation(3)> <translation(3)> <f> <k1> <k2>  # cameras
...
<X> <Y> <Z>  # points
...
```

Download from: https://grail.cs.washington.edu/projects/bal/

---

## Common Functions

```cpp
// Optimizer setup
optimizer.setVerbose(true);
optimizer.setAlgorithm(solver);

// Vertex operations
vertex->setId(id);
vertex->setEstimate(initial_value);
vertex->setFixed(true);  // Don't optimize this vertex
vertex->setMarginalized(true);  // Use Schur complement

// Edge operations
edge->setVertex(0, v0);
edge->setVertex(1, v1);
edge->setMeasurement(measurement);
edge->setInformation(info_matrix);
edge->setRobustKernel(new RobustKernelHuber());

// Optimization
optimizer.initializeOptimization();
int iters = optimizer.optimize(max_iterations);
double chi2 = optimizer.chi2();  // Total error
```

---

## Running Examples

```bash
# 2D pose graph
./g2o_basics

# Bundle adjustment with BAL dataset
./g2o_bundle_adjustment problem-49-7776-pre.txt
```

Docker:
```bash
docker run -it --rm slam_zero_to_hero:3_13
```

---

## Tips

1. **Always fix at least one vertex** to remove gauge freedom
2. **Use robust kernels** for outlier rejection
3. **Set marginalized=true** for points in BA (enables Schur complement)
4. **Check chi2()** before and after optimization to verify improvement

---

## References

- [g2o GitHub](https://github.com/RainerKuemmerle/g2o)
- [g2o Tutorial Paper](https://github.com/RainerKuemmerle/g2o/blob/master/doc/g2o.pdf)
- [BAL Dataset](https://grail.cs.washington.edu/projects/bal/)
