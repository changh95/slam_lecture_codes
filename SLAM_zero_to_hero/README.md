# SLAM_zero_to_hero

This repository contains code exercises for the lecture series - ['SLAM for Robotics & 3D Computer Vision' at FastCampus](https://fastcampus.co.kr/data_online_slam). This lecture series is delivered in Korean language.

![](title.png)

## How to use

Most of the code exercises are based on the base docker image. The base docker image contains numerous C++ libraries for SLAM.

You can build the base docker image using the following command.

```shell
docker build . --tag slam:base --progress=plain
echo "xhost +local:docker" >> ~/.profile
```

## Libraries in Base Image

| Library | Description |
|---------|-------------|
| **OpenCV 4.12** (with contrib) | Computer vision, feature detection (ORB, SIFT, TEBLID), ArUco markers |
| **Eigen 5.0** | Linear algebra, matrix operations |
| **Sophus** | Lie groups (SO3, SE3) for robotics |
| **Ceres Solver** | Nonlinear least squares optimization |
| **g2o** | Graph-based optimization for SLAM |
| **GTSAM** | Factor graph optimization |
| **PoseLib** | Minimal pose solvers (P3P, 5-point, homography) |
| **OpenGV** | Geometric vision algorithms (relative/absolute pose, triangulation) |
| **PCL** | Point cloud processing |
| **Pangolin** | 3D visualization |
| **easy_profiler** | CPU profiling with GUI |
| **SymForce** | Symbolic computation for robotics |
| **Rerun** | Modern 3D visualization for robotics |

## Table of contents

- Part 1: Introduction to SLAM
  - Chapter 1: What is SLAM?
    - Lecture introduction
    - Mobile robotics
    - What is SLAM?
    - Hardware used in SLAM
    - Types of SLAM
    - Applications of SLAM
    - Tips for studying SLAM
  - Chapter 2: Basic programming 
    - C++ and SLAM
    - [Basic C++ programming](ch02_02)
    - [Building C++ libraries](ch02_03)
    - [C++ CPU profiler](ch02_04)
    - [C++ memory profiler](ch02_05)
    - Python basics
    - [Basic Python programming](ch02_07)
    - [PyBind](ch02_08)
    - [ROS fundamentals](ch02_09)
  - Chapter 3: Basic maths
    - Rotation and translation in 3D space
    - Homogeneous coordinates
    - Lie Group
    - Basics of Lie algebra
    - [Eigen + Sophus library hands-on](1_21)
    - Continuous-time representation
  - Chapter 4: Sensor basics
    - Camera basics for robotics
    - Camera models
    - LiDAR basics
    - IMU basics
    - Radar basics
    - [Forward/Inverse kinematics](1_28)
    - [Sensor calibration](1_29)
    - [Kalibr package hands-on](1_30)
- Part 2: Dive into SLAM (Front-end)
  - Chapter 1: Image processing
    - Part 2 introduction
    - Local feature detection
    - [Classical local feature detection hands-on](2_3)
    - [Deep local feature detection hands-on](2_4)
    - Feature tracking basics
    - Advanced feature tracking in practice
    - [Feature tracking hands-on](2_7)
    - Global feature detection
    - [Global feature detection hands-on](2_9)
    - [Deep global feature detection hands-on](2_10)
  - Chapter 2: Multiple view geometry
    - Epipolar geometry
    - [Epipolar geometry hands-on](2_12)
    - Homography
    - [Homography hands-on](2_14)
    - [MonoVO hands-on](2_15)
    - Triangulation
    - [Triangulation hands-on](2_17)
    - Perspective-n-points
    - [Perspective-n-points hands-on](2_19)
    - RANSAC
    - Advanced RANSAC
    - [RANSAC hands-on](2_22)
    - M-estimator & MAXCON
  - Chapter 3: Point cloud processing
    - What is point cloud?
    - Introduction to PCL library
    - Point cloud preprocessing
    - [Point cloud preprocessing hands-on](2_27)
    - ICP
    - [ICP hands-on](2_29)
    - [Advanced ICP hands-on](2_30)
    - [Octree, Octomap, Bonxai hands-on](2_31)
- Chapter 3: Dive into SLAM (Back-end)
  - Chapter 1: Probabilistic graph inference
    - Part 3 introduction
    - Factor graph
    - Nonlinear least squares
    - Nonlinear optimization
    - Optimization on manifolds
    - Graph-based SLAM
    - Schur complement
    - Auto-diff
    - Continuous-time optimization
    - Sparsity in SLAM
    - Bundle adjustment
    - Nonlinear solvers
    - [g2o hands-on](3_13)
    - [GTSAM hands-on](3_14)
    - [Ceres-solver hands-on](3_15)
    - [SymForce hands-on](3_16)
    - [Kimera-RPGO hands-on](3_17)
  - Chapter 2: SLAM system architecture
    - SLAM systems
    - Various map representations
    - VSLAM system architecture
    - LiDAR SLAM system architecture
    - RADAR SLAM system architecture
    - Event SLAM system architecture
    - Inertial odometry basics
    - Leg odometry basics
    - Sensor fusion
- Chapter 4: Classical SLAM
  - Chapter 1: Introduction to classical SLAM pipelines
    - Part 4 introduction
    - Feature-based VSLAM
    - Direct VSLAM
    - Visual-inertial odometry
    - 2D LiDAR SLAM
    - 3D LiDAR SLAM
    - Sensor fusion SLAM
  - Chapter 2: Hands-on classical SLAM
    - ORB-SLAM 2
    - Basalt-VIO
    - Cartographer
    - KISS-SLAM
    - GLIM
    - FAST-LIO2
    - FAST-LIVO2
- Chapter 5: Advanced SLAM - AI Integration and Hardware Optimization
  - Chapter 1: AI + SLAM
    - Part 5 introduction
    - SLAM + Object detection + Segmentation
    - SLAM + Depth estimation
    - SLAM + Camera pose regression
    - SLAM + Deep feature matching
    - SLAM + Deep optical flow / scene flow
    - SLAM + Differentiable bundle adjustment
    - SLAM + Feed-forward 3D transformer
    - SLAM + NeRF / Implicit neural field
    - SLAM + Gaussian Splatting
    - SLAM + Video generation
    - SLAM + VLM/VLA
    - SLAM + 3D Scene graph
    - SLAM + Certifiably optimal algorithm
    - SLAM + Auto-encoder / diffusion
    - SLAM + Graph processor
  - Chapter 2: Hands on AI + SLAM
    - DSP-SLAM
    - Kimera
    - ConceptFusion
    - Gaussian Splatting SLAM
    - MASt3r-SLAM
    - PIN-SLAM
    - Suma++
  - Chapter 3: Hardware/Software optimization for SLAM performance
    - Differences between desktop, server, and embedded boards
    - Characteristics of real-time SLAM
    - Characteristics of auto-labeling / data-crunching SLAM
    - C++ build configuration optimization
    - SIMD acceleration and CPU optimization techniques
    - [SIMD acceleration hands-on](5_29)
    - Introduction to NVIDIA Jetson
    - [CUDA acceleration hands-on](5_31)
- Final projects
  - Project 1: SLAM for autonomous driving
  - Project 2: SLAM for drones
  - Project 3: SLAM for mobile scanner systems
  - Project 4: SLAM for quadruped robots
  - Project 5: SLAM for humanoid robots
  - Project 6: SLAM for VR/AR headsets

## Acknowledgements

ORB-SLAM 2/3 authors, Basalt-VIO authors, Cartographer authors, KISS-SLAM authors, GLIM authors, FAST-LIO2 authors, FAST-LIVO2 authors, DSP-SLAM authors, Kimera authors, ConceptFusion authors, MASt3r-SLAM authors, PIN-SLAM authors, Suma++ authors, and all the authors of the libraries used in this repository.

## Contributors

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
