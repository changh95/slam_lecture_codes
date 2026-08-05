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
    - [Basic C++ programming](part1_ch02_02)
    - [Building C++ libraries](part1_ch02_03)
    - [C++ CPU profiler](part1_ch02_04)
    - [C++ memory profiler](part1_ch02_05)
    - Python basics
    - [Basic Python programming](part1_ch02_07)
    - [PyBind](part1_ch02_08)
    - [ROS fundamentals](part1_ch02_09)
  - Chapter 3: Basic maths
    - Rotation and translation in 3D space
    - Homogeneous coordinates
    - Lie Group
    - Basics of Lie algebra
    - [Eigen + Sophus library hands-on](part1_ch03_05)
    - Continuous-time representation
  - Chapter 4: Sensor basics
    - Camera basics for robotics
    - Camera models
    - LiDAR basics
    - IMU basics
    - Radar basics
    - [Forward/Inverse kinematics](part1_ch04_06)
    - [Sensor calibration](part1_ch04_07)
    - [Kalibr package hands-on](part1_ch04_08)
- Part 2: Dive into SLAM (Front-end)
  - Chapter 1: Image processing
    - Part 2 introduction
    - Local feature detection
    - [Classical local feature detection hands-on](part2_ch01_03)
    - [Deep local feature detection hands-on](part2_ch01_04)
    - Feature tracking basics
    - Advanced feature tracking in practice
    - [Feature tracking hands-on](part2_ch01_07)
    - Global feature detection
    - [Global feature detection hands-on](part2_ch01_09)
    - [Deep global feature detection hands-on](part2_ch01_10)
  - Chapter 2: Multiple view geometry
    - Epipolar geometry
    - [Epipolar geometry hands-on](part2_ch02_02)
    - Homography
    - [Homography hands-on](part2_ch02_04)
    - [MonoVO hands-on](part2_ch02_05)
    - Triangulation
    - [Triangulation hands-on](part2_ch02_07)
    - Perspective-n-points
    - [Perspective-n-points hands-on](part2_ch02_09)
    - RANSAC
    - Advanced RANSAC
    - [RANSAC hands-on](part2_ch02_12)
    - M-estimator & MAXCON
  - Chapter 3: Point cloud processing
    - What is point cloud?
    - [Introduction to PCL library](part2_ch03_02)
    - Point cloud preprocessing
    - [Point cloud preprocessing hands-on](part2_ch03_04)
    - ICP
    - [ICP hands-on](part2_ch03_06)
    - [Advanced ICP hands-on](part2_ch03_07)
    - [Octree, Octomap, Bonxai hands-on](part2_ch03_08)
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
    - [g2o hands-on](part3_ch01_13)
    - [GTSAM hands-on](part3_ch01_14)
    - [Ceres-solver hands-on](part3_ch01_15)
    - [SymForce hands-on](part3_ch01_16)
    - [Kimera-RPGO hands-on](part3_ch01_17)
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
    - [ORB-SLAM 2](orb_slam2)
    - [Basalt-VIO](basalt)
    - [SVO Pro](svo_pro_open)
    - [Cartographer](cartographer)
    - [KISS-SLAM](kiss_slam)
    - [GLIM](glim)
    - [FAST-LIO2](fast_lio2)
    - [FAST-LIVO2](fast_livo2)
    - [Cerberus 2.0](cerberus_2)
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
    - [DSP-SLAM](dsp_slam)
    - [Kimera](kimera)
    - [ConceptFusion](concept_fusion)
    - [Gaussian Splatting SLAM](gaussian_splatting_slam)
    - [MASt3r-SLAM](mast3r_slam)
    - [PIN-SLAM](pin_slam)
    - [Suma++](suma_pp)
    - [MonoLaneMapping](monolane_mapping)
  - Chapter 3: Hardware/Software optimization for SLAM performance
    - Differences between desktop, server, and embedded boards
    - Characteristics of real-time SLAM
    - Characteristics of auto-labeling / data-crunching SLAM
    - C++ build configuration optimization
    - SIMD acceleration and CPU optimization techniques
    - [SIMD acceleration hands-on](part5_ch03_06)
    - Introduction to NVIDIA Jetson
    - [CUDA acceleration hands-on](part5_ch03_08)
- Final projects
  - Project 1: [SLAM for autonomous driving](monolane_mapping)
  - Project 2: [SLAM for drones](svo_pro_open)
  - Project 3: [SLAM for mobile scanner systems](uamc)
  - Project 4: [SLAM for quadruped robots](cerberus_2)
  - Project 5: SLAM for humanoid robots
  - Project 6: SLAM for VR/AR headsets

## Acknowledgements

ORB-SLAM 2/3 authors, Basalt-VIO authors, SVO/SVO Pro authors, Cartographer authors, KISS-SLAM authors, GLIM authors, FAST-LIO2 authors, FAST-LIVO2 authors, Cerberus/Cerberus 2.0 authors, DSP-SLAM authors, Kimera authors, ConceptFusion authors, MASt3r-SLAM authors, PIN-SLAM authors, Suma++ authors, MonoLaneMapping authors, and all the authors of the libraries used in this repository.

## Contributors

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
