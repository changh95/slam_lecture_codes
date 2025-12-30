# fastcampus_slam_codes

This repository contains code exercises for the following lecture series provided by @changh95 at FastCampus:

- ['Computer Vision, LiDAR processing, and Sensor Fusion for Autonomous Driving'](https://github.com/changh95/fastcampus_slam_codes/tree/main#zero-to-hero-slam-lectures-for-physical-ai-and-3d-computer-vision)
- ['SLAM Zero-to-Hero series for Physical AI and 3D Computer Vision'](https://github.com/changh95/fastcampus_slam_codes/tree/main#computer-vision-lidar-processing-and-sensor-fusion-for-autonomous-driving)

> Actively reworking the repository now. Stay tuned, because A LOT OF NEW TUTORIALS are on the way!

 </b>

## Zero-to-Hero SLAM lectures for Physical AI and 3D Computer Vision

![](./SLAM_zero_to_hero/title.png)

The course can be found [here](https://fastcampus.co.kr/data_online_slam).

The course content is essentially a superset of 'Computer Vision, LiDAR processing, and Sensor Fusion for Autonomous Driving', but with a more general focus within robotics, drones, AR/VR, autonomous driving.

### Table of Contents

- Chapter 1: Introduction to SLAM
  - 1.1 Lecture introduction
  - 1.2 Mobile robotics
  - 1.3 What is SLAM?
  - 1.4 Hardware used in SLAM
  - 1.5 Types of SLAM
  - 1.6 Applications of SLAM
  - 1.7 Tips for studying SLAM
  - 1.8 C++ and SLAM
  - 1.9 [Basic C++ programming](SLAM_zero_to_hero/1_9)
  - 1.10 [Building C++ libraries](SLAM_zero_to_hero/1_10)
  - 1.11 [C++ CPU profiler](SLAM_zero_to_hero/1_11)
  - 1.12 [C++ memory profiler](SLAM_zero_to_hero/1_12)
  - 1.13 Python basics
  - 1.14 [Basic Python programming](SLAM_zero_to_hero/1_14)
  - 1.15 [PyBind](SLAM_zero_to_hero/1_15)
  - 1.16 [ROS fundamentals](SLAM_zero_to_hero/1_16)
  - 1.17 Rotation and translation in 3D space
  - 1.18 Homogeneous coordinates
  - 1.19 Lie Group
  - 1.20 Basics of Lie algebra
  - 1.21 [Eigen + Sophus library hands-on](SLAM_zero_to_hero/1_21)
  - 1.22 Continuous-time representation
  - 1.23 Camera basics for robotics
  - 1.24 Camera models
  - 1.25 LiDAR basics
  - 1.26 IMU basics
  - 1.27 Radar basics
  - 1.28 Forward/Inverse kinematics
  - 1.29 Sensor calibration
  - 1.30 [Kalibr package hands-on](SLAM_zero_to_hero/1_30)
- Chapter 2: Dive into SLAM (Front-end)
  - 2.1 Part 2 introduction
  - 2.2 Local feature detection
  - 2.3 [Classical local feature detection hands-on](SLAM_zero_to_hero/2_3)
  - 2.4 [Deep local feature detection hands-on](SLAM_zero_to_hero/2_4)
  - 2.5 Feature tracking basics
  - 2.6 Advanced feature tracking in practice
  - 2.7 [Feature tracking hands-on](SLAM_zero_to_hero/2_7)
  - 2.8 Global feature detection
  - 2.9 [Global feature detection hands-on](SLAM_zero_to_hero/2_9)
  - 2.10 [Deep global feature detection hands-on](SLAM_zero_to_hero/2_10)
  - 2.11 Epipolar geometry
  - 2.12 [Epipolar geometry hands-on](SLAM_zero_to_hero/2_12)
  - 2.13 Homography
  - 2.14 [Homography hands-on](SLAM_zero_to_hero/2_14)
  - 2.15 [MonoVO hands-on](SLAM_zero_to_hero/2_15)
  - 2.16 Triangulation
  - 2.17 [Triangulation hands-on](SLAM_zero_to_hero/2_17)
  - 2.18 Perspective-n-points
  - 2.19 [Perspective-n-points hands-on](SLAM_zero_to_hero/2_19)
  - 2.20 RANSAC
  - 2.21 Advanced RANSAC
  - 2.22 [RANSAC hands-on](SLAM_zero_to_hero/2_22)
  - 2.23 M-estimator & MAXCON
  - 2.24 What is point cloud?
  - 2.25 Introduction to PCL library
  - 2.26 Point cloud preprocessing
  - 2.27 [Point cloud preprocessing hands-on](SLAM_zero_to_hero/2_27)
  - 2.28 ICP
  - 2.29 [ICP hands-on](SLAM_zero_to_hero/2_29)
  - 2.30 [Advanced ICP hands-on](SLAM_zero_to_hero/2_30)
  - 2.31 [Octree, Octomap, Bonxai hands-on](SLAM_zero_to_hero/2_31)
- Chapter 3: Dive into SLAM (Back-end)
  - 3.1 Part 3 introduction
  - 3.2 Factor graph
  - 3.3 Nonlinear least squares
  - 3.4 Nonlinear optimization
  - 3.5 Optimization on manifolds
  - 3.6 Graph-based SLAM
  - 3.7 Schur complement
  - 3.8 Auto-diff
  - 3.9 Continuous-time optimization
  - 3.10 Sparsity in SLAM
  - 3.11 Bundle adjustment
  - 3.12 Nonlinear solvers
  - 3.13 [g2o hands-on](SLAM_zero_to_hero/3_13)
  - 3.14 [GTSAM hands-on](SLAM_zero_to_hero/3_14)
  - 3.15 [Ceres-solver hands-on](SLAM_zero_to_hero/3_15)
  - 3.16 [SymForce hands-on](SLAM_zero_to_hero/3_16)
  - 3.17 SLAM systems
  - 3.18 Various map representations
  - 3.19 VSLAM system architecture
  - 3.20 LiDAR SLAM system architecture
  - 3.21 RADAR SLAM system architecture
  - 3.22 Event SLAM system architecture
  - 3.23 Inertial odometry basics
  - 3.24 Leg odometry basics
  - 3.25 Sensor fusion
- Chapter 4: Classical SLAM
  - 4.1 Part 4 introduction
  - 4.2 Feature-based VSLAM
  - 4.3 Direct VSLAM
  - 4.4 Visual-inertial odometry
  - 4.5 2D LiDAR SLAM
  - 4.6 3D LiDAR SLAM
  - 4.7 Sensor fusion SLAM
  - 4.8 ORB-SLAM 2
  - 4.9 Basalt-VIO
  - 4.10 Cartographer
  - 4.11 KISS-SLAM
  - 4.12 GLIM
  - 4.13 FAST-LIO2
  - 4.14 FAST-LIVO2
- Chapter 5: Advanced SLAM - AI Integration and Hardware Optimization
  - 5.1 Part 5 introduction
  - 5.2 SLAM + Object detection + Segmentation
  - 5.3 SLAM + Depth estimation
  - 5.4 SLAM + Camera pose regression
  - 5.5 SLAM + Deep feature matching
  - 5.6 SLAM + Deep optical flow / scene flow
  - 5.7 SLAM + Differentiable bundle adjustment
  - 5.8 SLAM + Feed-forward 3D transformer
  - 5.9 SLAM + NeRF / Implicit neural field
  - 5.10 SLAM + Gaussian Splatting
  - 5.11 SLAM + Video generation
  - 5.12 SLAM + VLM/VLA
  - 5.13 SLAM + 3D Scene graph
  - 5.14 SLAM + Certifiably optimal algorithm
  - 5.15 SLAM + Auto-encoder / diffusion
  - 5.16 SLAM + Graph processor
  - 5.17 DSP-SLAM
  - 5.18 Kimera
  - 5.19 ConceptFusion
  - 5.20 Gaussian Splatting SLAM
  - 5.21 MASt3r-SLAM
  - 5.22 PIN-SLAM
  - 5.23 Suma++
  - 5.24 Differences between desktop, server, and embedded boards
  - 5.25 Characteristics of real-time SLAM
  - 5.26 Characteristics of auto-labeling / data-crunching SLAM
  - 5.27 C++ build configuration optimization
  - 5.28 SIMD acceleration and CPU optimization techniques
  - 5.29 [SIMD acceleration hands-on](SLAM_zero_to_hero/5_29)
  - 5.30 Introduction to NVIDIA Jetson
  - 5.31 [CUDA acceleration hands-on](SLAM_zero_to_hero/5_31)
- Final projects
  - Project 1: SLAM for autonomous driving
  - Project 2: SLAM for drones
  - Project 3: SLAM for mobile scanner systems
  - Project 4: SLAM for quadruped robots
  - Project 5: SLAM for humanoid robots
  - Project 6: SLAM for VR/AR headsets

### Libraries in Base Docker Image

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

## Computer Vision, LiDAR processing, and Sensor Fusion for Autonomous Driving

![](./Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/title.png)

The course can be found [here](https://fastcampus.co.kr/data_online_autovehicle).

This course contains the following contents:

- Chapter 1: Introduction to SLAM
  - 1.1 Lecture introduction
  - 1.2 What is SLAM?
  - 1.3 Hardware for SLAM
  - 1.4 Types of SLAM
  - 1.5 Applications of SLAM
  - 1.6 Before we begin...
  - 1.7 [Basic C++ / CMake](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/1_7)
- Chapter 2: Introduction 3D Spaces
  - 2.1 3D rotation and translation
  - 2.2 [3D rotation and translation, using Eigen library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/2_2)
  - 2.3 Homogeneous coordinates
  - 2.4 Lie Group
  - 2.5 Basic Lie algebra
  - 2.6 [Lie Group and Lie algebra, using Sophus library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/2_6)
  - 2.7 How cameras work
  - 2.8 How LiDARs work
- Chapter 3: Image processing
  - 3.1 Local feature extraction & matching
  - 3.2 [Local feature extraction & matching, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/3_2)
  - 3.3 [Superpoint and Superglue, using C++ and TensorRT](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/3_3)
  - 3.4 Global feature extraction
  - 3.5 [Bag of Visual Words, using DBoW2 library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/3_5)
  - 3.6 [Learning-based global feature extraction, using PyTorch and Tensorflow libraries](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/3_6)
  - 3.7 Feature tracking
  - 3.8 [Optical flow, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/3_8)
- Chapter 4: Point cloud processing
  - 4.1 Introduction to point cloud processing
  - 4.2 [Introduction to point cloud processing, using PCL library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/4_2)
  - 4.3 Point cloud pre-processing
  - 4.4 [Point cloud pre-processing, using PCL library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/4_4)
  - 4.5 Iterative closest point
  - 4.6 [Iterative closest point, using PCL library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/4_6)
  - 4.7 Advanced ICP methods
  - 4.8 [Advanced ICP methods (G-ICP, NDT, TEASER++, KISS-ICP), using PCL library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/4_8)
  - 4.9 [Octree, Octomap, Bonxai, using PCL/Octomap/Bonxai libraries](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/4_9)
- Chapter 5: Multiple view geometry
  - 5.1 Epipolar geometry
  - 5.2 [Essential and Fundamental matrix estimation, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_2)
  - 5.3 Homography
  - 5.4 [Bird's eye view (BEV) projection, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_4)
  - 5.5 [Simple monocular visual odometry, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_5)
  - 5.6 Triangulation
  - 5.7 [Triangulation, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_7)
  - 5.8 Perspective-n-Points (PnP) and Direct Linear Transform (DLT)
  - 5.9 [Fiducial marker tracking, using OpenCV library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_9)
  - 5.10 RANSAC
  - 5.11 Advanced RANSAC methods (USAC)
  - 5.12 [RANSAC and USAC, using OpenCV and RansacLib libraries](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_12)
  - 5.13 Graph-based SLAM
  - 5.14 Least squares
  - 5.15 Schur complement
  - 5.16 Bundle adjustment
  - 5.17 [Bundle adjustment, using Ceres-Solver library](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/5_17)
- Chapter 6: Visual-SLAM
  - 6.1 Overview of feature-based VSLAM
  - 6.2 Overview of direct VSLAM
  - 6.3 Overview of visual-inertial odometry (VIO)
  - 6.4 Spatial AI
  - 6.5 [ORB-SLAM2](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/orb_slam2), [ORB-SLAM3](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/orb_slam3)
  - 6.6 [DynaVINS](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/dynavins)
  - 6.7 [CubeSLAM](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/cubeslam)
- Chapter 7: LiDAR SLAM
  - 7.1 Overview of 2D LiDAR SLAM
  - 7.2 Overview of 3D LiDAR SLAM and LiDAR-inertial odometry
  - 7.3 [HDL-Graph-SLAM](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/hdl_graph_slam)
  - 7.4 [KISS-ICP](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/kiss_icp)
  - 7.5 [SHINE-Mapping](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/shine_mapping)
- Chapter 8: CI/CD for SLAM
  - 8.1 TDD and tests
  - 8.2 CI/CD
  - 8.3 CI agents
  - 8.4 CI/CD for Python SLAM projects
  - 8.5 CI/CD for C++ SLAM projects
- Final projects:
  - [DSP-SLAM](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/dsp_slam)
  - [Suma++](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/suma_plus_plus)
  - [Cartographer-KITTI](Computer_Vision_LiDAR_Processing_and_Sensor_Fusion_for_Autonomous_Driving/cartographer)


