# Performance Comparison: DGX Spark vs Desktop 1 (2018)

## Platforms

| | DGX Spark | Desktop 1 |
|---|---|---|
| **CPU** | NVIDIA Grace (Neoverse V2, ARM, 2024) | AMD Threadripper 2950X (Zen+, x86, 2018) |
| **GPU** | Blackwell (on-chip, unified memory) | NVIDIA RTX 2080 Ti (discrete, PCIe) |
| **RAM** | Unified LPDDR5x | 64GB DDR4 (NUMA) |
| **Memory Architecture** | Unified (CPU+GPU share memory) | Discrete (separate CPU RAM + GPU VRAM) |
| **Cache** | ~2MB L2/core | 512KB L2/core, 32MB L3 |

## Benchmark Results

Speedup > 1.0 means Desktop 1 is faster than DGX Spark.

### ORB-SLAM2 (TUM freiburg1_xyz)

| Block | DGX Spark (ms) | Desktop 1 (ms) | Speedup |
|---|---|---|---|
| FrameProcess | 19.5 | 22.2 | 0.88x |
| FeatureExtraction | 11.3 | 11.3 | 1.00x |
| FeatureExtraction/KeyPoints | 7.5 | 7.4 | 1.01x |
| TrackLocalMap | 5.6 | 5.2 | 1.07x |
| LocalMapping | 123.4 | 164.7 | 0.75x |
| LocalBA | 59.1 | 81.1 | 0.73x |
| GlobalBA | 4416.3 | 1.3 | 3355x* |
| LoopClosureDetection | 8.8 | 2.5 | 3.59x |
| SearchForTriangulation | 1.5 | 0.4 | 3.99x |
| SearchByProjection | 0.3 | 0.5 | 0.52x |
| SearchByBoW | 0.3 | 0.6 | 0.59x |

*GlobalBA difference due to different number of loop closures triggered.

### FAST-LIO2 (Hilti exp14_basement_2, 740 LiDAR frames)

| Block | DGX Spark (ms) | Desktop 1 (ms) | Speedup |
|---|---|---|---|
| FrameProcess | 3.5 | 5.1 | 0.68x |
| Preprocessing | 1.6 | 2.4 | 0.68x |
| IMUProcessing | 1.1 | 1.8 | 0.59x |
| IMUProcessing/Undistort | 1.1 | 1.8 | 0.59x |
| EKFUpdate | 1.7 | 1.9 | 0.88x |
| ICP | 0.4 | 0.4 | 1.01x |
| MapUpdate | 0.07 | 0.20 | 0.37x |
| PointCloudPublish | 0.3 | 0.6 | 0.44x |

### Kimera (EuRoC MH_01_easy)

| Block | DGX Spark (ms) | Desktop 1 (ms) | Speedup |
|---|---|---|---|
| FrameProcess | 8.9 | 12.8 | 0.69x |
| FeatureExtraction | 7.2 | 13.8 | 0.52x |
| FeatureTracking | 2.4 | 2.8 | 0.86x |
| StereoMatching | 9.0 | 14.5 | 0.62x |
| RANSAC | 1.4 | 1.4 | 1.00x |
| VIOOptimization | 10.9 | 12.6 | 0.87x |
| BackendUpdate | 11.5 | 13.7 | 0.84x |

### GLIM CPU (Hilti exp14_basement_2)

| Block | DGX Spark (ms) | Desktop 1 (ms) | Speedup |
|---|---|---|---|
| FrameProcess | 13.5 | 14.9 | 0.91x |
| Preprocessing | 4.5 | 2.9 | **1.57x** |
| LocalMapping | 1.4 | 6.4 | 0.22x |

### GLIM GPU (Hilti exp14_basement_2)

| Block | DGX Spark (ms) | Desktop 1 (ms) | Speedup |
|---|---|---|---|
| FrameProcess | 14.8 | 15.7 | 0.94x |
| Preprocessing | 4.6 | 2.8 | **1.63x** |
| GlobalMapping | 7.0 | 11.1 | 0.63x |
| LocalMapping | 1.2 | 7.0 | 0.17x |

## Analysis

### Why DGX Spark wins most SLAM tasks

1. **Single-thread IPC**: Grace ARM cores (Neoverse V2, 2024) have ~50% better instructions-per-clock than Zen+ (2018). Most SLAM pipelines are largely single-threaded in the critical path.

2. **Memory latency & cache**: Grace has larger L2 caches (2MB/core vs 512KB/core). SLAM data structures (kd-trees, factor graphs, pose graphs) are pointer-heavy with random-access patterns. Larger caches keep more nodes resident, reducing cache misses.

3. **Unified memory (for GPU-accelerated SLAM)**: CPU and GPU share physical memory with zero-copy access. Eliminates PCIe transfer latency (~10us per transfer) for GPU scan matching -> CPU pose update pipeline.

4. **No NUMA penalties**: Threadripper 2950X has 2 NUMA nodes. Cross-node memory access adds latency. DGX Spark has flat memory access.

### Why Desktop 1 wins preprocessing (1.57-1.63x)

- **AVX2 SIMD**: x86 AVX2 (256-bit) accelerates point cloud filtering, voxelization, and normal estimation. These are regular, vectorizable operations on contiguous arrays.
- Preprocessing is **compute-bound** (arithmetic on point arrays), not memory-bound. This favors x86's wider SIMD units.

### Performance pattern by workload type

| Workload Type | Winner | Reason |
|---|---|---|
| Point cloud preprocessing | **Desktop 1** | AVX2 SIMD on regular contiguous data |
| ICP / RANSAC | Tie | Compute-bound, similar performance |
| Frame processing | DGX Spark | Better single-thread IPC |
| IMU / EKF update | DGX Spark | Single-thread IPC + cache |
| Mapping / optimization | **DGX Spark** | Memory latency + cache for random-access graph structures |
| Loop closure detection | Desktop 1 | BoW lookup benefits from large x86 L3 cache |

### CPU-only SLAM vs GPU-accelerated SLAM

For **CPU-only SLAM** (FAST-LIO2, KISS-SLAM, Cartographer):
- Unified memory provides **no benefit** (no GPU involved)
- DGX Spark wins purely due to **newer CPU cores with better IPC**, not architectural advantage
- A modern x86 CPU (Zen 4/5) would likely match or beat DGX Spark on CPU-only SLAM
- Key factors: single-thread IPC, L2/L3 cache size, memory latency

For **GPU-accelerated SLAM** (GLIM GPU, cuVSLAM, GPU-based NeRF SLAM):
- Unified memory provides a **structural advantage**
- Zero-copy CPU-GPU data sharing eliminates the PCIe bottleneck
- This advantage grows as more SLAM components move to GPU
- LocalMapping shows **5-6x advantage** for DGX Spark — the biggest gap

### Multi-threading utilization

Most SLAM algorithms use only 2-4 threads by design:
- ORB-SLAM2: 3 threads (tracking, mapping, loop closing)
- FAST-LIO2: Effectively single-threaded pipeline (`MP_PROC_NUM=3` for ICP only)
- Kimera: 2 threads (frontend + backend)
- GLIM: 4 threads

The Threadripper's 16 cores are largely underutilized. SLAM's pipeline dependencies (tracking must finish before mapping) limit parallelism. The exceptions are:
- Voxblox: Highly parallel TSDF integration (uses many worker threads)
- Bundle adjustment: Internally parallelized by Eigen/GTSAM but diminishing returns past 4-8 threads

### What would a modern desktop need to beat DGX Spark?

| Component | Recommendation |
|---|---|
| CPU | Zen 4/5 (Ryzen 9 7950X or Threadripper 7000) for comparable IPC at higher clocks |
| RAM | DDR5-6000+ for 2x bandwidth over DDR4 |
| GPU | RTX 4090 or better for GPU-accelerated SLAM |
| Key limitation | Discrete CPU+GPU architecture still has PCIe bottleneck for GPU SLAM |

A Zen 4/5 desktop would likely win on CPU-only SLAM and match DGX Spark on most tasks, but unified memory systems (DGX Spark, Jetson Thor, Apple M-series) will have a growing advantage as more SLAM algorithms adopt GPU acceleration.

## Conclusion

The DGX Spark's advantage over Desktop 1 (2018 Threadripper) comes from two orthogonal factors:
1. **6 years of CPU generational improvement** (IPC, cache, memory) — this can be matched by upgrading to modern x86
2. **Unified memory architecture** — this is a structural advantage for GPU-accelerated SLAM that discrete architectures cannot replicate

For the SLAM community: as algorithms increasingly leverage GPU (deep feature matching, differentiable optimization, neural radiance fields), unified memory architectures will become the optimal platform. For traditional CPU-only SLAM, a modern high-IPC processor with large caches remains the best choice regardless of architecture.
