# Desktop 2 — Hardware & Methodology

## Platform

| Component | Spec |
|---|---|
| **CPU** | AMD Ryzen 9 7950X (Zen 4, 16C/32T, 5.88 GHz boost, AVX-512, 64MB L3) |
| **GPU** | NVIDIA GeForce RTX 5090 (Blackwell, sm_120, 32 GB GDDR7, 600 W TDP) |
| **RAM** | 124 GB DDR5 |
| **Storage** | NVMe SSD, 1.6 TB free |
| **OS** | Ubuntu 22.04.5 LTS, kernel 6.8.0-107-generic |
| **CUDA driver** | 580.126.18 (CUDA 13.0 capable) |
| **Container engine** | Podman 4.x (Docker CLI compatible) |
| **Date** | 2026-05-09 |

## Comparison context

This platform is the natural rebuttal to the open question in `perf_bench/spark_vs_desktop_2018.md`:

> *"A Zen 4/5 desktop would likely win on CPU-only SLAM and match DGX Spark on most tasks, but unified memory systems will have a growing advantage as more SLAM algorithms adopt GPU acceleration."*

Desktop 2 is **exactly that hypothetical Zen 4 desktop**, paired with a Blackwell-era discrete GPU (RTX 5090, replacing Desktop 1's 6-year-old RTX 2080 Ti).

| | DGX Spark | Desktop 1 | **Desktop 2 (this run)** |
|---|---|---|---|
| CPU arch | ARM Neoverse V2 (2024) | x86 Zen+ (2018) | **x86 Zen 4 (2022)** |
| Cores / threads | 20 / 20 | 16 / 32 | **16 / 32** |
| L2 / core | ~2 MB | 512 KB | **1 MB** |
| L3 total | shared | 32 MB | **64 MB** |
| SIMD | NEON + SVE2 | AVX2 | **AVX2 + AVX-512** |
| Memory | LPDDR5x unified | DDR4 NUMA | **DDR5 flat (single CCD pair)** |
| GPU | Blackwell on-chip | RTX 2080 Ti (Turing) | **RTX 5090 (Blackwell)** |
| GPU memory | unified | 11 GB GDDR6 (PCIe) | **32 GB GDDR7 (PCIe5 x16)** |

## Build configuration (max-perf)

All Dockerfiles patched in this benchmark to add explicit Zen 4 tuning (the
existing dgx_spark / desktop_1 numbers used stock Release builds with default
flags):

```
-DCMAKE_CXX_FLAGS="-march=native -mtune=native -O3 -DNDEBUG [-fopenmp -ffast-math]"
```

This unlocks **AVX-512** code paths in Eigen, GTSAM, gtsam_points, and PCL. All
GTSAM builds keep `-DGTSAM_WITH_TBB=ON` for parallel iSAM2 / factor graph
optimization (Intel TBB scales to all 32 threads). Voxblox keeps OpenMP and
sets `integrator_threads=32`. FAST-LIO2 keeps `MP_PROC_NUM=3` (its internal
threadpool is fixed-size; raising it does not improve scan-matching latency).

ORB-SLAM2's existing `Portable_ORB_SLAM2` CMakeLists already sets
`-march=native -O3` (no patch needed).

**Note on cross-platform fairness**: DGX Spark and Desktop 1 results were
collected with stock Release builds (no `-march=native`), so on Zen 4 they
would have shipped only AVX2 SIMD via `-march=x86-64`. Desktop 2's numbers
therefore reflect best-case Zen 4 performance, while the historical numbers
reflect default-build performance on those platforms. This asymmetry is
intentional and disclosed.

## Datasets used

| Framework | Dataset | Source |
|---|---|---|
| ORB-SLAM2 | TUM RGB-D `freiburg1_xyz` | `download_tum_3d.py` |
| FAST-LIO2 | Hilti 2022 `exp14_basement_2` (Hesai PandarXT-32 + IMU) | `download_hilti_2022.py exp14_basement_2` |
| GLIM (CPU/GPU) | Hilti 2022 `exp14_basement_2` | same as above |
| Kimera-VIO | EuRoC MAV `MH_01_easy` (stereo + IMU) | DSpace REST API direct (download script's URL is broken — research-collection.ethz.ch SPA wrapper returns HTML, not zip) |
| Voxblox | cow_and_lady (depth + Vicon pose) | DSpace REST API direct |

EuRoC and cow_and_lady were fetched via the DSpace REST API endpoint
`/server/api/core/bitstreams/{uuid}/content` rather than the legacy
`/bitstreams/{uuid}/download` URLs (which now return the Angular SPA shell
rather than a binary stream).

## Run methodology

1. **Quiescent system**: no concurrent builds, no concurrent SLAM workloads.
2. **CPU governor**: ⚠ defaults to `powersave` on this machine. To match
   DGX Spark / Desktop 1 numbers under their (likely default) scheduling,
   the governor was switched to `performance` for the timed runs. Required:
   `sudo cpupower frequency-set -g performance`.
3. **GPU**: persistence mode default; runs use `--gpus all` (cuvslam, nvblox,
   glim_gpu, mast3r_slam scope only — not in this run).
4. **Profiler**: `easy_profiler` `EASY_BLOCK` instrumentation injected by the
   patches in `profiling/<framework>/`. Output: `<framework>_profiler.prof`,
   converted to JSON via `easy_profiler_converter` for analysis with
   `profiling/analyze_profiler.py`.

## Files in this directory

* `HARDWARE.md` — this file
* `<framework>.prof` / `<framework>.json` — per-framework profiler dumps
* `../desktop_2_vs_others.md` — cross-platform comparison report (written
  after all benchmarks complete)
