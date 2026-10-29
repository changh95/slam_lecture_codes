# CUDA Acceleration Hands-on: A VIO Front End on the GPU with vilib

A visual-inertial odometry (VIO) front end does the same three things on every camera frame: build an
image pyramid, detect corners, and track the features of the previous frame into the new one. This
exercise times each step with **easy_profiler** on the CPU and on the GPU, using
[vilib](https://github.com/uzh-rpg/vilib), the CUDA front-end library of the Robotics and Perception
Group (UZH) behind the IROS 2020 paper *Faster than FAST: GPU-Accelerated Frontend for High-Speed VIO*.
Every run also checks that the CPU and GPU versions agree.

| Step | CPU, single thread | CPU, OpenCV | CUDA (vilib) |
|---|---|---|---|
| Image pyramid, 5 levels of 2x2 averaging | vilib `pyramid_create_cpu` (SSE2) | `cv::resize(INTER_AREA)`, 4 times | `pyramid_create_gpu` |
| FAST-9, 3x3 NMS, strongest corner per 32x32 cell | vilib `rosten::FASTCPU` (Rosten's C code) | vilib `opencv::FASTCPU` (`cv::FAST`) | `FASTGPU` |
| KLT front end: track, then detect in empty cells | - | `cv::calcOpticalFlowPyrLK` + Rosten FAST, on 1 thread and on all threads | `FeatureTrackerGPU` + `FASTGPU` |

The input is the left camera of [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
`MH_01_easy` (752x480, 20 Hz), frames 1000 to 1499 by default: steady flight at about 0.6 m/s
(ground truth). Around frames 400 to 800 the drone stands still, which would make tracking trivial.

**The tables compare the GPU's work, not the upload.** Each frame is copied to the GPU before its
timed block. On a desktop that copy crosses PCIe and costs more than some of the kernels, so it is
timed on its own and listed apart ("frame upload, not in the GPU rows"). On a Jetson the CPU and GPU
share one DRAM: there is no PCIe link, and a frame captured into memory the GPU can read directly
(pinned or unified memory) needs no copy at all. Add the upload row to the GPU row for the
end-to-end time on a desktop.

Timing GPU code needs one extra step. A kernel launch or an async copy returns as soon as the work
is queued, long before the GPU has done it. Every timed GPU block therefore ends with
`cudaDeviceSynchronize()`, so it measures what the CPU actually waits for: launch overhead and
kernels (and, for FAST, the copy of its small result back to the host).

---

## Project Structure

```
part5_ch03_08/
├── README.md
├── CMakeLists.txt
├── Dockerfile                    # slam:base + CUDA 12.8 + vilib (pinned, 3 build fixes)
├── include/
│   ├── benchmark_common.hpp      # arguments, EuRoC loader, GPU check, pinned frames, timing loop
│   ├── klt_reference.hpp         # vilib's KLT step in plain C++ (checks the CUDA tracker)
│   └── profile_summary.hpp       # reads a .prof back and prints the comparison table
└── examples/
    ├── pyramid_benchmark.cpp     # vilib CPU / cv::resize vs vilib CUDA kernels, three ways to upload
    ├── fast_benchmark.cpp        # Rosten / cv::FAST vs vilib CUDA FAST, per-cell agreement
    ├── tracker_benchmark.cpp     # OpenCV KLT front end vs vilib FeatureTrackerGPU
    └── prof_summary.cpp          # comparison tables for any .prof file
```

The examples are plain C++. Every CUDA kernel lives in `libvilib.so`, which the Dockerfile builds
for the GPU given by `CUDA_ARCH`.

---

## Requirements

- **An NVIDIA GPU** and a driver for CUDA 12.8 or newer (R570+). The image is built for
  `sm_120` (RTX 50xx); for another card pass its compute capability, e.g.
  `--build-arg CUDA_ARCH=86` for an RTX 30xx or `89` for an RTX 40xx.
- **Docker with the NVIDIA Container Toolkit**, so that `--gpus all` works.
- **EuRoC `MH_01_easy` in ASL format** (`mav0/cam0/data/*.png`, 2.6 GB). The repository's
  downloader puts it where the commands below expect it:

  ```bash
  python3 ../download_euroc_mav.py    # -> ~/data/euroc_mav/MH_01_easy/mav0/...
  ```

- OpenCV 4.x and easy_profiler come from `slam:base`.

---

## Build

```bash
# Docker (needs the slam:base image built from ../Dockerfile)
docker build . -t slam_zero_to_hero:part5_ch03_08

# Another GPU, e.g. an RTX 4090
docker build . --build-arg CUDA_ARCH=89 -t slam_zero_to_hero:part5_ch03_08
```

The Dockerfile clones vilib at its last upstream commit (June 2021) and fixes three things before
building it (details in the Dockerfile comments):

1. vilib compiles with `-Werror`, and GCC 13 warns about `DetectorBase::detect()` being hidden
   by `FASTGPU`'s overloads. The warning is harmless, so `-Werror` goes.
2. `libvilib.so` uses OpenCV's `imgproc`, `features2d` and `highgui` but links only `core`, so
   every program that links it fails with undefined references.
3. Unpatched, vilib silently builds for nvcc's default `sm_52`, whatever
   `-DCMAKE_CUDA_ARCHITECTURES` says. It calls `check_language(CUDA)` before `project()`, when CMake
   has not picked a build tool yet, so the check fails and vilib falls back to the old FindCUDA
   path, which ignores the variable. `sm_52` code still runs (the driver compiles the embedded PTX
   at start-up) but is not what we want to time. Passing `-DCMAKE_CUDA_COMPILER` skips the check,
   and `cuobjdump` verifies that the library holds `sm_<CUDA_ARCH>` code.

A local build needs vilib built the same way, then `mkdir build && cd build && cmake .. && make -j4`.

---

## Run

Mount `MH_01_easy` and pass the GPU to the container:

```bash
docker run --rm --gpus all -v ~/data/euroc_mav/MH_01_easy:/data/MH_01_easy:ro \
    slam_zero_to_hero:part5_ch03_08 ./pyramid_benchmark
```

Each demo takes optional arguments: `<demo> [cam0/data folder] [frames] [first frame]`. The defaults
are `/data/MH_01_easy/mav0/cam0/data`, 500 frames, and frame 1000. All frames are loaded into memory
before anything is timed. Every variant runs once per frame after 20 unprofiled warm-up calls. The
tracker warms up on the first 20 frames, then starts over from frame 0, because it carries tracks
from one frame to the next.

Each demo records every call in an `EASY_BLOCK`, writes `<demo>.prof` to the current directory,
reads that file back, and prints:

1. **Verification**: the GPU results compared with a CPU reference over all frames
2. **One table** of easy_profiler's median and mean in µs, with the speed-up of the median over the
   first CPU row. Rows starting with `-` are the stages of the GPU row above them. The upload rows
   at the bottom are not part of any GPU row. The mean also counts rare expensive calls, such as
   the frames in which the tracker detects new features.

The outputs below were recorded on an NVIDIA GeForce RTX 5090 (driver 580, PCIe link at 5.0 x8 on
this machine) with an AMD Ryzen 9 7950X (16 cores / 32 threads), in the
`slam_zero_to_hero:part5_ch03_08` image.

### 1. Image pyramid

```bash
docker run --rm --gpus all -v ~/data/euroc_mav/MH_01_easy:/data/MH_01_easy:ro \
    slam_zero_to_hero:part5_ch03_08 ./pyramid_benchmark
```

```
========================================
Image pyramid: CPU vs CUDA (easy_profiler)
========================================
GPU: NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), CUDA runtime 12.8, driver 13.0
Frames: 500 (752x480, #1000-#1499) from /data/MH_01_easy/mav0/cam0/data

Verification (reference: vilib_CPU, all 500 frames):
  vilib_GPU :  L1 max 0 (0.0% px)  L2 max 0 (0.0% px)  L3 max 0 (0.0% px)  L4 max 0 (0.0% px)
  OpenCV_CPU:  L1 max 1 (44.3% px)  L2 max 2 (82.0% px)  L3 max 3 (94.3% px)  L4 max 3 (97.9% px)

=== easy_profiler summary: pyramid_benchmark.prof ===

Pyramid                          median us     mean us      vs CPU   calls
--------------------------------------------------------------------------
  vilib_CPU                           47.5        46.4       1.00x     500
  OpenCV_CPU                          35.3        34.6       1.34x     500
  vilib_GPU                           18.5        18.7       2.57x     500
  frame upload, not in the GPU rows:
    upload                           103.3       103.8                 500
    upload_pinned                     94.7        94.7                 500
    upload_pinned_1D                  24.4        24.7                 500
(vs CPU = speed-up of the median over vilib_CPU; "- " rows are stages of the row above)

Profile saved to: pyramid_benchmark.prof
```

- **`vilib_GPU`** is the four pyramid kernels, with level 0 already on the GPU.
- **`upload`** copies the frame the way vilib does it: one `cudaMemcpy2DAsync` from ordinary
  (pageable) host memory into the pitched device buffer.
- **`upload_pinned`** is the same copy from pinned host memory.
- **`upload_pinned_1D`** copies the pinned frame contiguously into a plain device buffer, then
  into the pitched buffer on the device. The same 361 KB arrive almost 4x faster (see
  [Things to know](#things-to-know-before-using-vilib)).
- **`L1 max 1 (44.3% px)`**: OpenCV's level 1 differs from vilib's by at most 1, on 44% of the
  pixels. vilib averages with `(a + b + c + d) >> 2`, which rounds down; `INTER_AREA` rounds to
  nearest. Each level starts from the previous one, so the difference grows to 3 by level 4.

### 2. FAST corners with one corner per cell

```bash
docker run --rm --gpus all -v ~/data/euroc_mav/MH_01_easy:/data/MH_01_easy:ro \
    slam_zero_to_hero:part5_ch03_08 ./fast_benchmark
```

```
========================================
FAST-9 + grid NMS: CPU vs CUDA (easy_profiler)
========================================
GPU: NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), CUDA runtime 12.8, driver 13.0
Frames: 500 (752x480, #1000-#1499) from /data/MH_01_easy/mav0/cam0/data
FAST-9, threshold 10, 32x32 cells, border 8 px

Corners per frame: vilib_CPU 331.1, vilib_GPU 331.1 (of 360 cells)
Verification (reference: vilib_CPU, all 500 frames):
  vilib_GPU : 165320 of 165544 shared cells pick the same pixel (99.86%), 224 of the other 224 pick an equally scored corner; 0 cells only in vilib_CPU, 0 only here
  OpenCV_CPU: 57819 of 164888 shared cells pick the same pixel (35.07%), 0 of the other 107069 pick an equally scored corner; 656 cells only in vilib_CPU, 299 only here
  (cv::FAST scores corners differently, so it may keep another corner of the same cell)

=== easy_profiler summary: fast_benchmark.prof ===

FAST                             median us     mean us      vs CPU   calls
--------------------------------------------------------------------------
  vilib_CPU                         2752.8      2779.0       1.00x     500
  OpenCV_CPU                         889.8       900.0       3.09x     500
  vilib_GPU                           21.2        28.6     129.73x     500
  frame upload, not in the GPU rows:
    upload                           105.0       105.2                 500
(vs CPU = speed-up of the median over vilib_CPU; "- " rows are stages of the row above)

Profile saved to: fast_benchmark.prof
```

The GPU picks the same corner as Rosten's CPU code in every cell, up to ties: in 224 of the 165,544
cells (over 500 frames) the cell holds two corners with the same score, and the two
implementations break the tie differently. `cv::FAST` uses another score, so in about two thirds of
the cells it keeps a different corner.

`vilib_GPU` is the detection with the frame already on the GPU: the corner response and grid NMS
kernels, plus copying the per-cell results (5.6 KB) back to the host. vilib keeps the kernel-only
step private, so the small copy stays in. The mean (28.6 µs) is above the median because a few
calls took much longer.

### 3. KLT front end

```bash
docker run --rm --gpus all -v ~/data/euroc_mav/MH_01_easy:/data/MH_01_easy:ro \
    slam_zero_to_hero:part5_ch03_08 ./tracker_benchmark
```

```
========================================
KLT feature tracking: CPU vs CUDA (easy_profiler)
========================================
GPU: NVIDIA GeForce RTX 5090 (sm_120, 170 SMs), CUDA runtime 12.8, driver 13.0
Frames: 500 (752x480, #1000-#1499) from /data/MH_01_easy/mav0/cam0/data
5 levels, 16x16 patch, up to 200 features, detect below 100 tracks

Features per frame (mean)
                 tracked         new   frames with detection
  OpenCV           140.9         2.7                      12
  vilib_GPU        140.8         5.5                      26

Verification: every step vilib took (a feature from frame k to k+1), redone on the CPU
  vilib's rules in plain C++ (klt_reference.hpp): 70398 kept + 2642 dropped, 0 different keep/drop decisions, max |diff| 8.0e-03 px

=== easy_profiler summary: tracker_benchmark.prof ===

Tracker                          median us     mean us      vs CPU   calls
--------------------------------------------------------------------------
  OpenCV_1T                         1025.9      1094.9       1.00x     500
  OpenCV_32T                         415.7       462.1       2.47x     500
  vilib_GPU                           84.9        88.7      12.08x     500
    - pyramid                         18.3        18.1                 500
    - track                           66.4        70.2                 500
  frame upload, not in the GPU rows:
    upload                           106.5       106.6                 500
(vs CPU = speed-up of the median over OpenCV_1T; "- " rows are stages of the row above)

Profile saved to: tracker_benchmark.prof
```

Both front ends use vilib's default template: a feature is always aligned against its patch in the
frame where it was detected (as in SVO), starting from its position in the previous frame. The CPU
side keeps the pyramid of every frame that still has features detected in it, and tracks each group
with one `cv::calcOpticalFlowPyrLK(..., OPTFLOW_USE_INITIAL_FLOW)` call.

- **`vilib_GPU`** starts with the frame on the GPU: vilib's `Frame` constructor uploads it and
  builds its pyramid before the timed block.
  - **`pyramid`** builds the 5-level pyramid again inside the block. It is the same pyramid, so
    the block times exactly the GPU work of one frame.
  - **`track`** is `FeatureTrackerGPU::track()`: KLT for every feature, FAST in the empty cells
    when fewer than 100 tracks survive, and the templates of the new features.
- **Detection** runs in 26 of the 500 frames on vilib's side and in 12 on OpenCV's. Those frames
  account for most of the gap between OpenCV's mean and median: 12 CPU detections of about 2.75 ms
  each add up to 33 ms, close to the (1094.9 - 1025.9) µs x 500 = 34.5 ms between them.
- **Verification** redoes every step vilib took (a feature at vilib's position in frame k, aligned
  in frame k+1) with [`include/klt_reference.hpp`](include/klt_reference.hpp), which follows vilib's
  rules in plain C++. It lands within 0.008 px of the GPU and keeps or drops exactly the same
  tracks: the CUDA kernels compute what vilib's algorithm specifies, on every step of this
  sequence.

### Other sequences, other machines

```bash
# Another EuRoC sequence (python3 ../download_euroc_mav.py MH_02_easy), 1000 frames from frame 0
docker run --rm --gpus all -v ~/data/euroc_mav/MH_02_easy:/data/MH_02_easy:ro \
    slam_zero_to_hero:part5_ch03_08 ./tracker_benchmark /data/MH_02_easy/mav0/cam0/data 1000 0

# Keep the .prof files: run in a mounted folder (--rm deletes the container's own files)
docker run --rm --gpus all -v ~/data/euroc_mav/MH_01_easy:/data/MH_01_easy:ro -v "$PWD":/out -w /out \
    slam_zero_to_hero:part5_ch03_08 /workspace/part5_ch03_08/build/tracker_benchmark

# Print the tables of a .prof again, e.g. one recorded on a Jetson
./build/prof_summary pyramid_benchmark.prof fast_benchmark.prof tracker_benchmark.prof
```

Any folder of equally sized grayscale PNGs works, sorted by file name.

**Podman.** `--gpus all` is Docker's flag. Podman 4.x and newer pass the GPU with
`--device nvidia.com/gpu=all` (CDI). The rootless Podman 3.4 on the machine above has no CDI
support, so the outputs were recorded with the NVIDIA OCI hook instead:
`--hooks-dir=<folder with oci-nvidia-hook.json> -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility`.

---

## Results

The three demos above, 752x480, medians in µs. The GPU column is the GPU's work with the frame
already on the GPU; the speed-up in parentheses is over the first CPU column.

| Step | CPU, reference | CPU, other | vilib CUDA | GPU stages |
|---|---:|---:|---:|---|
| Pyramid, 5 levels | 47.5 (vilib) | 35.3 (1.34x, OpenCV) | 18.5 (2.57x) | 4 kernels |
| FAST-9 + grid NMS | 2752.8 (Rosten) | 889.8 (3.09x, `cv::FAST`) | 21.2 (130x) | kernels + 5.6 KB of results back |
| KLT front end | 1025.9 (OpenCV, 1 thread) | 415.7 (2.47x, 32 threads) | 84.9 (12.1x) | 18.3 pyramid + 66.4 track |

On a desktop, add the upload of each 361 KB frame over PCIe:

| Upload of a 752x480 frame | median |
|---|---:|
| vilib's `cudaMemcpy2DAsync`, pageable memory | 103.3 µs |
| the same from pinned memory | 94.7 µs |
| pinned, contiguous, then a 2D copy on the device | 24.4 µs |

Over three runs in a row, the GPU medians of the pyramid and the tracker moved by up to 7% (pyramid
18.5–19.8 µs, tracker 83.1–84.9 µs). FAST's moved most, 21.2–25.7 µs. The CPU medians moved by up to
5% (e.g. OpenCV's tracker on one thread at 1006.7–1025.9 µs).

All three GPU results match their CPU reference: the pyramid pixel for pixel, FAST cell for cell
(up to ties between equally scored corners), and KLT within 0.008 px with the same keep/drop
decisions.

### What the numbers say

1. **Kernel against CPU, the GPU wins every step, by how much depends on the work per pixel.** FAST
   compares each pixel with the circle of 16 around it and scores every corner it finds: 2.75 ms
   for Rosten's code, 0.89 ms for `cv::FAST`, 21 µs on the GPU (130x and 42x). The pyramid only
   averages 2x2 blocks, about one add per byte: 47.5 µs for vilib's CPU code, 35.3 µs for OpenCV,
   18.5 µs on the GPU (2.6x and 1.9x).
2. **On a desktop, the upload decides the small steps.** vilib's upload takes 103 µs for a 361 KB
   frame, more than five pyramid runs on the GPU. With it, the GPU pyramid (122 µs) is slower than
   either CPU version. With the contiguous upload from pinned memory (24.4 µs) it takes 42.9 µs:
   faster than vilib's CPU code, still slower than `cv::resize`. FAST stays far ahead either way:
   126 µs with vilib's upload is 22x faster than Rosten's code and 7x faster than `cv::FAST`. A
   front end uploads each frame once and then runs the pyramid, the detector and KLT on it, which
   is how vilib is built. On a Jetson, whose CPU and GPU share one DRAM, the PCIe copy disappears.
3. **The GPU front end is 12.1x faster than one CPU thread and 4.9x faster than all 32.** OpenCV
   spreads the work over its threads and gains 2.5x. With vilib's own upload (106.5 µs) the desktop
   GPU is still 5.4x faster than one thread and 2.2x faster than 32.
4. **Small kernels are mostly fixed costs.** The four pyramid kernels read and write about 0.6 MB,
   about 0.3 µs of memory traffic at the RTX 5090's 1.8 TB/s, yet the stage takes 18.5 µs. In a
   side experiment with a 1920x1080 image, 5.7 times as many pixels, the same kernels took 20 µs,
   so the time hardly depends on the pixel count. Launching four kernels and waiting for them is
   part of it (one empty kernel plus a wait took 4.7 µs in another side experiment); small grids
   on the coarse levels may be another. For work this small, the number of launches and
   synchronizations matters more than the kernel code.
5. **The two trackers keep tracks by different rules.** Both track about 141 features per frame,
   but vilib adds 5.5 new ones per frame and OpenCV 2.7, so vilib loses tracks about twice as
   often. vilib drops a track as soon as one pyramid level does not converge in 30 iterations;
   OpenCV's status only flags a feature that leaves the image or has too little texture. They also
   differ in smaller rules: vilib cuts its 16x16 template at integer pixels from x-8 to x+7 and uses
   central differences for the gradients, OpenCV samples from x-7.5 to x+7.5 and uses a Scharr
   filter. EuRoC has no per-feature ground truth, so this exercise does not say which one tracks
   more accurately.

### Things to know before using vilib

- **The upload path.** `Frame` and `Subframe::copy_from()` upload with a `cudaMemcpy2DAsync` from the
  `cv::Mat` into pitched memory, even on a Jetson, where that becomes a copy within the same DRAM.
  How fast a 2D copy from the host is depends on the row width. EuRoC's rows are 752 bytes: from
  pinned memory the 2D copy took 94.7 µs and the contiguous one plus a device-side 2D copy 24.4 µs.
  In a side experiment on this machine, rows that are a multiple of 128 bytes (768, 1024, 1920)
  copied as fast as one contiguous block, while 752-, 1856- and 1916-byte rows took 2–5x longer.
- **Two default streams.** vilib passes `--default-stream per-thread` to nvcc only, so the kernels in
  its `.cu` files run on each host thread's own default stream, while its uploads and
  synchronizations in `.cpp` files use the legacy default stream. The two stay in order because the
  legacy stream synchronizes with the per-thread ones. Adding your own streams needs care.
- **Previous-frame templates drift.** With `klt_template_is_first_observation = false`, vilib
  re-cuts the template at the feature's latest, sub-pixel position, but templates are cut at integer
  pixels (`REFERENCE_PATCH_INTERPOLATION` is 0 in `feature_tracker_cuda_tools.cu`), so the
  fractional part is lost on every frame. In a side experiment on this sequence, vilib's positions
  were off from OpenCV's frame-to-frame KLT by -0.50 px in x and in y on average, on every frame:
  the tracks creep up and to the left. Keep the default (`true`).
- **Non-convergence ends a track.** The iteration limit is a compile-time constant
  (`FEATURE_TRACKER_MAX_ITERATION_COUNT` = 30); `klt_max_iter` in `FeatureTrackerOptions` is never read.
- **Frames stay alive while their tracks do.** Each track holds a `shared_ptr` to the frame it was
  detected in, so that frame's GPU pyramid is not returned to the `PyramidPool` until the last of
  its tracks ends. `VilibFrontEnd` preallocates 10 pyramids with `PyramidPool::init`; the pool
  allocates more if needed.
- **Timing GPU stages.** Each timed GPU block and stage ends with `cudaDeviceSynchronize()`, so the
  tracker's row includes a wait between its pyramid and its tracking that a real pipeline would not
  have.
- **The image is tied to its build machine.** vilib compiles its CPU code with `-march=native`, and
  the CUDA code only for `CUDA_ARCH`.

---

## How vilib's kernels work

The examples call vilib; the kernels are in vilib's `visual_lib/src/**/*.cu`. Each example's header
comment describes its kernels in more detail.

- **Pyramid** (`pyramid_gpu.cu`). Level 0 sits in a pitched buffer (`cudaMallocPitch`), so every
  row starts on an aligned address and the loads of a warp along a row coalesce. Each thread writes
  4 consecutive output pixels (2 or 1 when the width is not a multiple of 4): it reads 4 `uchar2`
  pairs from the top and the bottom input row and computes `(a + b + c + d) >> 2` for each 2x2
  block. One kernel launch per level, all in one stream.
- **FAST** (`fast_gpu_cuda_tools.cu`). One thread per pixel. A pre-check on two opposite pairs of
  circle pixels rejects most pixels at once. For the rest, the 16 circle pixels form a 16-bit
  "darker" and a 16-bit "brighter" mask, and a precomputed 65536-bit table answers "does this mask
  hold 9 contiguous ones?" in one lookup, with no loop that would make the threads of a warp diverge.
  The score goes into a float response image.
- **Grid NMS** (`detector_base_gpu_cuda_tools.cu`). One thread block per 32x32 cell. Each thread
  compares its pixels with their 3x3 neighbours using sign arithmetic instead of branches, then
  warp shuffles (`__shfl_down_sync`) and shared memory reduce the cell to its best corner. Only the
  per-cell results (360 positions and scores) are copied back, not the response image.
- **KLT** (`feature_tracker_cuda_tools.cu`). Inverse compositional Lucas-Kanade, one warp per
  feature. A separate kernel cuts the template patch and inverts its 2x2 Hessian once, when the
  feature is detected. Each iteration then only samples the new frame under the patch (8 pixels per
  thread for 16x16), takes the template's gradients from the stored patch, and adds up `J * r`;
  `__shfl_xor_sync` sums the 32 partial results in every lane at once, so all threads compute the
  same update without shared memory or `__syncthreads()`. The per-feature results live in mapped
  (zero-copy) host memory, which the CPU reads after one `cudaStreamSynchronize()`.
  [`include/klt_reference.hpp`](include/klt_reference.hpp) is the same algorithm as a plain C++
  loop, for reading next to the kernel.

---

## Exercises

1. **A faster upload for the tracker.** vilib's `Frame` uploads through `Subframe::copy_from()`,
   which ends in `opencv_copy_from_image_common()` (`visual_lib/src/storage/opencv.cpp`) with a
   `cudaMemcpy2DAsync` from the `cv::Mat`. Change it to copy the frame contiguously into a device
   staging buffer and then into the pitched buffer, as `upload_pinned_1D` does in
   `pyramid_benchmark`. Patch vilib in the Dockerfile (with `sed` or a patch file), rebuild, and
   compare the tracker's `upload` row before and after, with the frames in ordinary memory and in
   pinned memory (`cuda_bench::PinnedFrames`). What is the desktop's end-to-end speed-up
   (GPU row + upload) now?
2. **compute-sanitizer.** The image includes NVIDIA's `compute-sanitizer`. Run
   `compute-sanitizer --tool memcheck ./fast_benchmark /data/MH_01_easy/mav0/cam0/data 10`, then
   `--tool racecheck`, `--tool synccheck` and `--tool initcheck`, and the same for
   `tracker_benchmark`. On this machine the first three report nothing. `initcheck` reports host
   copies of 5760 bytes (360 cells x 16 bytes) from partly uninitialized device memory in
   `DetectorBaseGPU::copyGridToHost()`. Which per-cell values does the grid NMS kernel leave
   unwritten in a cell without a corner, and why do the results stay correct anyway?
3. **Larger frames.** Run `pyramid_benchmark` and `fast_benchmark` on KITTI's left grayscale camera
   (1241x376, `-v <kitti>/sequences/00/image_0:/kitti:ro`, then `./fast_benchmark /kitti 500 0`)
   or on any video of your own at 1920x1080. Which GPU times grow with the pixel count, which do
   not, and why? What happens to the upload, with KITTI's 1241-byte rows?
4. **Illumination changes.** Set `affine_est_gain` and `affine_est_offset` to `true` in
   `VilibFrontEnd`, so that vilib also estimates a brightness gain and offset per feature. How do the
   track counts and the `track` stage change on EuRoC, which has auto-exposure? `klt_reference.hpp`
   does not model the affine terms, so expect the verification to report differences.
5. **Jetson.** The lecture just before this exercise introduces NVIDIA Jetson, where the CPU and GPU
   share the same DRAM. Build vilib there (for an Orin, `CUDA_ARCH=87`, from an L4T base image
   instead of `slam:base`), run the three benchmarks, and compare the `.prof` files with
   `prof_summary`. What happens to the upload rows on a GPU that has no PCIe link to cross?

---

## References

- B. Nagy, P. Foehn and D. Scaramuzza, [Faster than FAST: GPU-Accelerated Frontend for High-Speed VIO](http://rpg.ifi.uzh.ch/docs/IROS20_Nagy.pdf), IROS 2020
- [vilib](https://github.com/uzh-rpg/vilib) (BSD-3-Clause)
- M. Burri et al., [The EuRoC micro aerial vehicle datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets), IJRR 2016
- S. Baker and I. Matthews, Lucas-Kanade 20 Years On: A Unifying Framework, IJCV 2004 (inverse compositional KLT)
- E. Rosten and T. Drummond, [Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf), ECCV 2006
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [easy_profiler](https://github.com/yse/easy_profiler)
