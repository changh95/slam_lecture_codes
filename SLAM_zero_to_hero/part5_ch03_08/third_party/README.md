# vilib (vendored)

The CUDA front-end library of the Robotics and Perception Group (UZH), by Balazs Nagy et al.: a copy
of the last upstream commit, with its licenses. The Dockerfile copies `vilib/` into the image and
builds `libvilib.so` from it, so nothing is downloaded at build time.

| Folder | Upstream commit | License | Used by |
|---|---|---|---|
| `vilib/` | [uzh-rpg/vilib@7a12fe1](https://github.com/uzh-rpg/vilib/tree/7a12fe1db78ca755876469979394076f10fe577c) (June 25, 2021) | BSD-3-Clause (`LICENSE.txt`) | `libvilib.so`: every example except `prof_summary` |
| `vilib/visual_lib/src/feature_detection/fast/rosten/` | shipped in the same commit (Edward Rosten's FAST C code) | BSD-3-Clause (`LICENSE` there) | `rosten::FASTCPU`, the single-thread CPU baseline of `fast_benchmark` |

Every file the upstream commit tracks is here except `visual_lib/test/images/` (193 MB of test
images and Blender EXR depth maps). Only vilib's own test program uses them (`visual_lib/test`,
built by `visual_lib/Makefile`), and this exercise does not build it. To run those tests, take the
folder from the upstream commit.

`CMakeLists.txt` builds `libvilib.so` from `visual_lib/src/**/*.cpp` and `**/*.cu` and installs it with
its headers and a CMake package (`find_package(vilib)`, target `vilib::vilib`). The ROS nodes
(`ros/`), the Jetson clock scripts (`scripts/`), `visual_lib/Makefile` and `visual_lib/test/` are
kept for reference and are not built.

## Local changes

- **`CMakeLists.txt`, two build fixes**, each marked `[SLAM zero-to-hero patch]`:
  - `-Werror -Wfatal-errors` removed from the C++ options. GCC 13's `-Wall` includes
    `-Woverloaded-virtual`, which flags `DetectorBase::detect()` being hidden by `FASTGPU`'s
    overloads. The warning is harmless, but with `-Werror` it stops the build.
  - OpenCV's `imgproc`, `features2d` and `highgui` added to `find_package(OpenCV)` and to the link
    line. `libvilib.so` calls `cv::imshow`, `cv::FastFeatureDetector`, `cv::circle`, ... but linked
    only `opencv_core`, so every program that linked it failed with undefined references.

  A third problem is fixed from the command line, not in the file: unpatched, vilib builds for
  nvcc's default `sm_52` whatever `CMAKE_CUDA_ARCHITECTURES` says, because `check_language(CUDA)`
  runs before `project()` and fails. The Dockerfile passes `-DCMAKE_CUDA_COMPILER`, which skips the
  check, and verifies the result with `cuobjdump`. A `[SLAM zero-to-hero]` comment above
  `check_language` explains it for local builds.
- **Comments for the course** in the CUDA kernels and the host code around them. They explain why
  each part is built the way it is and which acceleration technique it uses. Every added comment
  starts with `[SLAM zero-to-hero]`, and only comment lines were added: no line of upstream code
  was changed or removed, so outside `CMakeLists.txt` a `diff` against the upstream commit shows
  additions only. Each annotated file opens with a "Design notes" block, and notes inside the code
  mark the decisions as they occur.

## Where to start reading

Follow a frame through the front end. Each file below opens with a "Design notes" block (what runs
where, which memory it uses, which techniques, and why), and notes in the code mark each decision.
Paths are under `vilib/visual_lib/`.

| Step | Files | What to look for |
|---|---|---|
| Upload | `src/storage/opencv.cpp`, `src/storage/subframe.cpp` | pitched device memory; pageable vs pinned host memory; why a 2D copy of 752-byte rows is slow. Exercise 1 starts in `opencv.cpp` |
| Memory pools | `src/storage/pyramid_pool.cpp`, `src/storage/subframe_pool.cpp` | preallocating, so that `cudaMalloc`/`cudaFree` (which waits for the whole device) stay out of the per-frame path |
| Pyramid | `src/preprocess/pyramid_gpu.cu`; CPU baseline `src/preprocess/pyramid_cpu.cpp` | several output pixels per thread, read-only (`const __restrict__`) loads, one launch per level; SSE2 / NEON on the CPU side |
| FAST | `src/feature_detection/fast/fast_gpu_cuda_tools.cu`, `fast_gpu.cpp` | one thread per pixel, the early-reject test, the 64K-entry lookup table that replaces the arc-search loop |
| Grid NMS | `src/feature_detection/detector_base_gpu_cuda_tools.cu`, `detector_base_gpu.cpp` | one block per cell, warp shuffles and shared memory for the reduction, only the per-cell results copied back. Exercise 2 points here |
| KLT | `src/feature_tracker/feature_tracker_cuda_tools.cu`, `feature_tracker_gpu.cpp` | one warp per feature, what inverse compositional KLT precomputes per template, the `__shfl_xor_sync` butterfly, mapped (zero-copy) host memory for the results |
| Helpers | `include/vilib/cuda_common.h`, `src/cuda_common.cpp`, `src/timergpu.cpp` | error checks (compiled out, see below), CUDA event timing vs the exercise's host timing |

## Upstream issues the comments point out

None of these changes the exercise's results; they matter when you reuse vilib or change its
settings. The comments at each place give the details.

- **No CUDA error is reported.** `CMakeLists.txt` passes `-DNDEBUG` to both compilers, which turns
  `CUDA_API_CALL()`, `CUDA_KERNEL_CHECK()` and every `assert` into no-ops, and the allocations in
  `subframe.cpp` are not checked at all. A failed launch or allocation goes unnoticed.
- **FAST arc lengths below 9.** `fast_gpu.h` says the GPU pre-checks are skipped for arcs shorter
  than 9, but nothing skips them, and they are exact only for arcs of 8 or more. An arc length of
  7 or less silently loses corners (the guarding `assert` is compiled out).
- **Grid NMS on small cells.** The warp shuffles use the full-warp mask, but with 32-pixel cells a
  block has 16 threads or fewer from pyramid level 3 on, so the offset-16 step reads lanes that do
  not exist. The exercise detects on level 0 only.
- **KLT patch sizes.** `feature_tracker_options.h` asks for `32 % patch_size == 0`, but a 4x4 patch
  gives each lane 0 pixels and the level is silently skipped: only 8, 16 and 32 work.
  `klt_max_iter` is never read (the limit is the compile-time `FEATURE_TRACKER_MAX_ITERATION_COUNT`).
- **KLT host/device race.** `FeatureTrackerGPU::track()` rewrites a mapped host buffer that the
  template kernels of the previous call may still be reading; nothing in vilib waits in between.
  The exercise synchronizes after every `track()`, which avoids it.
- **Pools.** The pyramid pool keeps the size and level count of its first `init()` and does not
  check later calls; `SubframePool` checks `empty()` outside its lock (a race with several
  threads).
- **NEON pyramid, odd heights.** On AArch64, `pyramid_cpu.cpp` reads and writes one row past the
  image when the height is odd (e.g. the 240x135 level of a 1080p frame). x86 is not affected.
