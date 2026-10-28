# SIMD Acceleration Hands-on: Plain C++ vs OpenCV vs komrad36's AVX2/SSE Kernels

A feature-based SLAM front-end runs the same three kernels on every frame: convert to grayscale,
detect FAST corners, and compute each corner's orientation (the first half of ORB). This exercise
times each kernel with **easy_profiler** in three versions: plain C/C++, OpenCV, and
[komrad36](https://github.com/komrad36)'s hand-written intrinsics. It then checks that all versions
produce the same output.

| Kernel | Plain C/C++ (baseline) | OpenCV | komrad36 (ISA) |
|---|---|---|---|
| BGR → gray | per-pixel loop (the formula of komrad36's `RGB2Y_ref`) | `cv::cvtColor(COLOR_BGR2GRAY)` | [RGB2Y](https://github.com/komrad36/RGB2Y) (AVX2) |
| FAST-9 + non-max suppression | Rosten's original C code | `cv::FAST(TYPE_9_16)` | [KFAST](https://github.com/komrad36/KFAST) (AVX2 + BMI1) |
| Orientation (intensity centroid, 7x7) | ORB's `IC_Angle` loop from OpenCV's `orb.cpp` | same loop: `cv::ORB` has no faster version | [FeatureAngle](https://github.com/komrad36/FeatureAngle) (SSE4.1) |

The plain C/C++ code (`src/plain_kernels.cpp` and Rosten's C files) is compiled twice:

- **`plain_O3`**, with `-O3`. This is an ordinary portable x86-64 Release build, and it is the
  reference: every table reports speed-ups relative to it. GCC may still auto-vectorize simple
  loops with SSE2.
- **`plain_native`**, with `-O3 -march=native`. GCC may use every extension of the build CPU
  (AVX2, AVX-512, …). This row shows how far the compiler alone gets with the same code. The
  result depends on the CPU the binary was built on.

`cv::cvtColor` and `cv::FAST` are not scalar code. OpenCV already vectorizes them with its universal
intrinsics, dispatched to AVX2 at runtime, and `cvtColor` also uses OpenCV's thread pool. Each
kernel runs single-threaded (`_1T`). The multi-threaded variants (`_MT`) run where the
implementation has one.

---

## Project Structure

```
part5_ch03_06/
├── README.md
├── CMakeLists.txt
├── Dockerfile
├── data/
│   ├── img_rgb.png                   # benchmark image (see data/README.md for source and license)
│   └── README.md
├── include/
│   ├── benchmark_common.hpp          # arguments, padded buffers, AVX2 check
│   ├── komrad36_compat.hpp           # GCC shims for komrad36's MSVC-era headers
│   ├── plain_kernels.hpp             # plain C/C++ baselines (plain_o3 / plain_native)
│   ├── rosten_prefix.h               # per-copy symbol names for Rosten's C code
│   └── profile_summary.hpp           # reads a .prof back and prints the timing table
├── src/
│   └── plain_kernels.cpp             # plain loops, compiled with -O3 and with -O3 -march=native
├── examples/
│   ├── rgb2y_benchmark.cpp           # plain loop vs cv::cvtColor vs RGB2Y
│   ├── kfast_benchmark.cpp           # Rosten vs cv::FAST vs KFAST
│   ├── feature_angle_benchmark.cpp   # ORB IC_Angle loop vs FeatureAngle
│   └── prof_summary.cpp              # timing table for any .prof file
└── third_party/komrad36/             # vendored kernels + licenses (see its README)
```

---

## Requirements

- **x86-64 CPU with AVX2 and BMI1** (Intel Haswell / AMD Zen 1 or newer). komrad36's kernels are
  x86 intrinsics, so ARM boards (Jetson, DGX Spark, Raspberry Pi) cannot run them. Porting a
  kernel to NEON is one of the exercises below.
- **OpenCV 4.x** and **easy_profiler**, both included in `slam:base`.

---

## Build

```bash
# Local
mkdir build && cd build
cmake ..
make -j4

# Docker (needs the slam:base image built from ../Dockerfile)
docker build . -t slam_zero_to_hero:part5_ch03_06
```

---

## Run

Every benchmark takes an optional image path and iteration count. The defaults are
`data/img_rgb.png` and 200 profiled calls per variant, each after 20 unprofiled warm-up calls.

### Local

```bash
./build/rgb2y_benchmark
./build/kfast_benchmark
./build/feature_angle_benchmark
./build/kfast_benchmark /path/to/image.png 500     # any image, 500 iterations

# Print the table again, e.g. for a .prof recorded on another machine
./build/prof_summary rgb2y_benchmark.prof kfast_benchmark.prof
```

### Docker

```bash
docker run -it --rm slam_zero_to_hero:part5_ch03_06

# Inside container
./rgb2y_benchmark
./kfast_benchmark
./feature_angle_benchmark
```

Each program records every call in an `EASY_BLOCK` and writes `<benchmark>.prof` to the current
directory. It then reads that file back and prints the median, mean, and min of each block, so the
table contains easy_profiler's own measurements. The `vs plain_O3` column is the median of
`<kernel>/plain_O3_1T` divided by the block median, so 1.00x is the plain C/C++ code built with
`-O3`, and larger numbers are faster.

`slam:base` does not include `profiler_gui`. To browse the timeline, build
[easy_profiler](https://github.com/yse/easy_profiler) with Qt on the host and open the `.prof` file.

---

## Results

Measured in `slam:base` (GCC 13.3, OpenCV 4.12.0) on an AMD Ryzen 9 7950X (16 cores / 32 threads,
AVX-512), with `data/img_rgb.png` (1226x370) and 200 calls per block:

| Block | Median [µs] | vs plain_O3 |
|---|---:|---:|
| `RGB2Y/plain_O3_1T` | 630.6 | 1.00x |
| `RGB2Y/plain_native_1T` | 195.8 | 3.22x |
| `RGB2Y/OpenCV_1T` | 71.4 | 8.83x |
| `RGB2Y/komrad36_1T` | 46.7 | **13.50x** |
| `RGB2Y/OpenCV_MT` | 14.9 | 42.22x |
| `RGB2Y/komrad36_MT` | 434.7 | 1.45x |
| `FAST/plain_O3_1T` | 2000.9 | 1.00x |
| `FAST/plain_native_1T` | 2007.2 | 1.00x |
| `FAST/OpenCV_1T` | 363.2 | 5.51x |
| `FAST/KFAST_1T` | 285.0 | **7.02x** |
| `FAST/KFAST_MT` | 495.5 | 4.04x |
| `Angle/plain_O3_1T` | 50.2 | 1.00x |
| `Angle/plain_native_1T` | 46.4 | 1.08x |
| `Angle/komrad36_1T` | 15.9 | **3.15x** |

Verification printed by the same runs:
- **RGB2Y**: `plain_native`, komrad36 1T, and komrad36 MT match `plain_O3` exactly. OpenCV
  differs by ±1 on 30% of the pixels, because it rounds `B·w_b + G·w_g + R·w_r` once, while the
  others round each product.
- **FAST**: every detector returns the same 2683 corners.
- **Angle**: `plain_native` matches `plain_O3` exactly, and FeatureAngle is within 2.3e-5°. The two
  `atan2` polynomial approximations account for the difference.

### What the numbers say

1. **Hand-written SIMD vs plain `-O3` C/C++: 13.5x, 7.0x, 3.2x.** At `-O3`, GCC leaves the RGB2Y
   loop scalar. `-fopt-info-vec-optimized` reports only `IC_Angle`'s two inner loops, vectorized
   with 16-byte SSE2 vectors.
2. **The compiler alone gets part of the way, and only on simple loops.** With `-march=native`,
   GCC vectorizes the RGB2Y loop with 32- and 64-byte (AVX-512) vectors, a 3.2x gain. Rosten's
   decision tree (branches) and `IC_Angle`'s short loops gain almost nothing (1.00x, 1.08x). Even
   with AVX-512, the plain RGB2Y loop is 4x slower than komrad36's AVX2 kernel. The loop widens
   every channel to `double`, 8 per 512-bit register, while RGB2Y works in `int16`, 16 per 256-bit
   register.
3. **komrad36 vs OpenCV: 1.5x (RGB2Y) and 1.3x (FAST).** OpenCV is already vectorized, so the
   remaining gain comes from data layout tricks (see below). For orientation, `cv::ORB` runs the
   plain `IC_Angle` loop, so FeatureAngle's 3.2x is also its gain over OpenCV.
4. **Spawning threads per call hurts on small images.** komrad36's `_MT` variants start up to 32
   `std::async` threads on every call. Creating those threads takes longer than the kernel itself on
   a 1226x370 frame, so `RGB2Y/komrad36_MT` is 9x slower than `RGB2Y/komrad36_1T`. OpenCV keeps a
   persistent thread pool, and its `_MT` cuts 71 µs to 15 µs. On a larger frame, threads do pay
   off. On komrad36's own 1920x1080 test image, we measured KFAST_MT at 724 µs against 3968 µs for
   `cv::FAST`.
5. **Why `-O3` and not `-O0` as the baseline.** In the same setup, a `-O0` build of the plain code
   is 1.8x (RGB2Y), 1.7x (FAST), and 2.9x (Angle) slower than `-O3`. Measuring against it would
   credit SIMD with work that any optimizing build already does.

---

## How the kernels work

**RGB2Y**: Loads 30 bytes (10 BGR pixels) and widens them to 16-bit lanes. `_mm256_mulhrs_epi16`
multiplies by Q15 fixed-point weights with rounding, so no float math is needed. `_mm256_shuffle_epi8`
(`pshufb`) gathers each pixel's B, G, and R products into one lane, and saturating adds produce
10 gray pixels per step.

**KFAST**: Tests 32 adjacent pixels at once. x86 has no unsigned byte compare, so both sides are
XORed with `0x80` and compared with `_mm256_cmpgt_epi8`. The four compass points (1, 5, 9, 13)
reject most candidates early, and `_mm256_movemask_epi8` packs the survivors into a 32-bit mask.
Survivors run the full 16-pixel circle test as a branch-free run-length count. `_tzcnt_u32` and
`_blsr_u32` (BMI1) then visit each set bit. Non-max suppression keeps only three rows of scores.

**FeatureAngle**: A 7x7 patch row is 7 bytes, so one 8-byte load widened to 8 x `int16` holds a
whole row. The circular mask's x and y weights are constant vectors, so `_mm_mullo_epi16` plus adds
accumulate both moments `m10` and `m01` without branches or a `u_max` table. `_mm_hadd_epi16`
reduces the lanes, and a polynomial `atan2` returns radians. ORB's `IC_Angle` walks the same
circular patch pixel by pixel. It returns the same moments.

### Two things to know before using these kernels

- **They read and write past the end of the image.** Loads and stores are 16 or 32 bytes wide, so
  the last pixels of the last row pull in bytes beyond the buffer. `createPadded()` in
  `include/benchmark_common.hpp` gives every image one spare row for this. Exercise 1 shows what
  happens without it.
- **Upstream RGB2Y crashes on KITTI-height images when multi-threaded.** Its thread split always
  launched `hw_concur - 1` chunks of `ceil(rows / hw_concur)` rows. A 370-row frame on 32 threads
  got 22 chunks of 17 rows (374 rows), so the last thread started at row 374 and segfaulted. The
  vendored copy has a one-line fix, documented in
  [third_party/komrad36/README.md](third_party/komrad36/README.md).

---

## Exercises

1. **Over-reads under AddressSanitizer.** In `createPadded()`, change `rows + 1` to `rows` and build
   with `-fsanitize=address` (`cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"`).
   ASan stops `rgb2y_benchmark` at a 16-byte read past the image in RGB2Y's `process()`
   (`heap-buffer-overflow`), and `kfast_benchmark` at a 32-byte read in KFAST's `processCols()`.
   Why does `feature_angle_benchmark` pass? Work out how far its 8-byte load can reach.
2. **What did the compiler do to the plain loops?** Add `-fopt-info-vec-optimized` to
   `PLAIN_FLAGS_o3` and `PLAIN_FLAGS_native` in `CMakeLists.txt` to see which loops GCC
   vectorized, and with how wide vectors. Then add a third copy built with `-O3 -mavx2`, the
   instruction set komrad36 uses, for a like-for-like comparison with RGB2Y.
3. **Threads without per-call spawning.** Rewrite the `_MT` path of `RGB2Y` or `KFAST` with
   `cv::parallel_for_`, which uses OpenCV's persistent pool, and compare it with `std::async`.
4. **ORB's real patch size.** ORB uses a 31x31 patch (`half_k = 15`). Change `kHalfPatch` in
   `feature_angle_benchmark.cpp`. `IC_Angle` scales directly. Then write an AVX2 version of
   FeatureAngle for 31 columns. Note that the moments no longer fit in `int16` at that size, so
   accumulate them in 32-bit lanes.
5. **ARM.** Port FeatureAngle to NEON (`arm_neon.h`). `vmovl_u8`, `vmulq_s16`, and `vaddvq_s16`
   map closely to the SSE calls. Run it on a Jetson or DGX Spark, and compare the `.prof` files
   with `prof_summary`.

---

## References

- komrad36: [KFAST](https://github.com/komrad36/KFAST), [RGB2Y](https://github.com/komrad36/RGB2Y), [FeatureAngle](https://github.com/komrad36/FeatureAngle) (MIT)
- E. Rosten and T. Drummond, [Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf), ECCV 2006
- E. Rublee et al., ORB: An efficient alternative to SIFT or SURF, ICCV 2011 (intensity-centroid orientation)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [easy_profiler](https://github.com/yse/easy_profiler)
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (sample image, CC BY-NC-SA 3.0)
