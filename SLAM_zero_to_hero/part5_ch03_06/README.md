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
intrinsics, dispatched to AVX2 at runtime.

Every kernel runs single-threaded (`_1T`), and the fast ones also multi-threaded (`_MT<threads>`,
for 4, 8, 16, … up to the hardware thread count):

- **`OpenCV_MT<n>`**: `cv::cvtColor` on OpenCV's own threads, with `cv::setNumThreads(n)`.
  `cv::FAST` and ORB's orientation step have no multi-threaded version.
- **`komrad36_MT<n>` / `KFAST_MT<n>`**: komrad36's kernels split into `n` stripes (rows for RGB2Y
  and FAST, keypoints for the angle) on komrad36's
  [FastThreadPool](https://github.com/komrad36/FastThreadPool). `include/fast_parallel_for.hpp`
  wraps it with persistent workers that spin while they wait. The kernels' own built-in
  multi-threaded mode is not used, because it starts new threads on every call (see
  [third_party/komrad36/README.md](third_party/komrad36/README.md#multi-threading)).

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
│   ├── fast_parallel_for.hpp         # parallel-for on komrad36's FastThreadPool (_MT variants)
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
└── third_party/komrad36/             # vendored kernels, FastThreadPool + licenses (see its README)
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

Measured in the `slam_zero_to_hero:part5_ch03_06` image (GCC 13.3, OpenCV 4.12.0) on an AMD Ryzen 9 7950X
(16 cores / 32 threads, AVX-512), with `data/img_rgb.png` (1920x1080) and 200 calls per block. All values are
medians in µs.

**Single-threaded**, with the speed-up over `plain_O3` in parentheses:

| Kernel | `plain_O3` | `plain_native` | OpenCV | komrad36 |
|---|---:|---:|---:|---:|
| RGB2Y | 2903.3 | 890.0 (3.26x) | 316.6 (9.17x) | **208.0 (13.96x)** |
| FAST | 6559.9 | 6552.4 (1.00x) | 1289.3 (5.09x) | **1006.9 (6.52x)** |
| Angle | 161.0 | 188.3 (0.86x) | same as `plain_O3` (`cv::ORB` runs this loop) | **48.5 (3.32x)** |

**Multi-threaded**, by thread count. OpenCV runs `cv::cvtColor` on its own threads. komrad36's kernels run on
FastThreadPool:

| Block | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| `RGB2Y/OpenCV_MT<n>` | 87.5 | 52.1 | 48.7 | 41.4 |
| `RGB2Y/komrad36_MT<n>` | 58.4 | **29.5** | 29.7 | 52.1 |
| `FAST/KFAST_MT<n>` | 375.3 | 203.0 | 118.3 | **96.7** |
| `Angle/komrad36_MT<n>` | 13.6 | **9.7** | 11.3 | 17.4 |
| Cores busy while idle (FastThreadPool) | 3.0 | 7.0 | 15.0 | 31.0–31.9 |

Verification printed by the same runs:
- **RGB2Y**: `plain_native`, komrad36 1T, and komrad36 MT at every thread count match `plain_O3`
  exactly. OpenCV differs by ±1 on 31% of the pixels, because it rounds `B·w_b + G·w_g + R·w_r` once,
  while the others round each product.
- **FAST**: every detector, at every thread count, returns the same 6821 corners.
- **Angle**: `plain_native` matches `plain_O3` exactly, and FeatureAngle is within 3.1e-5°. The two
  `atan2` polynomial approximations account for the difference. FeatureAngle MT matches its 1T run.

### What the numbers say

1. **Hand-written SIMD vs plain `-O3` C/C++: 14.0x, 6.5x, 3.3x on one thread.** At `-O3`, GCC leaves
   the RGB2Y loop scalar. `-fopt-info-vec-optimized` reports only `IC_Angle`'s two inner loops,
   vectorized with 16-byte SSE2 vectors.
2. **The compiler alone gets part of the way, and only on simple loops.** With `-march=native`,
   GCC vectorizes the RGB2Y loop with 32- and 64-byte (AVX-512) vectors, a 3.3x gain. Rosten's
   decision tree gains nothing, and `IC_Angle`'s short 3–7 pixel loops get 14% slower. Even with
   AVX-512, the plain RGB2Y loop is 4.3x slower than komrad36's AVX2 kernel. The loop widens
   every channel to `double`, 8 per 512-bit register, while RGB2Y works in `int16`, 16 per 256-bit
   register.
3. **komrad36 vs OpenCV on one thread: 1.5x (RGB2Y) and 1.3x (FAST).** OpenCV is already
   vectorized, so the remaining gain comes from data layout tricks (see below). For orientation,
   `cv::ORB` runs the plain `IC_Angle` loop, so FeatureAngle's 3.3x is also its gain over OpenCV.
4. **With threads, komrad36 still wins, at a price.**
   - **RGB2Y**: komrad36 on 8 threads (29.5 µs) is 1.4x faster than OpenCV's best (41.4 µs on 32
     threads), with a quarter of the threads.
   - **FAST**: KFAST keeps scaling up to 32 threads. At 96.7 µs it is 10.4x faster than its single
     thread and 13.3x faster than `cv::FAST`, which has no multi-threaded version.
   - **Angle**: 48.5 µs of single-threaded work is too little to split far. 8 threads are fastest
     (9.7 µs), and more threads only add dispatch overhead.
   - **Past 16 threads**, the 7950X has no more physical cores, and RGB2Y and the angle get slower.
     A likely cause is that spinning workers on the same physical core take execution resources from
     the working ones.
   - **The price**: FastThreadPool's workers spin, so a pool of `n` threads keeps `n - 1` cores at
     100% even with no work ("Cores busy while idle").
5. **Why `-O3` and not `-O0` as the baseline.** In the same setup, a `-O0` build of the plain code
   is 1.8x (RGB2Y), 1.8x (FAST), and 2.9x (Angle) slower than `-O3`. Measuring against it would
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
- **FastThreadPool's workers never sleep.** They spin on the queue, so a pool of `n` threads keeps
  `n - 1` cores at 100% for as long as it exists, with or without work. Each benchmark prints this
  ("Cores busy while idle"). In a SLAM system, these spinning workers would take cores away from
  tracking, mapping, and loop closing.

---

## Exercises

1. **Over-reads under AddressSanitizer.** In `createPadded()`, change `rows + 1` to `rows` and build
   with `-fsanitize=address` (`cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"`).
   ASan stops `kfast_benchmark` at a 32-byte read past the image in KFAST (`_KFAST`).
   `rgb2y_benchmark` and `feature_angle_benchmark` pass on the 1920x1080 image. Work out how far
   past the last pixel their loads reach at this width, and why `cv::Mat`'s allocator hides it.
2. **What did the compiler do to the plain loops?** Add `-fopt-info-vec-optimized` to
   `PLAIN_FLAGS_o3` and `PLAIN_FLAGS_native` in `CMakeLists.txt` to see which loops GCC
   vectorized, and with how wide vectors. Then add a third copy built with `-O3 -mavx2`, the
   instruction set komrad36 uses, for a like-for-like comparison with RGB2Y.
3. **Workers that sleep.** FastThreadPool's README lists "sleep instead of spin after a while" as
   a TODO. In `FastParallelFor::workerLoop()`, put a worker to sleep after it has spun for a while
   without finding work, and wake it in `run()` (e.g. with a `std::condition_variable`).
   Compare the `_MT` medians and "Cores busy while idle" before and after.
4. **ORB's real patch size.** ORB uses a 31x31 patch (`half_k = 15`). Change `kHalfPatch` in
   `feature_angle_benchmark.cpp`. `IC_Angle` scales directly. Then write an AVX2 version of
   FeatureAngle for 31 columns. Note that the moments no longer fit in `int16` at that size, so
   accumulate them in 32-bit lanes.
5. **ARM.** Port FeatureAngle to NEON (`arm_neon.h`). `vmovl_u8`, `vmulq_s16`, and `vaddvq_s16`
   map closely to the SSE calls. Run it on a Jetson or DGX Spark, and compare the `.prof` files
   with `prof_summary`.

---

## References

- komrad36: [KFAST](https://github.com/komrad36/KFAST), [RGB2Y](https://github.com/komrad36/RGB2Y), [FeatureAngle](https://github.com/komrad36/FeatureAngle), [FastThreadPool](https://github.com/komrad36/FastThreadPool) (MIT)
- E. Rosten and T. Drummond, [Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf), ECCV 2006
- E. Rublee et al., ORB: An efficient alternative to SIFT or SURF, ICCV 2011 (intensity-centroid orientation)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [easy_profiler](https://github.com/yse/easy_profiler)
- Benchmark image: LG Electronics, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) (see [data/README.md](data/README.md))
