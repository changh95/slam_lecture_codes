# SIMD Acceleration Hands-on: Plain C++ vs OpenCV vs komrad36's AVX2/SSE Kernels and Their NEON Ports

A feature-based SLAM front-end runs the same three kernels on every frame: convert to grayscale,
detect FAST corners, and compute each corner's orientation (the first half of ORB). This exercise
times each kernel with **easy_profiler** in three versions: plain C/C++, OpenCV, and
[komrad36](https://github.com/komrad36)'s hand-written intrinsics. It then checks that all versions
produce the same output. On an AArch64 CPU such as a Jetson, the same benchmarks run this exercise's
NEON ports of komrad36's kernels (`neon/`) in place of the AVX2/SSE originals.

| Kernel | Plain C/C++ (baseline) | OpenCV | komrad36, x86-64 (ISA) | NEON port, AArch64 |
|---|---|---|---|---|
| BGR → gray | per-pixel loop (the formula of komrad36's `RGB2Y_ref`) | `cv::cvtColor(COLOR_BGR2GRAY)` | [RGB2Y](https://github.com/komrad36/RGB2Y) (AVX2) | [`neon/RGB2Y_neon.h`](neon/RGB2Y_neon.h) |
| FAST-9 + non-max suppression | Rosten's original C code | `cv::FAST(TYPE_9_16)` | [KFAST](https://github.com/komrad36/KFAST) (AVX2 + BMI1) | [`neon/KFAST_neon.h`](neon/KFAST_neon.h) |
| Orientation (intensity centroid, 7x7) | ORB's `IC_Angle` loop from OpenCV's `orb.cpp` | same loop: `cv::ORB` has no faster version | [FeatureAngle](https://github.com/komrad36/FeatureAngle) (SSE4.1) | [`neon/FeatureAngle_neon.h`](neon/FeatureAngle_neon.h) |

The plain C/C++ code (`src/plain_kernels.cpp` and Rosten's C files) is compiled twice:

- **`plain_O3`** (the `-O3` column), with `-O3`. This is an ordinary portable Release build, and it is the
  reference: every table reports speed-ups relative to it. GCC may still auto-vectorize simple
  loops with SSE2 (x86-64) or NEON (AArch64, where NEON is part of the base instruction set).
- **`plain_native`** (the `-O3 -march=native` column). GCC may use every extension of the build CPU
  (AVX2, AVX-512, … on x86-64; e.g. the dot-product instructions or SVE on newer Arm cores). This
  column shows how far the compiler alone gets with the same code. The result depends on the CPU the
  binary was built on.

`cv::cvtColor` and `cv::FAST` are not scalar code. OpenCV already vectorizes them with its universal
intrinsics: AVX2, dispatched at runtime, on x86-64, and NEON on AArch64.

Every kernel runs single-threaded (`_1T`) and multi-threaded (`_MT<threads>`, for 4, 8, 16, … up to
the hardware thread count):

- **`OpenCV_MT<n>`**: `cv::cvtColor` on OpenCV's own threads, with `cv::setNumThreads(n)`.
  `cv::FAST` and ORB's orientation step have no multi-threaded version.
- **`komrad36_MT<n>`**: komrad36's kernels split into `n` stripes (rows for RGB2Y
  and FAST, keypoints for the angle) on komrad36's
  [FastThreadPool](https://github.com/komrad36/FastThreadPool). `include/fast_parallel_for.hpp`
  wraps it with persistent workers that spin while they wait. KFAST's stripes overlap by 4 rows
  so that non-max suppression is exact at the seams, and they are capped at one per 16 rows.
  RGB2Y converts each stripe's last row into a scratch row (see "Things to know" below). The
  kernels' own built-in multi-threaded mode is not used, because it starts new threads on every
  call (see [third_party/komrad36/README.md](third_party/komrad36/README.md#multi-threading)).

On AArch64 the `komrad36` variant is the NEON port: its blocks are named `NEON_1T` and `NEON_MT<n>`,
and the tables have a `NEON` column. The NEON port of RGB2Y writes each row exactly, so its stripes need
no scratch row.

---

## Project Structure

```
part5_ch03_06/
├── README.md
├── CMakeLists.txt
├── Dockerfile                        # x86-64 image, FROM slam:base
├── Dockerfile.jetson                 # arm64 image (Jetson, DGX Spark), Ubuntu 22.04 + OpenCV from apt
├── data/
│   ├── img_rgb.png                   # benchmark image (see data/README.md for source and license)
│   └── README.md
├── include/
│   ├── benchmark_common.hpp          # arguments, padded buffers, CPU check, SIMD_VARIANT
│   ├── fast_parallel_for.hpp         # parallel-for on komrad36's FastThreadPool (_MT variants)
│   ├── komrad36_compat.hpp           # GCC shims for komrad36's MSVC-era headers
│   ├── plain_kernels.hpp             # plain C/C++ baselines (plain_o3 / plain_native)
│   ├── rosten_prefix.h               # per-copy symbol names for Rosten's C code
│   └── profile_summary.hpp           # reads a .prof back and prints the two comparison tables
├── src/
│   └── plain_kernels.cpp             # plain loops, compiled with -O3 and with -O3 -march=native
├── examples/
│   ├── rgb2y_benchmark.cpp           # plain loop vs cv::cvtColor vs RGB2Y
│   ├── kfast_benchmark.cpp           # Rosten vs cv::FAST vs KFAST
│   ├── feature_angle_benchmark.cpp   # ORB IC_Angle loop vs FeatureAngle
│   └── prof_summary.cpp              # comparison tables for any .prof file
├── neon/                             # NEON ports of komrad36's kernels, same interface (AArch64)
│   ├── RGB2Y_neon.h
│   ├── KFAST_neon.h
│   └── FeatureAngle_neon.h
└── third_party/komrad36/             # komrad36's 4 repos, kernel headers commented (see its README)
```

---

## Requirements

- **x86-64 CPU with AVX2 and BMI1** (Intel Haswell / AMD Zen 1 or newer) for komrad36's kernels,
  **or an AArch64 CPU** (Jetson Orin / Xavier, DGX Spark, Raspberry Pi 4/5, …) for the NEON ports.
  CMake picks the kernels from the CPU it builds on.
- **OpenCV 4.x** and **easy_profiler**: included in `slam:base` (x86-64) and installed by
  `Dockerfile.jetson` (arm64).

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

The Dockerfile copies this folder into the image, including komrad36's code in `third_party/komrad36/`.
Nothing is cloned or downloaded at build time.

**Jetson (or any arm64 Linux machine).** `slam:base` is built for x86-64 only, so `Dockerfile.jetson`
starts from Ubuntu 22.04 (the userland of JetPack 6), installs OpenCV 4.5.4 from Ubuntu's packages, and
builds easy_profiler at the commit `slam:base` uses, so `.prof` files move between the two. The image
uses only the CPU, so it needs no NVIDIA container runtime, and it should work on JetPack 5 as well.
Build the image on the machine that will run it: `plain_native` is compiled with `-march=native` for the
CPU it is built on, so an image built on an Orin may not run on a Xavier, and one built under QEMU on an
x86-64 PC gets no CPU-specific flags at all.

```bash
# On the Jetson
docker build -f Dockerfile.jetson -t slam_zero_to_hero:part5_ch03_06-jetson .

# On an x86-64 PC, emulated (needs qemu-user-static; far too slow for timing, fine for checking)
docker build --platform linux/arm64 -f Dockerfile.jetson -t slam_zero_to_hero:part5_ch03_06-jetson .
```

A local build on the Jetson works the same way as on x86-64 once OpenCV (`sudo apt install libopencv-dev`)
and easy_profiler are installed.

---

## Run

Each demo takes an optional image path and iteration count: `<demo> [image] [iterations]`. The
defaults are `data/img_rgb.png` and 200 profiled calls per variant, each variant after 20
unprofiled warm-up calls.

Each demo records every call in an `EASY_BLOCK`, writes `<demo>.prof` to the current directory,
reads that file back, and prints:

1. **Verification**: every variant's output compared with the plain `-O3` result. FeatureAngle MT is
   compared with FeatureAngle 1T, because its angles differ from OpenCV's by float rounding.
2. **Cores busy while idle**: what FastThreadPool's spinning workers cost
3. **Two comparison tables** of easy_profiler's medians in µs, with the speed-up over single-threaded
   `-O3` in parentheses
   - **[1] Single thread**: `-O3`, `-O3 -march=native`, OpenCV, komrad36
   - **[2] Multi thread**: OpenCV and komrad36 on 4, 8, 16, … threads. `-` means the library has
     no multi-threaded version.

The outputs below were recorded on an AMD Ryzen 9 7950X (16 cores / 32 threads, AVX-512) in the
`slam_zero_to_hero:part5_ch03_06` image.

### 1. RGB2Y: BGR → gray

```bash
docker run --rm slam_zero_to_hero:part5_ch03_06 ./rgb2y_benchmark
# local build: ./build/rgb2y_benchmark
```

```text
========================================
RGB2Y: BGR -> gray (easy_profiler)
========================================
Image: /workspace/part5_ch03_06/data/img_rgb.png (1920x1080)
Iterations per variant: 200, OpenCV threads: 32, std::thread::hardware_concurrency: 32

Verification (reference: plain -O3 loop):
  plain native: 0 pixels differ (max |diff| = 0)
  komrad36 1T : 0 pixels differ (max |diff| = 0)
  OpenCV 1T   : 649801 pixels differ (max |diff| = 1)
  OpenCV MT   : 649801 pixels differ (max |diff| = 1)
  komrad36 MT : 0 pixels differ (summed over all thread counts)
  (OpenCV rounds B*w_b + G*w_g + R*w_r once; the others round each product, so +-1 is expected)

FastThreadPool workers spin even without work. Cores busy while idle:
  4 threads: 3.0 cores
  8 threads: 7.0 cores
  16 threads: 15.0 cores
  32 threads: 31.0 cores

=== easy_profiler summary: rgb2y_benchmark.prof ===
median of 200 calls in us; (x) = speed-up over single-threaded -O3

[1] Single thread
Kernel                   -O3   -O3 -march=native              OpenCV            komrad36
----------------------------------------------------------------------------------------
RGB2Y         2903.1 (1.00x)       889.3 (3.26x)       314.1 (9.24x)      208.9 (13.90x)

[2] Multi thread
Kernel   Threads              OpenCV            komrad36
--------------------------------------------------------
RGB2Y          4       87.3 (33.25x)       55.2 (52.59x)
RGB2Y          8       50.3 (57.70x)       40.4 (71.90x)
RGB2Y         16       49.0 (59.20x)      26.0 (111.84x)
RGB2Y         32       45.6 (63.60x)       53.6 (54.20x)

Profile saved to: rgb2y_benchmark.prof (open with profiler_gui)
```

### 2. KFAST: FAST-9 corners with non-max suppression

```bash
docker run --rm slam_zero_to_hero:part5_ch03_06 ./kfast_benchmark
# local build: ./build/kfast_benchmark
```

```text
========================================
KFAST: FAST-9 + NMS (easy_profiler)
========================================
Image: /workspace/part5_ch03_06/data/img_rgb.png (1920x1080)
Threshold: 20, iterations per variant: 200, std::thread::hardware_concurrency: 32

Verification (reference: plain -O3, 6821 corners):
  plain native: 6821 corners, identical to plain -O3
  OpenCV      : 6821 corners, identical to plain -O3
  KFAST 1T    : 6821 corners, identical to plain -O3
  KFAST MT    : identical to plain -O3 at every thread count

FastThreadPool workers spin even without work. Cores busy while idle:
  4 threads: 3.0 cores
  8 threads: 6.9 cores
  16 threads: 15.0 cores
  32 threads: 30.8 cores

=== easy_profiler summary: kfast_benchmark.prof ===
median of 200 calls in us; (x) = speed-up over single-threaded -O3

[1] Single thread
Kernel                   -O3   -O3 -march=native              OpenCV            komrad36
----------------------------------------------------------------------------------------
FAST          6540.4 (1.00x)      6536.6 (1.00x)      1282.2 (5.10x)      1055.0 (6.20x)

[2] Multi thread
Kernel   Threads              OpenCV            komrad36
--------------------------------------------------------
FAST           4                   -      374.1 (17.48x)
FAST           8                   -      205.4 (31.84x)
FAST          16                   -      120.6 (54.25x)
FAST          32                   -       99.9 (65.47x)

Profile saved to: kfast_benchmark.prof (open with profiler_gui)
```

### 3. FeatureAngle: keypoint orientation

```bash
docker run --rm slam_zero_to_hero:part5_ch03_06 ./feature_angle_benchmark
# local build: ./build/feature_angle_benchmark
```

```text
========================================
FeatureAngle: keypoint orientation (easy_profiler)
========================================
Image: /workspace/part5_ch03_06/data/img_rgb.png (1920x1080)
Keypoints: 6821 (FAST threshold 20), iterations per variant: 200

Verification (reference: plain -O3 IC_Angle):
  plain native: mean |diff| = 0 deg, max |diff| = 0 deg
  komrad36    : mean |diff| = 5.13992e-06 deg, max |diff| = 3.05176e-05 deg
  komrad36 MT : identical to komrad36 1T at every thread count
  (same atan2 polynomial as cv::fastAtan2, in radians instead of degrees: the difference is float rounding)

FastThreadPool workers spin even without work. Cores busy while idle:
  4 threads: 3.0 cores
  8 threads: 7.0 cores
  16 threads: 15.0 cores
  32 threads: 31.0 cores

=== easy_profiler summary: feature_angle_benchmark.prof ===
median of 200 calls in us; (x) = speed-up over single-threaded -O3

[1] Single thread
Kernel                   -O3   -O3 -march=native              OpenCV            komrad36
----------------------------------------------------------------------------------------
Angle          151.9 (1.00x)       185.2 (0.82x)                   -        48.4 (3.14x)

[2] Multi thread
Kernel   Threads              OpenCV            komrad36
--------------------------------------------------------
Angle          4                   -       14.3 (10.62x)
Angle          8                   -       12.6 (12.02x)
Angle         16                   -       11.1 (13.69x)
Angle         32                   -        16.9 (9.01x)

Profile saved to: feature_angle_benchmark.prof (open with profiler_gui)
```

### 4. On a Jetson: the NEON ports

```bash
docker run --rm slam_zero_to_hero:part5_ch03_06-jetson ./rgb2y_benchmark
docker run --rm slam_zero_to_hero:part5_ch03_06-jetson ./kfast_benchmark
docker run --rm slam_zero_to_hero:part5_ch03_06-jetson ./feature_angle_benchmark
```

The output has the same layout, with `NEON` in place of `komrad36` / `KFAST`. No Jetson timings are
recorded in this README yet. The NEON code was checked by cross-compiling it for aarch64 (GCC 11, OpenCV
4.12) and running it under QEMU. Below is an excerpt of the verification lines it printed, with the
kernel names added on the left (emulated timings say nothing about a Jetson, so they are left out):

```text
RGB2Y      plain native: 0 pixels differ (max |diff| = 0)
           NEON 1T     : 0 pixels differ (max |diff| = 0)
           NEON MT     : 0 pixels differ (summed over all thread counts)
FAST       NEON 1T     : 6821 corners, identical to plain -O3
           NEON MT     : identical to plain -O3 at every thread count
Angle      NEON        : mean |diff| = 5.12881e-06 deg, max |diff| = 3.05176e-05 deg
           NEON MT     : identical to NEON 1T at every thread count
```

### Other images, other machines

```bash
# Any image, 500 calls per variant
docker run --rm -v /path/to/images:/images:ro slam_zero_to_hero:part5_ch03_06 ./kfast_benchmark /images/frame.png 500

# Keep the .prof files: run in a mounted folder (--rm deletes the container's own files)
docker run --rm -v "$PWD":/out -w /out slam_zero_to_hero:part5_ch03_06 /workspace/part5_ch03_06/build/rgb2y_benchmark

# Print the tables of a .prof again, e.g. one recorded on another machine
./build/prof_summary rgb2y_benchmark.prof kfast_benchmark.prof

# On a Jetson: the same demos in the arm64 image, .prof files kept in the current folder
docker run --rm -v "$PWD":/out -w /out slam_zero_to_hero:part5_ch03_06-jetson /workspace/part5_ch03_06/build/kfast_benchmark
```

A `.prof` recorded on the Jetson prints with the x86-64 image's `prof_summary` too (its SIMD column is
then `NEON`), so the two machines can be compared side by side.

`slam:base` does not include `profiler_gui`. To browse the timeline, build
[easy_profiler](https://github.com/yse/easy_profiler) with Qt on the host and open the `.prof` file.

---

## Results

The three runs above, combined. All values are medians in µs, with the speed-up over
single-threaded `-O3` in parentheses.

**[1] Single thread**

| Kernel | `-O3` | `-O3 -march=native` | OpenCV | komrad36 |
|---|---:|---:|---:|---:|
| RGB2Y | 2903.1 (1.00x) | 889.3 (3.26x) | 314.1 (9.24x) | **208.9 (13.90x)** |
| FAST | 6540.4 (1.00x) | 6536.6 (1.00x) | 1282.2 (5.10x) | **1055.0 (6.20x)** |
| Angle | 151.9 (1.00x) | 185.2 (0.82x) | - (`cv::ORB` runs the `-O3` loop) | **48.4 (3.14x)** |

**[2] Multi thread** (OpenCV: `cv::cvtColor` on its own threads; komrad36: FastThreadPool)

| Kernel | Threads | OpenCV | komrad36 |
|---|---:|---:|---:|
| RGB2Y | 4 | 87.3 (33.25x) | 55.2 (52.59x) |
| RGB2Y | 8 | 50.3 (57.70x) | 40.4 (71.90x) |
| RGB2Y | 16 | 49.0 (59.20x) | **26.0 (111.84x)** |
| RGB2Y | 32 | 45.6 (63.60x) | 53.6 (54.20x) |
| FAST | 4 | - | 374.1 (17.48x) |
| FAST | 8 | - | 205.4 (31.84x) |
| FAST | 16 | - | 120.6 (54.25x) |
| FAST | 32 | - | **99.9 (65.47x)** |
| Angle | 4 | - | 14.3 (10.62x) |
| Angle | 8 | - | 12.6 (12.02x) |
| Angle | 16 | - | **11.1 (13.69x)** |
| Angle | 32 | - | 16.9 (9.01x) |

While idle, FastThreadPool's workers kept 3.0, 6.9–7.0, 15.0, and 30.8–31.0 cores busy at 4, 8, 16,
and 32 threads.

Verification printed by the same runs:
- **RGB2Y**: `-O3 -march=native` and komrad36 (1T, and MT at every thread count) match `-O3`
  exactly. OpenCV differs by ±1 on 31% of the pixels, because it rounds `B·w_b + G·w_g + R·w_r`
  once, while the others round each product.
- **FAST**: every detector, at every thread count, returns the same 6821 corners.
- **Angle**: `-O3 -march=native` matches `-O3` exactly, and FeatureAngle is within 3.1e-5° (one float ulp at
  256–512°). Both use the same `atan2` polynomial, FeatureAngle in radians and `cv::fastAtan2` in degrees, so
  the difference is float rounding. FeatureAngle MT matches its 1T run.

Single-threaded medians repeat within about 5% between runs, except the `-O3` angle loop
(152–181 µs over three runs). Multi-threaded medians vary by up to a third: in four runs of
`rgb2y_benchmark`, komrad36's fastest thread count was 8 or 16 threads (23–30 µs), and OpenCV's was
32 threads (41–47 µs).

### What the numbers say

1. **Hand-written SIMD vs plain `-O3` C/C++: 13.9x, 6.2x, 3.1x on one thread.** At `-O3`, GCC leaves
   the RGB2Y loop scalar. `-fopt-info-vec-optimized` reports only `IC_Angle`'s two inner loops,
   vectorized with 16-byte SSE2 vectors.
2. **The compiler alone gets part of the way, and only on simple loops.** With `-march=native`,
   GCC vectorizes the RGB2Y loop with 32- and 64-byte (AVX-512) vectors, a 3.3x gain. Rosten's
   decision tree gains nothing, and `IC_Angle`'s short 3–7 pixel loops get slower (0.82x). Even with
   AVX-512, the plain RGB2Y loop is 4.3x slower than komrad36's AVX2 kernel. The loop widens
   every channel to `double`, 8 per 512-bit register, while RGB2Y works in `int16`, 16 per 256-bit
   register.
3. **komrad36 vs OpenCV on one thread: 1.5x (RGB2Y) and 1.2x (FAST).** OpenCV is already
   vectorized, so the remaining gain comes from data layout tricks (see below). For orientation,
   `cv::ORB` runs the plain `IC_Angle` loop, so FeatureAngle's 3.1x is also its gain over OpenCV.
4. **With threads, komrad36 still wins, at a price.**
   - **RGB2Y**: komrad36 on 16 threads (26.0 µs) is 1.8x faster than OpenCV's best (45.6 µs on 32
     threads), with half the threads. Across four runs, the gap was 1.4–2.0x.
   - **FAST**: komrad36's KFAST keeps scaling up to 32 threads. At 99.9 µs it is 10.6x faster than
     its single thread and 12.8x faster than `cv::FAST`, which has no multi-threaded version.
   - **Angle**: 48.4 µs of single-threaded work is too little to split far. 8–16 threads are
     fastest (11–13 µs), and 32 threads are slower again.
   - **Past 16 threads**, the 7950X has no more physical cores, and RGB2Y and the angle get slower.
     Two likely causes: spinning workers on the same physical core take execution resources from
     the working ones, and `run()` puts the `n` tasks on the queue one at a time, so dispatch cost
     grows with `n`. In a separate measurement, dispatching empty stripes took 5.5 µs on 16 threads
     and 15.6 µs on 32.
   - **The price**: FastThreadPool's workers spin, so a pool of `n` threads keeps `n - 1` cores at
     100% even with no work ("Cores busy while idle").
5. **Why `-O3` and not `-O0` as the baseline.** In a separate run on the same image, a `-O0` build of the plain code
   is 1.8x (RGB2Y), 1.8x (FAST), and 2.9x (Angle) slower than `-O3`. Measuring against it would
   credit SIMD with work that any optimizing build already does.

---

## How the kernels work

A summary of each kernel follows. The headers in `third_party/komrad36/` explain each design decision where it
is made, in comments that start with `[SLAM zero-to-hero]`: a "Design notes" block at the top of each file,
then notes next to the code.

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
reduces the lanes, and a polynomial `atan2` (the one `cv::fastAtan2` uses) returns radians. ORB's `IC_Angle` walks the same
circular patch pixel by pixel. It returns the same moments.

### NEON ports (AArch64)

The files in `neon/` keep komrad36's interface and algorithms, so the benchmarks only switch the
`#include`, and they return the same results: RGB2Y bit for bit, KFAST the same corners and scores, and
FeatureAngle the same moments. Each file opens with notes on how the x86 design maps to NEON. The main
differences:

| | komrad36 (x86-64) | NEON port (AArch64) |
|---|---|---|
| Register width | 256-bit AVX2 (128-bit SSE for FeatureAngle) | 128-bit: KFAST tests 16 pixels per step instead of 32 |
| RGB2Y: de-interleave BGR | `packus` + three `pshufb` masks + a cross-half add | `vld3q_u8`: one load splits 16 pixels into B, G and R registers |
| RGB2Y: Q15 product | `_mm256_mulhrs_epi16` | `vqrdmulhq_s16`, the same rounding |
| KFAST: unsigned byte compare | XOR both sides with `0x80`, then a signed compare | `vcgtq_u8` / `vcltq_u8` directly |
| KFAST: which lanes passed? | `movemask` → 32-bit mask, `tzcnt` / `blsr` | `vshrn_n_u16` → 64-bit integer with 4 bits per lane: `!= 0` for "any?", `ctz` / 4 for which |
| KFAST: max of the 16 arc scores | `32767 - phminposuw(32767 - x)` | `vmaxvq_s16`, one instruction |
| FeatureAngle: accumulate, sum lanes | `pmullw` + `paddw`; shuffle / `phaddw` / shuffle | `vmlaq_s16`; `vaddvq_s16`, one instruction |
| Row ends (RGB2Y, KFAST) | full-width loads and stores past the row (needs padding; RGB2Y spills into the next row) | the last step is moved back to end on the last pixel: nothing outside the row is read or written |

The NEON KFAST does not port KFAST's "advance by only 16 when the left half has no candidate"
shortcut (exercise 5).

### Things to know before using these kernels

- **komrad36's kernels read and write past the end of the image.** Loads and stores are 16 or 32
  bytes wide, so the last pixels of the last row pull in bytes beyond the buffer. `createPadded()` in
  `include/benchmark_common.hpp` gives every image one spare row for this. Exercise 1 shows what
  happens without it. The NEON ports stay inside the image, except that FeatureAngle (on both
  architectures) reads one byte right of its 7x7 patch.
- **komrad36's RGB2Y rows spill into the next row.** It stores 16 bytes for every 10 output pixels, so when
  the width modulo 10 is 1 to 5, each row writes up to 5 zero bytes into the start of the next row.
  One thread overwrites them when it converts that row. When two threads split the image, a
  stripe's last row would clobber the next stripe's first row, so `rgb2y_benchmark` converts that
  row into a scratch row and copies it back. komrad36's own multi-threaded RGB2Y has no such
  guard ([details](third_party/komrad36/README.md#multi-threading)).
- **FastThreadPool's workers never sleep.** They spin on the queue, so a pool of `n` threads keeps
  `n - 1` cores at 100% for as long as it exists, with or without work. Each benchmark prints this
  ("Cores busy while idle"). In a SLAM system, these spinning workers would take cores away from
  tracking, mapping, and loop closing.

---

## Exercises

1. **Over-reads under AddressSanitizer (x86-64).** In `createPadded()`, change `rows + 1` to `rows` and build
   with `-fsanitize=address` (`cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"`).
   ASan stops `kfast_benchmark` at a 32-byte read past the image in KFAST (`_KFAST`).
   `rgb2y_benchmark` and `feature_angle_benchmark` pass on the 1920x1080 image. Work out how far
   past the last pixel their loads reach at this width, and why `cv::Mat`'s allocator hides it.
   On AArch64 the NEON RGB2Y and KFAST read nothing past the image (they move their last step back
   instead), so only FeatureAngle's one extra byte right of the patch is left to find.
2. **What did the compiler do to the plain loops?** Add `-fopt-info-vec-optimized` to
   `PLAIN_FLAGS_o3` and `PLAIN_FLAGS_native` in `CMakeLists.txt` to see which loops GCC
   vectorized, and with how wide vectors. Then add a third copy built with `-O3 -mavx2`, the
   instruction set komrad36 uses, for a like-for-like comparison with RGB2Y.
3. **Workers that sleep.** FastThreadPool's [README](third_party/komrad36/FastThreadPool/README.md) lists "sleep instead of spin after a while" as
   a TODO. In `FastParallelFor::workerLoop()`, put a worker to sleep after it has spun for a while
   without finding work, and wake it in `run()` (e.g. with a `std::condition_variable`).
   Compare the `_MT` medians and "Cores busy while idle" before and after.
4. **ORB's real patch size.** ORB uses a 31x31 patch (`half_k = 15`). Change `kHalfPatch` in
   `feature_angle_benchmark.cpp`, and drop keypoints closer than 15 pixels to the border (FAST
   only keeps 3), or the patch reads leave the image. `IC_Angle` scales directly. Then write an AVX2 version of
   FeatureAngle for 31 columns. Note that the moments no longer fit in `int16` at that size, so
   accumulate them in 32-bit lanes.
5. **NEON on a Jetson.** Build `Dockerfile.jetson` on a Jetson, run the three benchmarks, and print
   their `.prof` files next to the x86-64 ones with `prof_summary`. Then try two changes to the
   NEON ports and compare the medians:
   - KFAST's shortcut: when the low 8 lanes of a window have no candidate, advance by 8 columns
     instead of 16. In `neon/KFAST_neon.h`, let `processWindow()` return how far to advance (after
     its "any candidate?" test) and use that in `_KFAST`'s column loop. Does it pay off with 16 lanes, as
     it does with 32?
   - Round once: rewrite `gray16()` in `neon/RGB2Y_neon.h` with `vmull_u8` / `vmlal_u8` on Q8 weights
     and one `vrshrn_n_u16`. It needs fewer instructions, but like OpenCV's result, it no longer
     matches the plain loop on every pixel. Check the verification lines.

---

## References

- komrad36: [KFAST](https://github.com/komrad36/KFAST), [RGB2Y](https://github.com/komrad36/RGB2Y), [FeatureAngle](https://github.com/komrad36/FeatureAngle), [FastThreadPool](https://github.com/komrad36/FastThreadPool) (MIT)
- E. Rosten and T. Drummond, [Machine learning for high-speed corner detection](https://www.edwardrosten.com/work/rosten_2006_machine.pdf), ECCV 2006
- E. Rublee et al., ORB: An efficient alternative to SIFT or SURF, ICCV 2011 (intensity-centroid orientation)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Arm Intrinsics reference (NEON)](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- Arm, [Porting x86 vector bitmask optimizations to Arm NEON](https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon) (the `vshrn` movemask substitute)
- [easy_profiler](https://github.com/yse/easy_profiler)
- Benchmark image: LG Electronics, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) (see [data/README.md](data/README.md))
