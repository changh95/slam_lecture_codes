/**
 * @file benchmark_common.hpp
 * @brief Helpers shared by the three SIMD benchmarks (arguments, padded buffers, CPU check)
 */

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#ifndef SIMD_DATA_DIR
#define SIMD_DATA_DIR "data"
#endif

// The SIMD variant: komrad36's AVX2/SSE kernels on x86-64, this exercise's NEON ports of them
// (neon/) on AArch64. It names the easy_profiler blocks, e.g. "RGB2Y/komrad36_1T" or "RGB2Y/NEON_1T".
#if defined(__aarch64__)
#define SIMD_VARIANT "NEON"
#else
#define SIMD_VARIANT "komrad36"
#endif

namespace simd_bench {

struct BenchmarkArgs {
    std::string image_path = std::string(SIMD_DATA_DIR) + "/img_rgb.png";  // 1920x1080, see data/README.md
    int iterations = 200;  // profiled calls per variant
    int warmup = 20;       // unprofiled calls per variant (page faults, caches, thread pools)
};

inline BenchmarkArgs parseArgs(int argc, char** argv) {
    BenchmarkArgs args;
    if (argc > 1) args.image_path = argv[1];
    if (argc > 2) args.iterations = std::max(1, std::atoi(argv[2]));
    return args;
}

/**
 * @brief komrad36's kernels need AVX2, and KFAST also needs BMI1 (tzcnt, blsr).
 *        The NEON ports need nothing beyond AArch64, where NEON is mandatory.
 */
inline bool cpuSupportsKernels() {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi");
#elif defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Pad a verification label to 12 characters, so the colons line up
 */
inline std::string label(std::string name) {
    if (name.size() < 12) name.resize(12, ' ');
    return name;
}

/**
 * @brief Allocate an image with one spare row after the last one
 *
 * komrad36's kernels load and store whole registers (16 or 32 bytes), so near
 * the end of the image they read, and in RGB2Y's case write, up to ~31 bytes past
 * the last pixel. The spare row keeps those accesses inside our own buffer.
 * The NEON ports stay inside the image (FeatureAngle aside, which reads one byte
 * right of its patch), but the benchmarks use the same buffers on both.
 * rowRange() of a continuous Mat is still continuous, so data/step/cols are
 * exactly those of a plain rows x cols Mat.
 */
inline cv::Mat createPadded(int rows, int cols, int type) {
    cv::Mat buffer(rows + 1, cols, type, cv::Scalar::all(0));
    return buffer.rowRange(0, rows);
}

/**
 * @brief Copy an image into a padded buffer (see createPadded)
 *
 * copyTo() and cvtColor() into a Mat of matching size and type reuse its
 * memory, so writing into the returned view keeps the padding.
 */
inline cv::Mat copyPadded(const cv::Mat& src) {
    cv::Mat dst = createPadded(src.rows, src.cols, src.type());
    src.copyTo(dst);
    return dst;
}

/**
 * @brief Load the benchmark image as 8-bit BGR, or exit with a hint
 */
inline cv::Mat loadBgr(const BenchmarkArgs& args) {
    cv::Mat image = cv::imread(args.image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: could not load " << args.image_path << std::endl;
        std::cerr << "Usage: <benchmark> [image_path] [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Image: " << args.image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    return image;
}

/**
 * @brief Load the benchmark image as 8-bit grayscale in a padded buffer
 */
inline cv::Mat loadGrayPadded(const BenchmarkArgs& args) {
    const cv::Mat bgr = loadBgr(args);
    cv::Mat gray = createPadded(bgr.rows, bgr.cols, CV_8UC1);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

/**
 * @brief Run a callable a few times without profiling it
 */
template <typename F>
void warmup(const BenchmarkArgs& args, F&& fn) {
    for (int i = 0; i < args.warmup; ++i) fn();
}

inline void exitIfUnsupportedCpu() {
    if (!cpuSupportsKernels()) {
        std::cerr << "Error: this CPU lacks AVX2/BMI1, which komrad36's kernels require "
                     "(x86-64 Haswell / Zen 1 or newer)." << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace simd_bench
