/**
 * @file rgb2y_benchmark.cpp
 * @brief BGR -> grayscale: plain C++ loop vs OpenCV cv::cvtColor vs komrad36's RGB2Y (AVX2)
 *
 * This example demonstrates:
 * 1. A plain per-pixel loop built with -O3 (the baseline) and with -O3 -march=native
 * 2. OpenCV cv::cvtColor(COLOR_BGR2GRAY), single-threaded and with OpenCV's thread pool
 * 3. komrad36's RGB2Y, single-threaded and multi-threaded
 * 4. Timing every call with easy_profiler and checking the outputs agree
 *
 * How RGB2Y works:
 * - Loads 30 bytes (10 BGR pixels) per step and widens them to 16-bit lanes
 * - _mm256_mulhrs_epi16 multiplies by Q15 fixed-point weights (0.114, 0.587, 0.299)
 *   with rounding, so no float math is needed
 * - _mm256_shuffle_epi8 (pshufb) gathers the B, G and R products of each pixel
 *   into the same lane, then saturating adds produce 10 gray pixels at once
 */

#include "benchmark_common.hpp"
#include "komrad36_compat.hpp"
#include "plain_kernels.hpp"
#include "profile_summary.hpp"

#include "RGB2Y/RGB2Y.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <easy/profiler.h>

#include <iostream>
#include <string>

/**
 * @brief Print how many pixels differ between two gray images, and by how much
 */
void compareGray(const std::string& label, const cv::Mat& result, const cv::Mat& reference) {
    const int differing = cv::countNonZero(result != reference);
    const double max_diff = cv::norm(result, reference, cv::NORM_INF);
    std::cout << "  " << label << ": " << differing << " pixels differ (max |diff| = " << max_diff << ")"
              << std::endl;
}

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    simd_bench::exitIfUnsupportedCpu();
    const simd_bench::BenchmarkArgs args = simd_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "RGB2Y: BGR -> gray (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    const cv::Mat bgr = simd_bench::copyPadded(simd_bench::loadBgr(args));
    const int cols = bgr.cols;
    const int rows = bgr.rows;
    const int default_threads = cv::getNumThreads();
    std::cout << "Iterations per variant: " << args.iterations << ", OpenCV threads: " << default_threads
              << ", std::thread::hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

    cv::Mat gray_o3 = simd_bench::createPadded(rows, cols, CV_8UC1);
    cv::Mat gray_native = simd_bench::createPadded(rows, cols, CV_8UC1);
    cv::Mat gray_cv = simd_bench::createPadded(rows, cols, CV_8UC1);
    cv::Mat gray_k1 = simd_bench::createPadded(rows, cols, CV_8UC1);
    cv::Mat gray_cv_mt = simd_bench::createPadded(rows, cols, CV_8UC1);
    cv::Mat gray_k_mt = simd_bench::createPadded(rows, cols, CV_8UC1);

    auto plain_o3 = [&] { plain_o3::rgbToGray(bgr.data, cols, rows, gray_o3.data); };
    auto plain_native = [&] { plain_native::rgbToGray(bgr.data, cols, rows, gray_native.data); };
    auto opencv = [&](cv::Mat& out) { cv::cvtColor(bgr, out, cv::COLOR_BGR2GRAY); };
    // RGB2Y's 'stride' is in pixels and the output is written with a row pitch of 'cols'
    auto komrad_1t = [&] { RGB2Y<false, true>(bgr.data, cols, rows, cols, gray_k1.data); };
    auto komrad_mt = [&] { RGB2Y<true, true>(bgr.data, cols, rows, cols, gray_k_mt.data); };

    // ---- Single thread ----
    cv::setNumThreads(1);

    simd_bench::warmup(args, plain_o3);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/plain_O3_1T", profiler::colors::Grey500);
        plain_o3();
    }

    simd_bench::warmup(args, plain_native);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/plain_native_1T", profiler::colors::Grey800);
        plain_native();
    }

    simd_bench::warmup(args, [&] { opencv(gray_cv); });
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/OpenCV_1T", profiler::colors::Blue500);
        opencv(gray_cv);
    }

    simd_bench::warmup(args, komrad_1t);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/komrad36_1T", profiler::colors::Orange500);
        komrad_1t();
    }

    // ---- Multi-thread: OpenCV's thread pool vs komrad36's std::async split ----
    cv::setNumThreads(default_threads);

    simd_bench::warmup(args, [&] { opencv(gray_cv_mt); });
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/OpenCV_MT", profiler::colors::Blue800);
        opencv(gray_cv_mt);
    }

    simd_bench::warmup(args, komrad_mt);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("RGB2Y/komrad36_MT", profiler::colors::Orange800);
        komrad_mt();
    }

    // ---- Verification against the plain -O3 loop ----
    std::cout << "\nVerification (reference: plain -O3 loop):" << std::endl;
    compareGray("plain native", gray_native, gray_o3);
    compareGray("komrad36 1T ", gray_k1, gray_o3);
    compareGray("komrad36 MT ", gray_k_mt, gray_o3);
    compareGray("OpenCV 1T   ", gray_cv, gray_o3);
    compareGray("OpenCV MT   ", gray_cv_mt, gray_o3);
    std::cout << "  (OpenCV rounds B*w_b + G*w_g + R*w_r once; the others round each product, so +-1 is expected)"
              << std::endl;

    const std::string profile_path = "rgb2y_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());
    simd_bench::printSummary(profile_path, simd_bench::readProfile(profile_path));

    std::cout << "\nProfile saved to: " << profile_path << " (open with profiler_gui)" << std::endl;
    return 0;
}
