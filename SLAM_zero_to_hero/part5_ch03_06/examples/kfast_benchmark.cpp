/**
 * @file kfast_benchmark.cpp
 * @brief FAST-9 corners with non-maximum suppression: Rosten's C code vs OpenCV cv::FAST vs komrad36's KFAST (AVX2)
 *
 * This example demonstrates:
 * 1. Rosten's original FAST-9 (plain C decision tree, the paper's reference code),
 *    built with -O3 (the baseline) and with -O3 -march=native
 * 2. OpenCV cv::FAST(TYPE_9_16, nonmax)
 * 3. komrad36's KFAST, single-threaded and split into row stripes on komrad36's
 *    FastThreadPool (include/fast_parallel_for.hpp), for 4, 8, 16, ... threads
 * 4. Timing every call with easy_profiler and checking all detectors return the same corners
 *
 * How KFAST works:
 * - Tests 32 neighbouring pixels per step with 256-bit byte compares
 * - x86 has no unsigned byte compare, so both sides are XORed with 0x80 and
 *   compared as signed bytes (_mm256_cmpgt_epi8)
 * - The four compass points (1, 5, 9, 13) reject most of the 32 pixels early;
 *   _mm256_movemask_epi8 turns the survivors into a 32-bit mask
 * - Survivors run the full 16-pixel circle test as a branch-free run-length
 *   count, and _tzcnt_u32 / _blsr_u32 (BMI1) visit the set bits of the mask
 */

#include "benchmark_common.hpp"
#include "fast_parallel_for.hpp"
#include "komrad36_compat.hpp"
#include "plain_kernels.hpp"
#include "profile_summary.hpp"

#include "KFAST/KFAST.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <easy/profiler.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// ORB's default FAST threshold (cv::ORB::create(..., fastThreshold = 20))
constexpr int kThreshold = 20;

using Corners = std::vector<std::pair<int, int>>;  // (y, x), sorted

template <typename T, typename GetX, typename GetY>
Corners toSortedCorners(const T* points, size_t count, GetX get_x, GetY get_y) {
    Corners corners(count);
    for (size_t i = 0; i < count; ++i) corners[i] = {get_y(points[i]), get_x(points[i])};
    std::sort(corners.begin(), corners.end());
    return corners;
}

Corners rostenCorners(const xy* points, int count) {
    return toSortedCorners(
        points, static_cast<size_t>(count), [](const xy& p) { return p.x; }, [](const xy& p) { return p.y; });
}

void compareCorners(const std::string& label, const Corners& result, const Corners& reference) {
    std::cout << "  " << label << ": " << result.size() << " corners, "
              << (result == reference ? "identical to plain -O3" : "DIFFERENT from plain -O3") << std::endl;
}

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    simd_bench::exitIfUnsupportedCpu();
    const simd_bench::BenchmarkArgs args = simd_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "KFAST: FAST-9 + NMS (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    const cv::Mat gray = simd_bench::loadGrayPadded(args);
    const int stride = static_cast<int>(gray.step);
    std::cout << "Threshold: " << kThreshold << ", iterations per variant: " << args.iterations
              << ", std::thread::hardware_concurrency: " << std::thread::hardware_concurrency() << std::endl;

    std::vector<cv::KeyPoint> kps_cv;
    std::vector<Keypoint> kps_k1, kps_k_mt;
    int num_o3 = 0, num_native = 0;
    xy* kps_o3 = nullptr;
    xy* kps_native = nullptr;

    auto plain_o3 = [&] {
        std::free(kps_o3);
        kps_o3 = plain_o3::fast9DetectNonmax(gray.data, gray.cols, gray.rows, stride, kThreshold, &num_o3);
    };
    auto plain_native = [&] {
        std::free(kps_native);
        kps_native = plain_native::fast9DetectNonmax(gray.data, gray.cols, gray.rows, stride, kThreshold, &num_native);
    };
    auto opencv = [&] { cv::FAST(gray, kps_cv, kThreshold, true, cv::FastFeatureDetector::TYPE_9_16); };
    auto kfast_1t = [&] { KFAST<false, true>(gray.data, gray.cols, gray.rows, stride, kps_k1, kThreshold); };

    // Multi-threaded: stripe s runs KFAST's row-range worker on its rows plus 4 rows above and
    // below, the same overlap KFAST's own multi-threaded split uses, so non-max suppression is
    // exact at the seams. Each stripe fills its own list; the lists are concatenated afterwards.
    int num_stripes = 1;
    std::vector<std::vector<Keypoint>> stripe_kps;
    auto kfast_stripe = [&](int s) {
        const int first_row = gray.rows * s / num_stripes, last_row = gray.rows * (s + 1) / num_stripes;
        const bool first = s == 0, last = s == num_stripes - 1;
        const int start = first ? 0 : first_row - 4;
        const int end = last ? gray.rows : last_row + 4;
        const uint8_t* data = gray.data + start * stride;
        std::vector<Keypoint>& kps = stripe_kps[s];
        kps.clear();
        if (first && last) _KFAST<true, true, true>(data, gray.cols, start, end - start, stride, kps, kThreshold);
        else if (first) _KFAST<true, true, false>(data, gray.cols, start, end - start, stride, kps, kThreshold);
        else if (last) _KFAST<true, false, true>(data, gray.cols, start, end - start, stride, kps, kThreshold);
        else _KFAST<true, false, false>(data, gray.cols, start, end - start, stride, kps, kThreshold);
    };

    cv::setNumThreads(1);  // cv::FAST is single-threaded anyway; keeps OpenCV's threads out of the picture

    simd_bench::warmup(args, plain_o3);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("FAST/plain_O3_1T", profiler::colors::Grey500);
        plain_o3();
    }

    simd_bench::warmup(args, plain_native);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("FAST/plain_native_1T", profiler::colors::Grey800);
        plain_native();
    }

    simd_bench::warmup(args, opencv);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("FAST/OpenCV_1T", profiler::colors::Blue500);
        opencv();
    }

    simd_bench::warmup(args, kfast_1t);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("FAST/komrad36_1T", profiler::colors::Orange500);
        kfast_1t();
    }

    // ---- Multi-thread: KFAST on FastThreadPool (cv::FAST has no multi-threaded version) ----
    auto kfast_corners = [](const std::vector<Keypoint>& kps) {
        return toSortedCorners(
            kps.data(), kps.size(), [](const Keypoint& k) { return k.x; }, [](const Keypoint& k) { return k.y; });
    };
    bool mt_identical = true;
    std::vector<std::pair<int, double>> idle_cores;
    for (const int threads : simd_bench::threadCounts()) {
        const std::string block = "FAST/komrad36_MT" + std::to_string(threads);
        simd_bench::FastParallelFor pool(threads);
        // KFAST needs a few rows per stripe for its 4-row margins (its own split uses rows / 16 at most)
        num_stripes = std::max(1, std::min(threads, gray.rows / 16));
        stripe_kps.assign(num_stripes, {});
        auto kfast_mt = [&] {
            pool.run(num_stripes, kfast_stripe);
            kps_k_mt.clear();
            for (const auto& kps : stripe_kps) kps_k_mt.insert(kps_k_mt.end(), kps.begin(), kps.end());
        };

        simd_bench::warmup(args, kfast_mt);
        for (int i = 0; i < args.iterations; ++i) {
            EASY_BLOCK(block.c_str(), profiler::colors::Orange800);
            kfast_mt();
        }
        idle_cores.emplace_back(threads, simd_bench::idleCoresBusy());
        mt_identical = mt_identical && kfast_corners(kps_k_mt) == rostenCorners(kps_o3, num_o3);
    }

    // ---- Verification: same corner set from every detector ----
    const Corners ref = rostenCorners(kps_o3, num_o3);

    std::cout << "\nVerification (reference: plain -O3, " << ref.size() << " corners):" << std::endl;
    compareCorners("plain native", rostenCorners(kps_native, num_native), ref);
    compareCorners("OpenCV      ", toSortedCorners(
                                       kps_cv.data(), kps_cv.size(),
                                       [](const cv::KeyPoint& k) { return static_cast<int>(k.pt.x); },
                                       [](const cv::KeyPoint& k) { return static_cast<int>(k.pt.y); }),
                   ref);
    compareCorners("KFAST 1T    ", kfast_corners(kps_k1), ref);
    std::cout << "  KFAST MT    : " << (mt_identical ? "identical to plain -O3 at every thread count"
                                                   : "DIFFERENT from plain -O3") << std::endl;

    std::cout << "\nFastThreadPool workers spin even without work. Cores busy while idle:" << std::endl;
    for (const auto& [threads, cores] : idle_cores)
        std::cout << "  " << threads << " threads: " << std::fixed << std::setprecision(1) << cores << " cores"
                  << std::endl;
    std::free(kps_o3);
    std::free(kps_native);

    const std::string profile_path = "kfast_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());
    simd_bench::printSummary(profile_path, simd_bench::readProfile(profile_path));

    std::cout << "\nProfile saved to: " << profile_path << " (open with profiler_gui)" << std::endl;
    return 0;
}
