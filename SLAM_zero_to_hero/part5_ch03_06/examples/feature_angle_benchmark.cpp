/**
 * @file feature_angle_benchmark.cpp
 * @brief Keypoint orientation (intensity centroid): ORB's IC_Angle loop vs komrad36's FeatureAngle (SSE)
 *
 * This example demonstrates:
 * 1. ORB's intensity-centroid angle, the plain loop from OpenCV's orb.cpp (ICAngles,
 *    not public API, so it is reproduced in src/plain_kernels.cpp) with cv::fastAtan2,
 *    built with -O3 (the baseline) and with -O3 -march=native.
 *    OpenCV has no vectorized version: cv::ORB runs exactly this loop.
 * 2. komrad36's featureAngle(): the same moments with 8 x 16-bit SSE lanes, single-threaded
 *    and with the keypoints split over komrad36's FastThreadPool (include/fast_parallel_for.hpp)
 * 3. Timing both over all FAST corners of the image with easy_profiler
 *
 * Both compute the angle of (m10, m01) over the circular 7x7 patch KORAL uses
 * (half_k = 3). ORB itself uses a 31x31 patch (half_k = 15); see the README exercises.
 *
 * How FeatureAngle works:
 * - Each patch row is 7 pixels, so one 8-byte load widened to 8 x int16 holds a row
 * - The x and y weights of the circular mask are constant vectors; _mm_mullo_epi16
 *   and adds accumulate m10 and m01 for all columns at once (no branches, no u_max table)
 * - _mm_hadd_epi16 and shuffles reduce the 8 lanes to one sum, and a polynomial
 *   atan2 turns (m10, m01) into radians
 */

#include "benchmark_common.hpp"
#include "fast_parallel_for.hpp"
#include "komrad36_compat.hpp"
#include "plain_kernels.hpp"
#include "profile_summary.hpp"

#include "FeatureAngle/FeatureAngle.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <easy/profiler.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

constexpr int kHalfPatch = 3;    // 7x7 patch, as in komrad36's FeatureAngle
constexpr int kThreshold = 20;   // FAST threshold for the keypoints whose angles we compute

/**
 * @brief Row half-widths of the circular patch, computed exactly like cv::ORB does
 *
 * For half_k = 3 this gives {3, 3, 2, 1}, the same mask komrad36 hard-codes.
 */
std::vector<int> computeUMax(int half_patch) {
    std::vector<int> umax(half_patch + 2);
    const int vmax = cvFloor(half_patch * std::sqrt(2.f) / 2 + 1);
    const int vmin = cvCeil(half_patch * std::sqrt(2.f) / 2);
    for (int v = 0; v <= vmax; ++v) umax[v] = cvRound(std::sqrt(static_cast<double>(half_patch * half_patch - v * v)));

    // Make sure we are symmetric
    for (int v = half_patch, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1]) ++v0;
        umax[v] = v0;
        ++v0;
    }
    return umax;
}

/**
 * @brief Largest and mean |difference| between two angle lists, in degrees (wraps at 360)
 */
void compareAngles(const std::string& label, const std::vector<float>& deg, const std::vector<float>& ref_deg) {
    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < deg.size(); ++i) {
        const double diff = std::fabs(std::remainder(static_cast<double>(deg[i]) - ref_deg[i], 360.0));
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    std::cout << "  " << label << ": mean |diff| = " << (deg.empty() ? 0.0 : sum_diff / deg.size())
              << " deg, max |diff| = " << max_diff << " deg" << std::endl;
}

int main(int argc, char** argv) {
    EASY_PROFILER_ENABLE;
    EASY_MAIN_THREAD;

    simd_bench::exitIfUnsupportedCpu();
    const simd_bench::BenchmarkArgs args = simd_bench::parseArgs(argc, argv);

    std::cout << "========================================" << std::endl;
    std::cout << "FeatureAngle: keypoint orientation (easy_profiler)" << std::endl;
    std::cout << "========================================" << std::endl;

    const cv::Mat gray = simd_bench::loadGrayPadded(args);
    const int step = static_cast<int>(gray.step);

    // FAST keeps a 3-pixel border, which is exactly what a 7x7 patch needs
    std::vector<cv::KeyPoint> keypoints;
    cv::FAST(gray, keypoints, kThreshold, true);
    std::vector<cv::Point> points;
    points.reserve(keypoints.size());
    for (const auto& kp : keypoints) points.emplace_back(cvRound(kp.pt.x), cvRound(kp.pt.y));
    const size_t n = points.size();
    std::cout << "Keypoints: " << n << " (FAST threshold " << kThreshold << "), iterations per variant: "
              << args.iterations << std::endl;

    const std::vector<int> umax = computeUMax(kHalfPatch);
    std::vector<float> angles_o3(n), angles_native(n), angles_k(n), angles_k_mt(n);

    auto plain_o3 = [&] { plain_o3::icAngles(gray, points, umax, kHalfPatch, angles_o3.data()); };
    auto plain_native = [&] { plain_native::icAngles(gray, points, umax, kHalfPatch, angles_native.data()); };
    auto komrad = [&] {
        for (size_t i = 0; i < n; ++i) angles_k[i] = featureAngle(gray.data, points[i].x, points[i].y, step);
    };

    // Multi-threaded: stripe s computes the angles of its share of the keypoints
    int num_stripes = 1;
    auto komrad_stripe = [&](int stripe) {
        const size_t first = n * stripe / num_stripes, last = n * (stripe + 1) / num_stripes;
        for (size_t i = first; i < last; ++i) angles_k_mt[i] = featureAngle(gray.data, points[i].x, points[i].y, step);
    };

    simd_bench::warmup(args, plain_o3);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("Angle/plain_O3_1T", profiler::colors::Grey500);
        plain_o3();
    }

    simd_bench::warmup(args, plain_native);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("Angle/plain_native_1T", profiler::colors::Grey800);
        plain_native();
    }

    simd_bench::warmup(args, komrad);
    for (int i = 0; i < args.iterations; ++i) {
        EASY_BLOCK("Angle/komrad36_1T", profiler::colors::Orange500);
        komrad();
    }

    // ---- Multi-thread: FeatureAngle on FastThreadPool (cv::ORB computes angles single-threaded) ----
    bool mt_identical = true;
    std::vector<std::pair<int, double>> idle_cores;
    for (const int threads : simd_bench::threadCounts()) {
        const std::string block = "Angle/komrad36_MT" + std::to_string(threads);
        simd_bench::FastParallelFor pool(threads);
        num_stripes = threads;
        auto komrad_mt = [&] { pool.run(num_stripes, komrad_stripe); };

        std::fill(angles_k_mt.begin(), angles_k_mt.end(), 0.f);
        simd_bench::warmup(args, komrad_mt);
        for (int i = 0; i < args.iterations; ++i) {
            EASY_BLOCK(block.c_str(), profiler::colors::Orange800);
            komrad_mt();
        }
        idle_cores.emplace_back(threads, simd_bench::idleCoresBusy());
        mt_identical = mt_identical && angles_k_mt == angles_k;
    }

    // ---- Verification: komrad36 returns radians in (-pi, pi], the plain loop degrees in [0, 360) ----
    std::vector<float> angles_k_deg(n);
    for (size_t i = 0; i < n; ++i) angles_k_deg[i] = static_cast<float>(angles_k[i] * 180.0 / CV_PI);

    std::cout << "\nVerification (reference: plain -O3 IC_Angle):" << std::endl;
    compareAngles("plain native", angles_native, angles_o3);
    compareAngles("komrad36    ", angles_k_deg, angles_o3);
    std::cout << "  komrad36 MT : " << (mt_identical ? "identical to komrad36 1T at every thread count"
                                                   : "DIFFERENT from komrad36 1T") << std::endl;
    std::cout << "  (same atan2 polynomial as cv::fastAtan2, in radians instead of degrees: the difference is float rounding)" << std::endl;

    std::cout << "\nFastThreadPool workers spin even without work. Cores busy while idle:" << std::endl;
    for (const auto& [threads, cores] : idle_cores)
        std::cout << "  " << threads << " threads: " << std::fixed << std::setprecision(1) << cores << " cores"
                  << std::endl;

    const std::string profile_path = "feature_angle_benchmark.prof";
    profiler::dumpBlocksToFile(profile_path.c_str());
    simd_bench::printSummary(profile_path, simd_bench::readProfile(profile_path));

    std::cout << "\nProfile saved to: " << profile_path << " (open with profiler_gui)" << std::endl;
    return 0;
}
