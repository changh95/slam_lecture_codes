/**
 * @file plain_kernels.hpp
 * @brief Plain C/C++ baselines: straightforward code without intrinsics
 *
 * src/plain_kernels.cpp and Rosten's FAST-9 C code are compiled twice, so the
 * benchmarks show what the compiler alone makes of the same code:
 *   plain_o3     : -O3                 portable Release build (GCC may use SSE2 on x86-64, NEON on AArch64)
 *   plain_native : -O3 -march=native   GCC may use every extension of the build CPU (AVX2, AVX-512, SVE, ...)
 */

#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

extern "C" {
#include "KFAST/Rosten/fast.h"  // xy
}

#define SIMD_BENCH_DECLARE_PLAIN_KERNELS(ns)                                                               \
    namespace ns {                                                                                          \
    /* BGR -> gray, one pixel at a time (the formula of komrad36's RGB2Y_ref) */                             \
    void rgbToGray(const uint8_t* bgr, int cols, int rows, uint8_t* gray);                                   \
    /* Rosten's FAST-9 with non-maximum suppression; the caller free()s the result */                        \
    xy* fast9DetectNonmax(const uint8_t* gray, int cols, int rows, int stride, int threshold, int* count);  \
    /* Intensity-centroid angle in degrees, the loop of OpenCV's orb.cpp (ICAngles) */                      \
    void icAngles(const cv::Mat& gray, const std::vector<cv::Point>& points, const std::vector<int>& umax,  \
                  int half_k, float* angles);                                                               \
    }

SIMD_BENCH_DECLARE_PLAIN_KERNELS(plain_o3)
SIMD_BENCH_DECLARE_PLAIN_KERNELS(plain_native)
