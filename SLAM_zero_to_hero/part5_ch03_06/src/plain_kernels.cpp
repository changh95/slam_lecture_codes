/**
 * @file plain_kernels.cpp
 * @brief Plain C/C++ baselines, compiled once per flag set (see include/plain_kernels.hpp)
 *
 * CMake defines PLAIN_NAMESPACE (plain_o3 / plain_native) and ROSTEN_PREFIX for each copy.
 */

#include "rosten_prefix.h"  // before fast.h, so the Rosten calls below use this copy's symbols
#include "plain_kernels.hpp"

#include <cstdlib>

namespace PLAIN_NAMESPACE {

void rgbToGray(const uint8_t* bgr, int cols, int rows, uint8_t* gray) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            const int idx = 3 * (i * cols + j);
            gray[i * cols + j] = uint8_t(0.114 * bgr[idx] + 0.5f) + uint8_t(0.587 * bgr[idx + 1] + 0.5f) +
                                 uint8_t(0.299 * bgr[idx + 2] + 0.5f);
        }
    }
}

// fast9_detect_nonmax() from Rosten's fast.c
xy* fast9DetectNonmax(const uint8_t* gray, int cols, int rows, int stride, int threshold, int* count) {
    int num_corners = 0;
    xy* corners = fast9_detect(gray, cols, rows, stride, threshold, &num_corners);
    int* scores = fast9_score(gray, stride, corners, num_corners, threshold);
    xy* nonmax = nonmax_suppression(corners, scores, num_corners, count);
    std::free(corners);
    std::free(scores);
    return nonmax;
}

void icAngles(const cv::Mat& gray, const std::vector<cv::Point>& points, const std::vector<int>& umax, int half_k,
              float* angles) {
    const int step = static_cast<int>(gray.step1());
    for (size_t i = 0; i < points.size(); ++i) {
        const uchar* center = &gray.at<uchar>(points[i].y, points[i].x);

        int m_01 = 0, m_10 = 0;

        // Treat the center line differently, v = 0
        for (int u = -half_k; u <= half_k; ++u) m_10 += u * center[u];

        // Go line by line in the circular patch
        for (int v = 1; v <= half_k; ++v) {
            // Proceed over the two lines
            int v_sum = 0;
            const int d = umax[v];
            for (int u = -d; u <= d; ++u) {
                const int val_plus = center[u + v * step];
                const int val_minus = center[u - v * step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }
        angles[i] = cv::fastAtan2(static_cast<float>(m_01), static_cast<float>(m_10));
    }
}

}  // namespace PLAIN_NAMESPACE
