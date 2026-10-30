/**
 * @file klt_reference.hpp
 * @brief vilib's KLT step (FeatureTrackerGPU without the affine terms) in plain C++
 *
 * tracker_benchmark uses this to check the CUDA tracker: given the same template
 * frame, first position, start position and new frame, it must land on the same
 * spot and make the same keep/drop decision. It follows feature_tracker_cuda_tools.cu
 * rule for rule. The kernel does the same work with one warp per feature: with a
 * 16x16 patch, each of the 32 threads takes one column in every 2nd row (8 pixels),
 * and the per-thread sums are added with __shfl_xor_sync where this loop just keeps adding.
 *
 * Inverse compositional Lucas-Kanade, coarse to fine:
 * - Template: the patch around the feature's first position in the frame it was
 *   detected in, cut at integer pixels (vilib does not interpolate it; at a level
 *   l > 0 the position x / 2^l is truncated), plus a 1-pixel border for the gradients
 * - Gradients J and the 2x2 Hessian H = sum(J J^T) come from the template only.
 *   vilib stores the patch and H^-1 once per level; the gradients it recomputes from
 *   the stored patch in every iteration (cheap), as this loop does
 * - Each iteration: bilinear samples of the new frame under the patch, residual
 *   r = I - T, update = H^-1 sum(J r), position -= update; stop when |update|^2 < min
 * - A level that does not converge within max_iterations ends the track; a patch
 *   that leaves the image skips to the next finer level, and on the finest level
 *   ends the track
 */

#pragma once

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace cuda_bench {

/**
 * @param template_pyramid pyramid of the frame the feature was detected in (vilib's 2x2-average pyramid)
 * @param first            the feature's position in that frame (level-0 pixels)
 * @param pyramid          pyramid of the new frame
 * @param position         in: the start (the feature's position in the previous frame); out: the result
 * @return false if vilib would drop the track
 */
inline bool kltReferenceStep(const std::vector<cv::Mat>& template_pyramid, const cv::Point2f& first,
                             const std::vector<cv::Mat>& pyramid, cv::Point2f& position, int patch_size,
                             int max_iterations, float min_update_squared) {
    const int half = patch_size / 2;
    const int stride = patch_size + 2;  // template row length, with the 1-pixel border
    std::vector<int> T(stride * stride);

    bool converged = false, left_image = true;
    for (int level = static_cast<int>(pyramid.size()) - 1; (converged || left_image) && level >= 0; --level) {
        const float scale = static_cast<float>(1 << level);
        const cv::Mat& ref = template_pyramid[level];
        const cv::Mat& img = pyramid[level];

        // Template at integer pixels; a level where it does not fit is skipped
        const int x0 = static_cast<int>(std::floor(first.x / scale - (half + 1)));
        const int y0 = static_cast<int>(std::floor(first.y / scale - (half + 1)));
        if (x0 < 0 || y0 < 0 || x0 + patch_size + 1 >= img.cols || y0 + patch_size + 1 >= img.rows) continue;
        for (int y = 0; y < stride; ++y)
            for (int x = 0; x < stride; ++x) T[y * stride + x] = ref.at<uchar>(y0 + y, x0 + x);

        // Hessian of the template, inverted once
        float h00 = 0.f, h01 = 0.f, h11 = 0.f;
        for (int y = 0; y < patch_size; ++y)
            for (int x = 0; x < patch_size; ++x) {
                const int* t = &T[(y + 1) * stride + (x + 1)];
                const float jx = 0.5f * (t[1] - t[-1]), jy = 0.5f * (t[stride] - t[-stride]);
                h00 += jx * jx;
                h01 += jx * jy;
                h11 += jy * jy;
            }
        const float inv_det = 1.0f / (h00 * h11 - h01 * h01);
        const float i00 = h11 * inv_det, i01 = -h01 * inv_det, i11 = h00 * inv_det;
        // vilib skips a level when the first element of H^-1 is NaN: the template has no vertical
        // gradient at all (h11 = 0, so det = 0 and 0 * inf), e.g. a flat patch. Same expression here.
        if (std::isnan(i00)) continue;

        float px = position.x / scale, py = position.y / scale;
        converged = false;
        left_image = false;
        for (int iter = 0; iter < max_iterations; ++iter) {
            if (std::isnan(px) || std::isnan(py)) break;
            const int u = static_cast<int>(std::floor(px)), v = static_cast<int>(std::floor(py));
            if (u < half || v < half || u >= img.cols - half || v >= img.rows - half) {
                left_image = true;
                break;
            }
            const float ax = px - u, ay = py - v;
            const float w00 = (1 - ax) * (1 - ay), w01 = ax * (1 - ay), w10 = (1 - ax) * ay, w11 = ax * ay;

            float jr0 = 0.f, jr1 = 0.f;
            for (int y = 0; y < patch_size; ++y) {
                const uchar* row = img.ptr<uchar>(v - half + y) + (u - half);
                for (int x = 0; x < patch_size; ++x) {
                    const uchar* p = row + x;
                    const float sample = w00 * p[0] + w01 * p[1] + w10 * p[img.step] + w11 * p[img.step + 1];
                    const int* t = &T[(y + 1) * stride + (x + 1)];
                    const float r = sample - static_cast<float>(t[0]);
                    jr0 += r * 0.5f * (t[1] - t[-1]);
                    jr1 += r * 0.5f * (t[stride] - t[-stride]);
                }
            }
            const float dx = i00 * jr0 + i01 * jr1, dy = i01 * jr0 + i11 * jr1;
            px -= dx;
            py -= dy;
            if (dx * dx + dy * dy < min_update_squared) {
                converged = true;
                break;
            }
        }
        position = cv::Point2f(px * scale, py * scale);
    }
    return converged;
}

}  // namespace cuda_bench
