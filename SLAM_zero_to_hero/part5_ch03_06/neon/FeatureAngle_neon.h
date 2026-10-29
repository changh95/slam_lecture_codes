/**
 * @file FeatureAngle_neon.h
 * @brief NEON (AArch64) port of komrad36's FeatureAngle: intensity-centroid angle of a 7x7 patch
 *
 * Derived from komrad36's FeatureAngle (third_party/komrad36/FeatureAngle, MIT License,
 * Copyright (c) 2021 Kareem Omar). It has the same interface as FeatureAngle.h. The moments
 * are the same integers, and fastAtan2 is komrad36's polynomial, unchanged, so the angles
 * agree up to float rounding (GCC fuses multiply-adds into FMA by default on AArch64).
 *
 * How the SSE design maps to NEON:
 *  1. The same layout: one patch row (7 pixels + 1 ignored) per 8 x int16 register.
 *     vld1_u8 loads 8 bytes and vmovl_u8 widens them, like movq + pmovzxbw.
 *  2. Multiply-accumulate is one instruction. vmlaq_s16 / vmlsq_s16 compute acc +/- r * w,
 *     where SSE needs pmullw followed by paddw / psubw.
 *  3. Horizontal sums are one instruction. vaddvq_s16 (AArch64) adds all 8 lanes; SSE needs
 *     the three-step shuffle / hadd / shuffle reduction of FeatureAngle.h.
 *  4. The same 16-bit bounds: no partial sum exceeds 52 * 255 = 13260, so int16 lanes are exact.
 * As in FeatureAngle.h, each row load reads one byte right of the patch (weight 0), so
 * px + 4 must still be inside the buffer.
 */

#pragma once

#include <arm_neon.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace feature_angle_neon {

constexpr float PI = 3.1415927f;

/**
 * @brief komrad36's atan2 approximation, unchanged (see the note in FeatureAngle.h): radians in [-pi, pi]
 */
inline float fastAtan2(float y, float x) {
    const float ax = std::fabs(x);
    const float ay = std::fabs(y);
    float a;
    if (ax >= ay) {
        const float c = ay / (ax + FLT_MIN);
        const float cc = c * c;
        a = (((-0.0443265555479f * cc + 0.1555786518f) * cc - 0.325808397f) * cc + 0.9997878412f) * c;
    } else {
        const float c = ax / (ay + FLT_MIN);
        const float cc = c * c;
        a = PI * 0.5f - (((-0.0443265555479f * cc + 0.1555786518f) * cc - 0.325808397f) * cc + 0.9997878412f) * c;
    }
    if (x < 0.0f) a = PI - a;
    if (y < 0.0f) a = -a;
    return a;
}

//     0 1 2 3 4 5 6
//   +--------------
// 0 | - - x x x - -
// 1 | - x x x x x -
// 2 | x x x x x x x
// 3 | x x x o x x x
// 4 | x x x x x x x
// 5 | - x x x x x -
// 6 | - - x x x - -
//
// The weights of FeatureAngle.h: x inside the circle, |y| inside the circle, 0 outside and in lane 7
inline const int16x8_t xwt0 = {0, 0, -1, 0, 1, 0, 0, 0};
inline const int16x8_t xwt1 = {0, -2, -1, 0, 1, 2, 0, 0};
inline const int16x8_t xwt2 = {-3, -2, -1, 0, 1, 2, 3, 0};

inline const int16x8_t ywt0 = {0, 0, 3, 3, 3, 0, 0, 0};
inline const int16x8_t ywt1 = {0, 2, 2, 2, 2, 2, 0, 0};
inline const int16x8_t ywt2 = {1, 1, 1, 1, 1, 1, 1, 0};

inline __attribute__((always_inline)) int16x8_t loadRow(const uint8_t* __restrict const p) {
    return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
}

}  // namespace feature_angle_neon

/**
 * @brief Angle of the feature at (px, py) in radians, in [-pi, pi]; step is the row pitch in bytes
 */
inline float featureAngle(const uint8_t* const __restrict image, const int px, const int py, const int step) {
    using namespace feature_angle_neon;
    const uint8_t* __restrict p = image + (py - 3) * step + (px - 3);

    // Rows 0-2 (y = -3..-1): add x * I, subtract |y| * I
    int16x8_t r = loadRow(p);
    int16x8_t x = vmulq_s16(r, xwt0);
    int16x8_t y = vnegq_s16(vmulq_s16(r, ywt0));
    p += step;

    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt1);
    y = vmlsq_s16(y, r, ywt1);
    p += step;

    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt2);
    y = vmlsq_s16(y, r, ywt2);
    p += step;

    // Row 3, the center row: y = 0
    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt2);
    p += step;

    // Rows 4-6 (y = 1..3): add both terms
    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt2);
    y = vmlaq_s16(y, r, ywt2);
    p += step;

    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt1);
    y = vmlaq_s16(y, r, ywt1);
    p += step;

    r = loadRow(p);
    x = vmlaq_s16(x, r, xwt0);
    y = vmlaq_s16(y, r, ywt0);

    // One instruction per horizontal sum (addv)
    const float x_sum = static_cast<float>(vaddvq_s16(x));
    const float y_sum = static_cast<float>(vaddvq_s16(y));
    return fastAtan2(y_sum, x_sum);
}
