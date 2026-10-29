/**
 * @file RGB2Y_neon.h
 * @brief NEON (AArch64) port of komrad36's RGB2Y: BGR -> gray, bit-exact with RGB2Y.h
 *
 * Derived from komrad36's RGB2Y (third_party/komrad36/RGB2Y, MIT License,
 * Copyright (c) 2021 Kareem Omar). It has the same interface as RGB2Y.h, so the
 * benchmarks call it unchanged on AArch64, and it produces the same output.
 *
 * How the AVX2 design maps to NEON:
 *  1. De-interleaving is a load. vld3q_u8 reads 48 bytes (16 BGR pixels) and splits them
 *     into one register of B, one of G and one of R. RGB2Y.h needs packus, three vpshufb
 *     with hand-made masks and a final cross-half add for the same job, because x86 has
 *     no structured loads; NEON has vld2/vld3/vld4 for exactly this.
 *  2. The same Q15 math. vqrdmulhq_s16 (saturating rounding doubling multiply, high half)
 *     computes (2 * a * w + 2^15) >> 16 = (a * w + 2^14) >> 15, which is what
 *     _mm256_mulhrs_epi16 computes. Every product is rounded the same way, so the output
 *     matches RGB2Y and RGB2Y_ref bit for bit.
 *  3. 16 pixels per step. A 128-bit register holds 16 bytes, so one vld3q gives 16 pixels.
 *     Widening to int16 splits each channel into a low and a high half of 8 lanes;
 *     vmovl_high_u8 widens the high half directly, without extracting it first.
 *  4. Exact rows. Instead of a scalar tail, the last step of a row is moved back so that
 *     it ends on the row's last pixel; it recomputes up to 15 pixels that are already
 *     done, with the same result. Nothing is read or written outside the row, so the
 *     image needs no padding, and threads may split the rows anywhere. RGB2Y.h instead
 *     stores 16 bytes per 10 pixels and spills into the next row (see its "Stores" note).
 */

#pragma once

#include <arm_neon.h>

#include <cstdint>

namespace rgb2y_neon {

// The same weights and Q15 rounding as RGB2Y.h: 3736, 19235, 9798
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;
constexpr int16_t B_WT = static_cast<int16_t>(32768.0 * B_WEIGHT + 0.5);
constexpr int16_t G_WT = static_cast<int16_t>(32768.0 * G_WEIGHT + 0.5);
constexpr int16_t R_WT = static_cast<int16_t>(32768.0 * R_WEIGHT + 0.5);
constexpr int16_t THIRD = 10923;  // round(2^15 / 3), for equal weights

// round(a * w) in Q15, as _mm256_mulhrs_epi16 and vqrdmulhq_s16 compute it
inline int mulQ15(const int a, const int w) { return (a * w + 16384) >> 15; }

/**
 * @brief Gray values of 16 pixels (48 bytes of BGR)
 */
template <bool weight>
inline __attribute__((always_inline)) uint8x16_t gray16(const uint8_t* __restrict const pt) {
    const uint8x16x3_t bgr = vld3q_u8(pt);  // val[0] = B, val[1] = G, val[2] = R
    int16x8_t lo, hi;
    if (weight) {
        const int16x8_t wb = vdupq_n_s16(B_WT), wg = vdupq_n_s16(G_WT), wr = vdupq_n_s16(R_WT);
        // Widen to int16, then round(channel * weight) per product, as RGB2Y.h does. Each rounded
        // product is <= 255, so the int16 sum cannot overflow, and vqmovun below clamps it at 255
        // like RGB2Y.h's saturating byte adds.
        lo = vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bgr.val[0]))), wb);
        lo = vaddq_s16(lo, vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bgr.val[1]))), wg));
        lo = vaddq_s16(lo, vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bgr.val[2]))), wr));
        hi = vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_high_u8(bgr.val[0])), wb);
        hi = vaddq_s16(hi, vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_high_u8(bgr.val[1])), wg));
        hi = vaddq_s16(hi, vqrdmulhq_s16(vreinterpretq_s16_u16(vmovl_high_u8(bgr.val[2])), wr));
    } else {
        // Widening adds (vaddl/vaddw) unpack and add in one instruction; sums reach 765.
        // Then divide by 3 with the same Q15 multiply RGB2Y.h uses.
        const uint16x8_t sum_lo = vaddw_u8(vaddl_u8(vget_low_u8(bgr.val[0]), vget_low_u8(bgr.val[1])),
                                           vget_low_u8(bgr.val[2]));
        const uint16x8_t sum_hi = vaddw_high_u8(vaddl_high_u8(bgr.val[0], bgr.val[1]), bgr.val[2]);
        lo = vqrdmulhq_s16(vreinterpretq_s16_u16(sum_lo), vdupq_n_s16(THIRD));
        hi = vqrdmulhq_s16(vreinterpretq_s16_u16(sum_hi), vdupq_n_s16(THIRD));
    }
    // Narrow back to bytes with unsigned saturation: the low half's 8 bytes, then the high half's
    return vqmovun_high_s16(vqmovun_s16(lo), hi);
}

/**
 * @brief Convert one row of cols pixels; reads 3 * cols bytes and writes cols bytes, no more
 */
template <bool weight>
inline void convertRow(const uint8_t* __restrict const bgr, const int32_t cols, uint8_t* __restrict const gray) {
    if (cols < 16) {  // too narrow for one step
        for (int32_t j = 0; j < cols; ++j) {
            const int b = bgr[3 * j], g = bgr[3 * j + 1], r = bgr[3 * j + 2];
            const int y = weight ? mulQ15(b, B_WT) + mulQ15(g, G_WT) + mulQ15(r, R_WT) : mulQ15(b + g + r, THIRD);
            gray[j] = static_cast<uint8_t>(y > 255 ? 255 : y);
        }
        return;
    }
    int32_t j = 0;
    for (; j <= cols - 16; j += 16) vst1q_u8(gray + j, gray16<weight>(bgr + 3 * j));
    // Last 1..15 pixels: one more step, moved back to end on the last pixel (see design note 4)
    if (j < cols) vst1q_u8(gray + cols - 16, gray16<weight>(bgr + 3 * (cols - 16)));
}

}  // namespace rgb2y_neon

// ---- The interface of RGB2Y.h ----

/**
 * @brief One row. last_row only matters for RGB2Y.h; here every row is exact
 */
template <bool last_row, bool weight>
void processRow(const uint8_t* __restrict pt, const int32_t cols, uint8_t* const __restrict out) {
    rgb2y_neon::convertRow<weight>(pt, cols, out);
}

/**
 * @brief Rows start_row .. start_row + rows - 1. Input rows are 3 * stride bytes apart, output rows cols bytes apart
 */
template <bool weight>
inline void _RGB2Y(const uint8_t* __restrict const data, const int32_t cols, const int32_t start_row,
                   const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
    for (int32_t i = start_row; i < start_row + rows; ++i)
        rgb2y_neon::convertRow<weight>(data + 3 * i * stride, cols, out + i * cols);
}

template <bool multithread, bool weight>
void RGB2Y(const uint8_t* const __restrict image, const int width, const int height, const int stride,
           uint8_t* const __restrict out) {
    static_assert(!multithread, "the NEON port has no built-in threading: call _RGB2Y on your own threads");
    _RGB2Y<weight>(image, width, 0, height, stride, out);
}
