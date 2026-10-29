/**
 * @file KFAST_neon.h
 * @brief NEON (AArch64) port of komrad36's KFAST: FAST-9 corners with non-max suppression
 *
 * Derived from komrad36's KFAST (third_party/komrad36/KFAST, MIT License,
 * Copyright (c) 2021 Kareem Omar), which implements the detector of Rosten and Drummond,
 * "Machine learning for high-speed corner detection" (ECCV 2006); Edward Rosten's FAST code
 * is BSD-licensed (third_party/komrad36/KFAST/Rosten/LICENSE). It has the same
 * interface as KFAST.h and returns the same corners and scores as KFAST and as Rosten's
 * fast9_detect + nonmax_suppression.
 *
 * How the AVX2 design maps to NEON:
 *  1. 16 candidates per step instead of 32: a NEON register is 128 bits. The structure is
 *     KFAST's: circle point k of 16 adjacent pixels is one 16-byte load at ptr + offsets[k].
 *  2. No XOR trick. NEON compares unsigned bytes directly (vcgtq_u8 / vcltq_u8), so p and
 *     the circle pixels are compared as they are; AVX2 only has a signed byte compare.
 *     Saturating vqaddq_u8 / vqsubq_u8 build p + t and p - t, as in KFAST.
 *  3. No movemask. NEON has no instruction that collects one bit per lane. Instead,
 *     vshrn_n_u16(mask, 4) narrows the 16 byte lanes into a 64-bit integer with 4 bits per
 *     lane (laneBits below). "Any candidate in these 16?" is then a test of that integer
 *     against 0, and keeping one bit per lane, ctz / 4 is the index of the next corner
 *     (ctz compiles to rbit + clz; a single CTZ instruction exists only on cores with
 *     Armv8.9's CSSC extension).
 *  4. A horizontal max exists. The score's max over the 16 arcs is vmaxvq_s16, one
 *     instruction; KFAST needs the 32767 - phminposuw(32767 - x) trick because x86 has only
 *     a horizontal min. The 16 arcs take two int16x8 registers instead of one 256-bit one.
 *  5. No reads outside the image. The last window of a row is moved back to end at the last
 *     candidate column, and the columns it repeats are masked off. KFAST's last window
 *     instead reads up to 32 bytes past the end of the row (README exercise 1).
 *  6. Not ported: KFAST's "advance by only 16 if the left half has no candidate". Whether
 *     the 8-lane version pays off with 16 lanes is README exercise 5.
 * The pre-test on the compass points, the branch-free run-length count over 24 circle
 * points and the 3-row streaming non-max suppression are KFAST's, unchanged.
 *
 * Images narrower than 22 columns have no room for a 16-pixel window between the
 * 3-pixel borders; they go through a scalar version of the same test.
 */

#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

struct Keypoint {
    int32_t x;
    int32_t y;
    uint8_t score;

    Keypoint(const int32_t _x, const int32_t _y, const uint8_t _score) : x(_x), y(_y), score(_score) {}
};

namespace kfast_neon {

/**
 * @brief 16 byte lanes of 0x00 / 0xFF -> 64-bit integer with 4 bits per lane (lane k = bits 4k..4k+3)
 */
inline __attribute__((always_inline)) uint64_t laneBits(const uint8x16_t mask) {
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "lane k = bits 4k..4k+3 holds on little-endian only");
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(mask), 4)), 0);
}

/**
 * @brief KFAST's corner score: the largest threshold at which p is still a corner, + 1
 *
 * ring[n] = p - circle[n]. Arc a covers circle points a..a+8, so 9 loads at ring + 0 .. ring + 8
 * give lane a the 9 points of arc a: arcs 0-7 in one register, arcs 8-15 in the other.
 */
inline __attribute__((always_inline)) uint8_t cornerScore(const uint8_t* const p, const int32_t* const offsets) {
    const int16_t center = *p;
    int16_t ring[24];
    for (int n = 0; n < 24; ++n) ring[n] = static_cast<int16_t>(center - p[offsets[n]]);

    int16x8_t min_lo = vld1q_s16(ring), max_lo = min_lo;
    int16x8_t min_hi = vld1q_s16(ring + 8), max_hi = min_hi;
    for (int k = 1; k < 9; ++k) {
        const int16x8_t lo = vld1q_s16(ring + k);
        const int16x8_t hi = vld1q_s16(ring + 8 + k);
        min_lo = vminq_s16(min_lo, lo);
        max_lo = vmaxq_s16(max_lo, lo);
        min_hi = vminq_s16(min_hi, hi);
        max_hi = vmaxq_s16(max_hi, hi);
    }
    // Per arc: min > 0 if the whole arc is darker than p, -max > 0 if it is brighter
    const int16x8_t lo = vmaxq_s16(min_lo, vnegq_s16(max_lo));
    const int16x8_t hi = vmaxq_s16(min_hi, vnegq_s16(max_hi));
    return static_cast<uint8_t>(vmaxvq_s16(vmaxq_s16(lo, hi)));
}

/**
 * @brief The same corner test for one pixel, for images too narrow for a 16-pixel window
 *
 * In int, p + t and p - t need no clamping: a bound above 255 or below 0 is never passed.
 */
inline bool isCornerScalar(const uint8_t* const p, const int32_t* const offsets, const uint8_t threshold) {
    const int bright = p[0] + threshold, dark = p[0] - threshold;
    int b_run = 0, d_run = 0, longest = 0;
    for (int k = 0; k < 24; ++k) {
        const int c = p[offsets[k]];
        b_run = c > bright ? b_run + 1 : 0;
        d_run = c < dark ? d_run + 1 : 0;
        longest = std::max(longest, std::max(b_run, d_run));
    }
    return longest > 8;
}

/**
 * @brief Test the 16 candidates at columns j..j+15 of row y; lanes outside 'valid' are skipped
 */
template <bool nonmax_suppression>
inline __attribute__((always_inline)) void processWindow(
    const uint8_t* const ptr, const int32_t j, const uint8x16_t valid, const int32_t* const offsets,
    const uint8x16_t t, int32_t& num_corners, int32_t* const corners, uint8_t* const cur,
    std::vector<Keypoint>& keypoints, const int32_t y) {
    const uint8x16_t p = vld1q_u8(ptr);
    const uint8x16_t bright = vqaddq_u8(p, t);  // p + t, clamped at 255
    const uint8x16_t dark = vqsubq_u8(p, t);    // p - t, clamped at 0

    // Pre-test on Rosten's compass points 9, 5, 1, 13 (offsets 0, 4, 8, 12): a 9-arc contains
    // two neighbouring ones, so a pixel without such a pair cannot be a corner
    const uint8x16_t c9 = vld1q_u8(ptr + offsets[0]);
    const uint8x16_t c5 = vld1q_u8(ptr + offsets[4]);
    const uint8x16_t c1 = vld1q_u8(ptr + offsets[8]);
    const uint8x16_t c13 = vld1q_u8(ptr + offsets[12]);
    const uint8x16_t b9 = vcgtq_u8(c9, bright), b5 = vcgtq_u8(c5, bright);
    const uint8x16_t b1 = vcgtq_u8(c1, bright), b13 = vcgtq_u8(c13, bright);
    const uint8x16_t d9 = vcltq_u8(c9, dark), d5 = vcltq_u8(c5, dark);
    const uint8x16_t d1 = vcltq_u8(c1, dark), d13 = vcltq_u8(c13, dark);
    const uint8x16_t b_pairs = vorrq_u8(vorrq_u8(vandq_u8(b9, b5), vandq_u8(b5, b1)),
                                        vorrq_u8(vandq_u8(b1, b13), vandq_u8(b13, b9)));
    const uint8x16_t d_pairs = vorrq_u8(vorrq_u8(vandq_u8(d9, d5), vandq_u8(d5, d1)),
                                        vorrq_u8(vandq_u8(d1, d13), vandq_u8(d13, d9)));
    const uint8x16_t candidates = vandq_u8(vorrq_u8(b_pairs, d_pairs), valid);

    // One integer test decides for all 16 lanes (KFAST: movemask, then the same test)
    if (laneBits(candidates) == 0) return;

    // Full test: longest run of brighter / darker points over the 16 circle points plus the
    // first 8 again. (cnt - mask) & mask adds 1 where the point passes (mask = 0xFF = -1)
    // and resets the run where it fails.
    uint8x16_t b_cnt = vdupq_n_u8(0), d_cnt = vdupq_n_u8(0);
    uint8x16_t b_max = vdupq_n_u8(0), d_max = vdupq_n_u8(0);
    for (int k = 0; k < 24; ++k) {
        const uint8x16_t c = vld1q_u8(ptr + offsets[k]);
        const uint8x16_t b = vcgtq_u8(c, bright);
        const uint8x16_t d = vcltq_u8(c, dark);
        b_cnt = vandq_u8(vsubq_u8(b_cnt, b), b);
        d_cnt = vandq_u8(vsubq_u8(d_cnt, d), d);
        b_max = vmaxq_u8(b_max, b_cnt);
        d_max = vmaxq_u8(d_max, d_cnt);
    }
    const uint8x16_t is_corner = vandq_u8(vcgtq_u8(vmaxq_u8(b_max, d_max), vdupq_n_u8(8)), valid);

    // Visit the corners: bit 4k+3 of the lane bits is set iff lane k is a corner
    uint64_t bits = laneBits(is_corner) & 0x8888888888888888ull;
    while (bits) {
        const int32_t x = __builtin_ctzll(bits) >> 2;
        bits &= bits - 1;
        if (nonmax_suppression) {
            corners[num_corners++] = j + x;
            cur[j + x] = cornerScore(ptr + x, offsets);
        } else {
            keypoints.emplace_back(j + x, y, uint8_t(0));
        }
    }
}

}  // namespace kfast_neon

// ---- The interface of KFAST.h ----

/**
 * @brief Rows start_row .. start_row + rows - 1; data points at row start_row
 *
 * first_thread / last_thread: as in KFAST.h, a stripe that is not the first (last) one skips
 * the non-max suppression of the margin rows it shares with the stripe above (below).
 */
template <const bool nonmax_suppression, const bool first_thread, const bool last_thread>
void _KFAST(const uint8_t* __restrict const data, const int32_t cols, const int32_t start_row, const int32_t rows,
            const int32_t stride, std::vector<Keypoint>& keypoints, const uint8_t threshold) {
    keypoints.reserve(8500);

    // Rosten's circle pixels in the order 9, 8, ..., 1, 16, ..., 10, then 9 .. 2 again (as in KFAST.h)
    const int32_t offsets[24] = {3 * stride, 3 * stride + 1, 2 * stride + 2, stride + 3, 3, -stride + 3,
                                 -2 * stride + 2, -3 * stride + 1, -3 * stride, -3 * stride - 1, -2 * stride - 2,
                                 -stride - 3, -3, stride - 3, 2 * stride - 2, 3 * stride - 1, 3 * stride,
                                 3 * stride + 1, 2 * stride + 2, stride + 3, 3, -stride + 3, -2 * stride + 2,
                                 -3 * stride + 1};
    const uint8x16_t t = vdupq_n_u8(threshold);
    const uint8x16_t all_lanes = vdupq_n_u8(0xFF);
    static const uint8_t kLaneIndex[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const uint8x16_t lane_index = vld1q_u8(kLaneIndex);

    // Windows cover candidate columns j..j+15 and read columns j-3..j+18. The last full window
    // that stays inside the row starts at cols - 19 and ends on the last candidate, cols - 4.
    const int32_t last_window = cols - 19;

    // Non-max suppression state: 3 rotating rows of scores and of corner columns (as in KFAST.h)
    std::vector<uint8_t> score_rows;
    std::vector<int32_t> corner_rows;
    uint8_t* rowbuf[3] = {};
    int32_t* cornerbuf[3] = {};
    if (nonmax_suppression) {
        score_rows.assign(3 * static_cast<size_t>(cols), 0);
        corner_rows.assign(3 * (static_cast<size_t>(cols) + 1), 0);
        for (int b = 0; b < 3; ++b) {
            rowbuf[b] = score_rows.data() + b * cols;
            cornerbuf[b] = corner_rows.data() + b * (cols + 1) + 1;  // [-1] holds the row's corner count
        }
    }

    for (int32_t i = 3; i < rows - 2; ++i) {
        const uint8_t* const row = data + i * stride;
        uint8_t* cur = nullptr;
        int32_t* corners = nullptr;
        int32_t num_corners = 0;
        if (nonmax_suppression) {
            cur = rowbuf[i % 3];
            corners = cornerbuf[i % 3];
            std::memset(cur, 0, cols);
        }

        if (i < rows - 3 && cols < 22) {
            // Too narrow for a 16-pixel window: test each candidate column on its own
            for (int32_t j = 3; j < cols - 3; ++j) {
                if (!kfast_neon::isCornerScalar(row + j, offsets, threshold)) continue;
                if (nonmax_suppression) {
                    corners[num_corners++] = j;
                    cur[j] = kfast_neon::cornerScore(row + j, offsets);
                } else {
                    keypoints.emplace_back(j, start_row + i, uint8_t(0));
                }
            }
        } else if (i < rows - 3) {
            int32_t j = 3;
            for (; j <= last_window; j += 16)
                kfast_neon::processWindow<nonmax_suppression>(row + j, j, all_lanes, offsets, t, num_corners,
                                                              corners, cur, keypoints, start_row + i);
            // Columns j..cols-4 are left: rerun the last full window, with the lanes already done masked off
            if (j < cols - 3) {
                const uint8x16_t not_done = vcgeq_u8(lane_index, vdupq_n_u8(static_cast<uint8_t>(j - last_window)));
                kfast_neon::processWindow<nonmax_suppression>(row + last_window, last_window, not_done, offsets, t,
                                                              num_corners, corners, cur, keypoints, start_row + i);
            }
        }

        if (nonmax_suppression) {
            corners[-1] = num_corners;

            // first thread: skip first and last row; inner threads: skip first, second, last row;
            // last thread: skip first and second row (as in KFAST.h)
            if ((!last_thread && i == rows - 3) || (!first_thread && i == 4) || i == 3) continue;

            const uint8_t* const last = rowbuf[(i - 1) % 3];
            const uint8_t* const last2 = rowbuf[(i - 2) % 3];
            const int32_t* const prev = cornerbuf[(i - 1) % 3];
            for (int32_t k = 0; k < prev[-1]; ++k) {
                const int32_t j = prev[k];
                const uint8_t score = last[j];
                // Keep a corner whose score beats all 8 neighbours; & instead of && avoids 8 branches
                if ((score > last[j - 1]) & (score > last[j + 1]) & (score > cur[j - 1]) & (score > cur[j]) &
                    (score > cur[j + 1]) & (score > last2[j - 1]) & (score > last2[j]) & (score > last2[j + 1])) {
                    keypoints.emplace_back(j, start_row + i - 1, score);
                }
            }
        }
    }
}

template <const bool multithreading, const bool nonmax_suppression>
void KFAST(const uint8_t* __restrict const data, const int32_t cols, const int32_t rows, const int32_t stride,
           std::vector<Keypoint>& keypoints, const uint8_t threshold) {
    static_assert(!multithreading, "the NEON port has no built-in threading: call _KFAST on your own threads");
    keypoints.clear();
    keypoints.reserve(8500);
    _KFAST<nonmax_suppression, true, true>(data, cols, 0, rows, stride, keypoints, threshold);
}
