/*******************************************************************
*   FeatureAngle.h
*   FeatureAngle
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Dec 27, 2016
*******************************************************************/
//
// Extremely fast and accurate vectorized (SSE) gradient direction
// (angle of rotation of a feature) finder for computer vision.
//
// Operates on a 7x7 patch, i.e. a standard FAST feature. Larger
// scale spaces can also call this function on the interpolated
// image data for accurate angle-finding at larger scale.
//
// Used in KORAL detector-descriptor pipeline.
// 
// Simply call like featureAngle(image, x, y, step)
// where 'image' is a pointer to uint8_t (grayscale) image data,
// 'x' and 'y' are the horizontal and vertical coordinates of the center
// of the feature, and 'step' is the row step (pitch) in bytes.
//
// A float representing the angle *IN RADIANS*, between -PI and PI,
// is returned.
//

// [SLAM zero-to-hero] Design notes, added for the course. Comments that start with
// [SLAM zero-to-hero] are ours; the code itself is upstream's, unchanged.
//
// The angle is ORB's intensity centroid, atan2(m01, m10), with m10 = sum(x * I) and
// m01 = sum(y * I) over the circular patch drawn below (x, y relative to the center).
// Keypoints are scattered across the image, so vectorizing across keypoints would need gathers.
// FeatureAngle vectorizes inside one patch instead:
//  1. One register per patch row. A row is 7 pixels, so one 8-byte load (movq) widened to
//     8 x int16 (pmovzxbw) holds it. The 8th lane, the pixel right of the patch, gets weight 0.
//  2. The circle is folded into the weights. Pixels outside it get x and y weights of 0, so
//     there is no branch and no per-row extent table (ORB's u_max).
//  3. y is constant along a row, so rows above the center subtract |y| * I and rows below add
//     it: 3 x-weight and 3 y-weight vectors, 6 constants in all, cover the 7 rows.
//  4. Exact 16-bit integer math. Over the mask, |x| sums to 52 and so does |y|, so no partial
//     sum can exceed 52 * 255 = 13260 in magnitude (m10 and m01 themselves stay within
//     26 * 255 = 6630), far inside int16. That allows mullo_epi16 + add_epi16 on 8 lanes per
//     register, instead of 32-bit or float math on 4.
//  5. SSE (128-bit), not AVX2. A 7-pixel row already fills an 8-lane int16 register. AVX2 could
//     hold two rows at once, but packing rows into the two halves costs an insert per pair and a
//     final cross-half add, for a patch of only 7 rows. SSE4.1 also runs on pre-AVX2 CPUs.
//  6. The lanes are summed once per patch (3 steps for 8 lanes), not once per row.
//  7. atan2 is a polynomial, not libm (see fastAtan2).
// This works because the patch is small: ORB's 31x31 patch would overflow int16 and need 32-bit
// lanes (README exercise 4).

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

constexpr float PI = 3.1415927f;

// [SLAM zero-to-hero] Scalar, called once per keypoint. atan(c) on [0, 1] is approximated by an
// odd 7th-degree polynomial in c = min(|x|, |y|) / max(|x|, |y|); pi/2 - a and the signs of x and
// y extend it to [-pi, pi]. FLT_MIN avoids 0 / 0 on a flat patch. The maximum error is 1.7e-4 rad
// (0.01 deg), at 45 deg. cv::fastAtan2 (OpenCV 4.x) uses the same coefficients, scaled to degrees,
// with range [0, 360). After converting units, this matches ORB's IC_Angle on the same 7x7 patch
// (half_k = 3, as in the course's src/plain_kernels.cpp) up to float rounding.
float fastAtan2(float y, float x) {
	const float ax = fabs(x);
	const float ay = fabs(y);
	float a;
	if (ax >= ay) {
		const float c = ay / (ax + FLT_MIN);
		const float cc = c*c;
		a = (((-0.0443265555479f*cc + 0.1555786518f)*cc - 0.325808397f)*cc + 0.9997878412f)*c;
	}
	else {
		const float c = ax / (ay + FLT_MIN);
		const float cc = c*c;
		a = PI * 0.5f - (((-0.0443265555479f*cc + 0.1555786518f)*cc - 0.325808397f)*cc + 0.9997878412f)*c;
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

// [SLAM zero-to-hero] Weights per patch row, one int16 lane per column (lane 7 holds the extra
// loaded byte; its weight is always 0). xwt holds x = -3..3 inside the circle and 0 outside; ywt
// holds |y| inside the circle. Rows 0 and 6 use *wt0, rows 1 and 5 *wt1, rows 2-4 *wt2; row 3
// (y = 0) has no y term.
static const __m128i xwt0 = _mm_setr_epi16(0, 0, -1, 0, 1, 0, 0, 0);
static const __m128i xwt1 = _mm_setr_epi16(0, -2, -1, 0, 1, 2, 0, 0);
static const __m128i xwt2 = _mm_setr_epi16(-3, -2, -1, 0, 1, 2, 3, 0);

static const __m128i ywt0 = _mm_setr_epi16(0, 0, 3, 3, 3, 0, 0, 0);
static const __m128i ywt1 = _mm_setr_epi16(0, 2, 2, 2, 2, 2, 0, 0);
static const __m128i ywt2 = _mm_setr_epi16(1, 1, 1, 1, 1, 1, 1, 0);

float featureAngle(const uint8_t* const __restrict image, const int px, const int py, const int step) {
	const uint8_t* __restrict p = image + (py - 3)*step + (px - 3);
	__m128i x = _mm_setzero_si128();
	__m128i y = _mm_setzero_si128();

	// [SLAM zero-to-hero] Row 0 (y = -3). movq loads 8 bytes, pmovzxbw widens them to 8 x int16, and
	// mullo multiplies by the weights (exact: |products| <= 3 * 255). y < 0 above the center, so the
	// y term is subtracted.
	__m128i r;
	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt0));
	y = _mm_sub_epi16(y, _mm_mullo_epi16(r, ywt0));
	p += step;

	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt1));
	y = _mm_sub_epi16(y, _mm_mullo_epi16(r, ywt1));
	p += step;

	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt2));
	y = _mm_sub_epi16(y, _mm_mullo_epi16(r, ywt2));
	p += step;

	// [SLAM zero-to-hero] Row 3, the center row: y = 0, so only the x term.
	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt2));
	p += step;

	// [SLAM zero-to-hero] Rows 4-6 (y = 1..3) mirror rows 2-0, with the y term added.
	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt2));
	y = _mm_add_epi16(y, _mm_mullo_epi16(r, ywt2));
	p += step;

	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt1));
	y = _mm_add_epi16(y, _mm_mullo_epi16(r, ywt1));
	p += step;

	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt0));
	y = _mm_add_epi16(y, _mm_mullo_epi16(r, ywt0));

	// [SLAM zero-to-hero] Horizontal sum of 8 int16 lanes in 3 steps: add the swapped 64-bit halves
	// (shuffle_epi32 78 = dwords 2,3,0,1), add adjacent pairs (hadd), then swap words 0 and 1
	// (shufflelo 225 = words 1,0,2,3) and add. Word 0 holds the total, read back as int16.
	x = _mm_add_epi16(x, _mm_shuffle_epi32(x, 78));
	x = _mm_hadd_epi16(x, x);
	x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, 225));
	const float x_sum = static_cast<float>(static_cast<int16_t>(_mm_cvtsi128_si32(x)));

	y = _mm_add_epi16(y, _mm_shuffle_epi32(y, 78));
	y = _mm_hadd_epi16(y, y);
	y = _mm_add_epi16(y, _mm_shufflelo_epi16(y, 225));
	const float y_sum = static_cast<float>(static_cast<int16_t>(_mm_cvtsi128_si32(y)));

	return fastAtan2(y_sum, x_sum);
}