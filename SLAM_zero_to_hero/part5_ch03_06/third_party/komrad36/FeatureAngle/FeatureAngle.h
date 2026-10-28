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

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

constexpr float PI = 3.1415927f;

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

	r = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
	x = _mm_add_epi16(x, _mm_mullo_epi16(r, xwt2));
	p += step;

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