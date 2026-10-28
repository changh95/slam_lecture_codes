/*******************************************************************
*   RGB2Y.h
*   RGB2Y
*
*	Author: Kareem Omar
*	kareem.h.omar@gmail.com
*	https://github.com/komrad36
*
*	Last updated Sep 29, 2018
*******************************************************************/
//
// Fastest CPU (AVX/SSE) implementation of RGB to grayscale.
// Roughly 3x faster than OpenCV's implementation with AVX2, or 2x faster
// than OpenCV's implementation if using SSE only.
//
// Converts an RGB color image to grayscale.
//
// You can use equal weighting by calling the templated
// function with weight set to 'false', or you
// can specify non-equal weights (only slightly slower).
//
// The default non-equal weights match OpenCV's default.
//
// For even more speed see the CUDA version:
// github.com/komrad36/CUDARGB2Y
//
// If you do not have AVX2, uncomment the #define NO_AVX_PLEASE below to
// use only SSE isntructions. NOTE THAT THIS IS ABOUT 50% SLOWER.
// A processor with full AVX2 support is highly recommended.
//
// All functionality is contained in RGB2Y.h.
// 'main.cpp' is a demo and test harness.
//

#pragma once

//#define NO_AVX_PLEASE

// Set your weights here.
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;

#include <algorithm>
#include <cstdint>
#include <future>
#include <immintrin.h>

// Internal; do NOT modify
constexpr uint16_t B_WT = static_cast<uint16_t>(32768.0 * B_WEIGHT + 0.5);
constexpr uint16_t G_WT = static_cast<uint16_t>(32768.0 * G_WEIGHT + 0.5);
constexpr uint16_t R_WT = static_cast<uint16_t>(32768.0 * R_WEIGHT + 0.5);

static const __m256i weight_vec = _mm256_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT);

template<bool weight>
void RGB2Y_ref(const uint8_t* __restrict const data, const int32_t cols, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	if (!weight) {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				const auto idx = 3 * (i*stride + j);
				out[i*stride + j] = (static_cast<uint32_t>(data[idx]) + static_cast<uint32_t>(data[idx + 1]) + static_cast<uint32_t>(data[idx + 2]) + 1) / 3;
			}
		}
	}
	else {
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				const auto idx = 3 * (i*stride + j);
				out[i*stride + j] = uint8_t(B_WEIGHT * data[idx] + 0.5f) + uint8_t(G_WEIGHT * data[idx + 1] + 0.5f) + uint8_t(R_WEIGHT * data[idx + 2] + 0.5f);
			}
		}
	}
}

template<bool last_row_and_col, bool weight>
void process(const uint8_t* __restrict const pt, const int32_t cols_minus_j, uint8_t* const __restrict out) {
	__m128i h3;
	if (weight) {
		__m256i in1 = _mm256_mulhrs_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt))), weight_vec);
		__m256i in2 = _mm256_mulhrs_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt + 15))), weight_vec);
		__m256i mul = _mm256_packus_epi16(in1, in2);
		__m256i b1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, -1, -1, 9, 12, -1, -1, -1, -1, -1, -1));
		__m256i g1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(1, 4, 7, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1));
		__m256i r1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1));
		__m256i accum = _mm256_adds_epu8(r1, _mm256_adds_epu8(b1, g1));
		h3 = _mm_adds_epu8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));
	}
	else {
		__m256i in1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(pt)));
		in1 = _mm256_inserti128_si256(in1, _mm_loadu_si128((const __m128i*)(pt + 15)), 1);
		__m256i b1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(0, -1, 3, -1, 6, -1, 9, -1, 12, -1, -1, -1, -1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, -1, -1, -1, -1, -1, -1));
		__m256i g1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1));
		__m256i r1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
		__m256i sum = _mm256_adds_epu16(r1, _mm256_adds_epu16(b1, g1));
		__m256i accum = _mm256_mulhrs_epi16(sum, _mm256_set1_epi16(10923));
		__m256i shuf = _mm256_shuffle_epi8(accum, _mm256_setr_epi8(0, 2, 4, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, -1, -1, -1, -1, -1, -1));
		h3 = _mm_or_si128(_mm256_extracti128_si256(shuf, 1), _mm256_castsi256_si128(shuf));
	}
	if (last_row_and_col) {
		switch (cols_minus_j) {
		case 15:
			out[14] = static_cast<uint8_t>(_mm_extract_epi8(h3, 14));
		case 14:
			out[13] = static_cast<uint8_t>(_mm_extract_epi8(h3, 13));
		case 13:
			out[12] = static_cast<uint8_t>(_mm_extract_epi8(h3, 12));
		case 12:
			out[11] = static_cast<uint8_t>(_mm_extract_epi8(h3, 11));
		case 11:
			out[10] = static_cast<uint8_t>(_mm_extract_epi8(h3, 10));
		case 10:
			out[9] = static_cast<uint8_t>(_mm_extract_epi8(h3, 9));
		case 9:
			out[8] = static_cast<uint8_t>(_mm_extract_epi8(h3, 8));
		case 8:
			out[7] = static_cast<uint8_t>(_mm_extract_epi8(h3, 7));
		case 7:
			out[6] = static_cast<uint8_t>(_mm_extract_epi8(h3, 6));
		case 6:
			out[5] = static_cast<uint8_t>(_mm_extract_epi8(h3, 5));
		case 5:
			out[4] = static_cast<uint8_t>(_mm_extract_epi8(h3, 4));
		case 4:
			out[3] = static_cast<uint8_t>(_mm_extract_epi8(h3, 3));
		case 3:
			out[2] = static_cast<uint8_t>(_mm_extract_epi8(h3, 2));
		case 2:
			out[1] = static_cast<uint8_t>(_mm_extract_epi8(h3, 1));
		case 1:
			out[0] = static_cast<uint8_t>(_mm_extract_epi8(h3, 0));
		}
	}
	else {
		_mm_storeu_si128(reinterpret_cast<__m128i*>(out), h3);
	}
}

template<bool last_row, bool weight>
void processRow(const uint8_t* __restrict pt, const int32_t cols, uint8_t* const __restrict out) {
	int j = 0;
	for (; j < cols - 10; j += 10, pt += 30) {
		process<false, weight>(pt, cols - j, out + j);
	}
	process<last_row, weight>(pt, cols - j, out + j);
}

template<bool weight>
void __forceinline _RGB2Y(const uint8_t* __restrict const data, const int32_t cols, const int32_t start_row, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	int i = start_row;
	for (; i < start_row + rows - 1; ++i) {
		processRow<false, weight>(data + 3 * i * stride, cols, out + i * cols);
	}
	processRow<true, weight>(data + 3 * i * stride, cols, out + i * cols);
}

template<bool multithread, bool weight>
void RGB2Y(const uint8_t* const __restrict image, const int width, const int height, const int stride, uint8_t* const __restrict out) {
	if (multithread) {
		const int32_t hw_concur = std::min(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
		if (hw_concur > 1) {
			std::vector<std::future<void>> fut(hw_concur);
			const int thread_stride = (height - 1) / hw_concur + 1;
			int i = 0, start = 0;
			for (; i < std::min(height - 1, hw_concur - 1); ++i, start += thread_stride) {
				fut[i] = std::async(std::launch::async, _RGB2Y<weight>, image, width, start, thread_stride, stride, out);
			}
			fut[i] = std::async(std::launch::async, _RGB2Y<weight>, image, width, start, height - start, stride, out);
			for (int j = 0; j <= i; ++j) fut[j].wait();
		}
		else {
			_RGB2Y<weight>(image, width, 0, height, stride, out);
		}
	}
	else {
		_RGB2Y<weight>(image, width, 0, height, stride, out);
	}
}

