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

// [SLAM zero-to-hero] Design notes, added for the course. Comments that start with
// [SLAM zero-to-hero] are ours; the code itself is upstream's, unchanged.
//
// Gray = wB*B + wG*G + wR*R for each pixel of interleaved BGR bytes (B0 G0 R0 B1 G1 R1 ...).
// process() converts 10 pixels per call, and its design follows from four choices:
//  1. 16-bit fixed point, not float. The weights are stored as Q15 integers (weight * 2^15),
//     and _mm256_mulhrs_epi16 returns round(pixel * weight) in one instruction. A 256-bit
//     register holds 16 int16 lanes but only 8 floats, and no int <-> float conversion is needed.
//  2. Multiply first, de-interleave afterwards. weight_vec repeats B, G, R in the order of the
//     bytes in memory, so pixels are multiplied as loaded. The products (each <= 255) are then
//     packed back to bytes, where all 30 of them fit in one register for the shuffles.
//  3. 10 pixels = 2 x 15 bytes. 15 is the largest multiple of 3 that fits in a 16-byte load, so
//     both loads start on a B byte and share one weight vector, and each load widens to exactly
//     one 256-bit register.
//  4. One lane crossing, at the very end. AVX2 shuffles bytes only within each 128-bit half. The
//     shuffle masks leave every product in its half and put pixel k's products at byte k, so a
//     single extract + add of the two halves completes all 10 pixels.
// The price is memory slack: a group reads 31 bytes for its 30 and stores 16 bytes for its 10,
// so buffers need spare bytes after the image (see "Stores" in process()).

#pragma once

// [SLAM zero-to-hero] This version of the header has no SSE fallback: NO_AVX_PLEASE is not read
// anywhere, and the code below always needs AVX2 (plus SSE4.1 for _mm_extract_epi8).
//#define NO_AVX_PLEASE

// Set your weights here.
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;

#include <algorithm>
#include <cstdint>
#include <future>
#include <immintrin.h>

// [SLAM zero-to-hero] Q15: weight * 32768, rounded (3736, 19235, 9798). _mm256_mulhrs_epi16(a, w)
// computes (a * w + 2^14) >> 15, i.e. round(a * weight). The weights are int16 lanes, so each
// must be non-negative and round to at most 32767 (weight < 0.99998). Each product is rounded on
// its own, as in RGB2Y_ref, and for every byte value the Q15 product equals RGB2Y_ref's, so the
// two give identical images. OpenCV rounds the sum once instead (and uses 3735 for B, so its
// weights sum to exactly 32768), which is why cv::cvtColor differs from both by +-1 on some
// pixels.
// Internal; do NOT modify
constexpr uint16_t B_WT = static_cast<uint16_t>(32768.0 * B_WEIGHT + 0.5);
constexpr uint16_t G_WT = static_cast<uint16_t>(32768.0 * G_WEIGHT + 0.5);
constexpr uint16_t R_WT = static_cast<uint16_t>(32768.0 * R_WEIGHT + 0.5);

// [SLAM zero-to-hero] One weight per byte of a 16-byte load that starts at a B byte:
// B G R B G R B G R B G R B G R B. Both loads in process() start at a B byte, so this one
// constant serves both and stays in a register.
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
		// [SLAM zero-to-hero] Load 16 bytes at pt and at pt + 15: pixels 0-4 and 5-9, each followed by
		// the next pixel's B. pt + 15 is again a B byte, so weight_vec lines up with both loads.
		// Widening u8 -> i16 (vpmovzxbw) turns each 16-byte load into exactly one 256-bit register.
		// The second load ends at pt + 30, one byte past this group's 30 bytes.
		__m256i in1 = _mm256_mulhrs_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt))), weight_vec);
		__m256i in2 = _mm256_mulhrs_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pt + 15))), weight_vec);
		// [SLAM zero-to-hero] The products are <= 255, so packing them to bytes loses nothing, and all
		// 32 products (30 are used) fit in one register. Like most AVX2 instructions, packus works on
		// each 128-bit half separately, so the bytes come out in this order:
		//   mul[ 0.. 7] = in1 products 0-7    B0 G0 R0 B1 G1 R1 B2 G2
		//   mul[ 8..15] = in2 products 0-7    B5 G5 R5 B6 G6 R6 B7 G7
		//   mul[16..23] = in1 products 8-15   R2 B3 G3 R3 B4 G4 R4 B5
		//   mul[24..31] = in2 products 8-15   R7 B8 G8 R8 B9 G9 R9 B10
		__m256i mul = _mm256_packus_epi16(in1, in2);
		// [SLAM zero-to-hero] vpshufb cannot move a byte between the two 128-bit halves. Doing that
		// would take extra lane-crossing permutes (vpermq moves whole qwords only, with 3 cycles of
		// latency on Intel). Instead each mask keeps a product in the half it is in and moves pixel
		// k's product to byte k of that half; -1 writes 0. For B: B0, B1, B2, B6, B7 end up in the low
		// half (bytes 0, 1, 2, 6, 7), B3, B4, B5, B8, B9 in the high half (bytes 3, 4, 5, 8, 9).
		// In2's copy of B5, and B10 (the next group's first byte), are not used.
		__m256i b1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, -1, -1, 9, 12, -1, -1, -1, -1, -1, -1));
		__m256i g1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(1, 4, 7, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1));
		__m256i r1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1));
		// [SLAM zero-to-hero] Add B + G + R inside each half, then add the low half to the high half
		// (the kernel's one lane crossing, vextracti128): byte k of h3 is gray pixel k (k = 0..9), and
		// bytes 10-15 are 0. A saturating add is one instruction like a wrapping one, and clamps at 255
		// instead of wrapping around if the rounded products of custom weights add up to more.
		__m256i accum = _mm256_adds_epu8(r1, _mm256_adds_epu8(b1, g1));
		h3 = _mm_adds_epu8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));
	}
	else {
		// [SLAM zero-to-hero] Equal weights, (B + G + R) / 3. Here the two 16-byte loads (15 bytes used
		// each) go into the two 128-bit halves of one register, 5 pixels each, so the same in-half
		// shuffle extracts a channel from both halves at once. The -1 entries zero every other byte,
		// so each shuffle also widens its channel to 16 bits (sums reach 765).
		__m256i in1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(pt)));
		in1 = _mm256_inserti128_si256(in1, _mm_loadu_si128((const __m128i*)(pt + 15)), 1);
		__m256i b1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(0, -1, 3, -1, 6, -1, 9, -1, 12, -1, -1, -1, -1, -1, -1, -1, 0, -1, 3, -1, 6, -1, 9, -1, 12, -1, -1, -1, -1, -1, -1, -1));
		__m256i g1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1, 1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1));
		__m256i r1 = _mm256_shuffle_epi8(in1, _mm256_setr_epi8(2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1, 2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1));
		__m256i sum = _mm256_adds_epu16(r1, _mm256_adds_epu16(b1, g1));
		// [SLAM zero-to-hero] Division by 3 as a fixed-point multiply: 10923 = round(2^15 / 3), and
		// mulhrs computes (sum * 10923 + 2^14) >> 15. x86 SIMD has no integer divide, and a float
		// divide would need conversions both ways.
		__m256i accum = _mm256_mulhrs_epi16(sum, _mm256_set1_epi16(10923));
		// [SLAM zero-to-hero] Keep the low byte of each 16-bit result: the low half's 5 pixels go to
		// bytes 0-4, the high half's to bytes 5-9 of its own half. The two sets do not overlap, so an
		// OR of the two halves merges them.
		__m256i shuf = _mm256_shuffle_epi8(accum, _mm256_setr_epi8(0, 2, 4, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, -1, -1, -1, -1, -1, -1));
		h3 = _mm_or_si128(_mm256_extracti128_si256(shuf, 1), _mm256_castsi256_si128(shuf));
	}
	// [SLAM zero-to-hero] Stores. Every full group stores its 10 pixels and 6 zeros; the next group's
	// store overwrites the zeros. A row's last group of n pixels stores 16 - n bytes of overhang
	// into the next row. Rows are converted in order, so the next row overwrites that too, and the
	// row loop needs no tail code. Only the last group of the last row of a call writes exactly its
	// 1..10 pixels, byte by byte with pextrb (_mm_extract_epi8). The group before it still stores 6
	// bytes past its own 10 pixels, so for cols >= 8, when cols % 10 is 1..5, the call writes
	// 6 - cols % 10 zero bytes past its last row: into the next row, which may belong to another
	// thread, or past the end of the output buffer. In images narrower than 8 pixels an earlier
	// row's 16-byte store reaches further, up to 16 - 2 * cols bytes past the end.
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

// [SLAM zero-to-hero] 10 pixels (30 input bytes) per step. The loop stops with 1..10 pixels left,
// and those go through process<last_row>.
template<bool last_row, bool weight>
void processRow(const uint8_t* __restrict pt, const int32_t cols, uint8_t* const __restrict out) {
	int j = 0;
	for (; j < cols - 10; j += 10, pt += 30) {
		process<false, weight>(pt, cols - j, out + j);
	}
	process<last_row, weight>(pt, cols - j, out + j);
}

// [SLAM zero-to-hero] Converts rows start_row .. start_row + rows - 1: the per-thread worker of the
// multi-threaded mode below, and the function this exercise calls from FastThreadPool workers.
// Input rows are 3 * stride bytes apart, output rows cols bytes apart.
template<bool weight>
void __forceinline _RGB2Y(const uint8_t* __restrict const data, const int32_t cols, const int32_t start_row, const int32_t rows, const int32_t stride, uint8_t* const __restrict out) {
	int i = start_row;
	for (; i < start_row + rows - 1; ++i) {
		processRow<false, weight>(data + 3 * i * stride, cols, out + i * cols);
	}
	processRow<true, weight>(data + 3 * i * stride, cols, out + i * cols);
}

// [SLAM zero-to-hero] multithread = true starts n = min(hardware threads, height / 16) std::async
// tasks on every call, each of ceil(height / n) rows. The last task gets what is left, which can
// be fewer than 16 rows, or negative, so that the chunks run past the image (../README.md,
// "Multi-threading"). Starting and joining the threads can also cost more than the conversion
// itself: 0.65 ms for 32 threads vs 0.21 ms for a single-threaded 1080p frame on the README's
// Ryzen 9 7950X. The exercise therefore runs _RGB2Y on persistent FastThreadPool workers.
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

