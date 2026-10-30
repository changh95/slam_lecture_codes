/*
 * Functions for creating image pyramids on the CPU
 * pyramid_cpu.cpp
 *
 * Copyright (c) 2019-2020 Balazs Nagy,
 * Robotics and Perception Group, University of Zurich
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// [SLAM zero-to-hero] Design notes, added for the course. Comments that start with
// [SLAM zero-to-hero] are ours; the code itself is upstream's, unchanged.
//
// The CPU reference for the GPU pyramid: "vilib_CPU" in pyramid_benchmark, the README's
// "CPU, single thread" column. Each level is the 2x2 average of the one before, rounded down,
// the same integer arithmetic as pyramid_gpu.cu, which is why the benchmark finds them identical.
// image_halfsample_cpu() picks one of three implementations per level:
//  1. SSE2 (x86-64): needs 16-byte aligned input and output, a width that is a multiple of 16,
//     and rows without padding (cols == step). For EuRoC only 752 -> 376 qualifies; 376, 188
//     and 94 are not multiples of 16, so levels 2 to 4 take the scalar loop.
//  2. NEON (ARM, e.g. a Jetson): needs only a width that is a multiple of 16.
//  3. The scalar loop, for anything else. vilib builds with -O3 -march=native, and -O3 turns on
//     GCC's auto-vectorizer, so this loop may end up vectorized too (check with -fopt-info-vec).
// Both SIMD paths turn 16 bytes of each of two input rows into 8 output pixels:
//  - De-interleave: the even and odd bytes of a row have to end up in separate lanes. SSE2 has
//    no de-interleaving load, so it widens to 16-bit lanes with an AND mask (even bytes) and a
//    1-byte shift plus the mask (odd bytes). NEON's vld2_u8 de-interleaves while it loads.
//  - Widen before adding: 4 * 255 needs 10 bits, so the sums live in 16-bit lanes, are shifted
//    right by 2 and narrowed back to bytes (packus / vshrn).
//  - STRICT_CONFORMANCE: SSE2's pavgb / pavgw (_mm_avg_epu8/16) would average in fewer
//    instructions but compute (a + b + 1) >> 1, rounding up twice, so the result would differ
//    from the scalar and GPU code. The strict path pays for the widening to stay exact.
// Only 128-bit SSE2 is used, although -march=native allows AVX2 on the course machine. All paths
// are single-threaded.
#include <assert.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "vilib/preprocess/pyramid.h"
//CPU Architecture specific includes
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define PYRAMID_SIMD_ENABLE         1
/*
 * Note to future self:
 * the SSE avergage instructions introduce a +1 in the numerator,
 * making the vectorized version's output different from non-vectorized
 * implementation. When using strict conformance, we do averaging using
 * a simple shift operation instead of the avg instruction.
 */
#define STRICT_CONFORMANCE          1

namespace vilib {

static void image_halfsample_cpu(const cv::Mat & in, cv::Mat & out);
#if PYRAMID_SIMD_ENABLE
#if defined(__SSE2__)
static void image_halfsample16_cpu_sse2(const cv::Mat & in, cv::Mat & out);
static inline bool is_address_aligned16(const void * ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0F) == 0x00;
}
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
static void image_halfsample_cpu_neon(const cv::Mat & in, cv::Mat & out);
#endif /* defined(__SSE2__) || defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

void pyramid_create_cpu(const cv::Mat & h_image_in,
                        std::vector<cv::Mat> & h_image_pyramid,
                        unsigned int levels,
                        bool deep_copy) {
  assert(h_image_in.type() == CV_8U);
  assert(h_image_in.rows > 0);
  assert(h_image_in.cols > 0);
  assert(levels > 0);

  h_image_pyramid.resize(levels);
  if(deep_copy) {
    h_image_pyramid[0] = h_image_in.clone();
  } else {
    h_image_pyramid[0] = h_image_in;
  }
  // [SLAM zero-to-hero] Every call allocates new cv::Mats for levels 1 and up (the assignment
  // drops the previous ones), so the vilib_CPU row includes 4 host allocations per frame. Host
  // malloc is cheap; device allocation is not (see storage/pyramid_pool.cpp).
  for(unsigned int i=1;i<levels;++i) {
    h_image_pyramid[i] = cv::Mat(h_image_pyramid[i-1].rows/2,
                                 h_image_pyramid[i-1].cols/2,
                                 CV_8U);
    image_halfsample_cpu(h_image_pyramid[i-1],h_image_pyramid[i]);
  }
}

void pyramid_create_cpu(std::vector<cv::Mat> & h_image_pyramid) {
  assert(h_image_pyramid[0].type() == CV_8U);
  assert(h_image_pyramid[0].rows > 0);
  assert(h_image_pyramid[0].cols > 0);
  assert(h_image_pyramid.size() > 1);

  for(unsigned int i=1;i<h_image_pyramid.size();++i) {
    h_image_pyramid[i] = cv::Mat(h_image_pyramid[i-1].rows/2,
                                 h_image_pyramid[i-1].cols/2,
                                 CV_8U);
    image_halfsample_cpu(h_image_pyramid[i-1],h_image_pyramid[i]);
  }
}

static void image_halfsample_cpu(const cv::Mat & in, cv::Mat & out) {
  assert(in.rows/2 == out.rows);
  assert(in.cols/2 == out.cols);
#if PYRAMID_SIMD_ENABLE
#if defined(__SSE2__)
  // [SLAM zero-to-hero] _mm_load_si128 faults on an address that is not 16-byte aligned, hence the
  // address checks. OpenCV allocates Mat data aligned, so a whole Mat passes; an ROI of a wider
  // image fails cols == step, which the SSE2 loop needs because it walks rows with pointer += w.
  if(is_address_aligned16(in.data) &&
     is_address_aligned16(out.data) &&
     ((in.cols % 16) == 0) &&
     static_cast<std::size_t>(in.cols) == in.step) {
    image_halfsample16_cpu_sse2(in,out);
    return;
  }
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
  if((in.cols % 16) == 0) {
    image_halfsample_cpu_neon(in,out);
    return;
  }
#endif /* defined(__SSE2__) || defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

  const int in_stride = in.step.p[0];
  const int out_stride = out.step.p[0];
  uint8_t* top = reinterpret_cast<uint8_t*>(in.data);
  uint8_t* bottom = top + in_stride;
  uint8_t* end = top + in_stride*in.rows;
  uint8_t* p = reinterpret_cast<uint8_t*>(out.data);
  // [SLAM zero-to-hero] Integer / 4 of a non-negative sum is the same truncation as the GPU's >> 2.
  // `bottom < end` is only a second guard: with out.rows = in.rows / 2, as both callers allocate,
  // `y < out.rows` already stops before the odd last row, which is dropped.
  for (int y=0; y < out.rows && bottom < end; y++, top += in_stride*2, bottom += in_stride*2, p += out_stride)
  {
    for (int x=0; x < out.cols; x++)
    {
       p[x] = static_cast<uint8_t>((static_cast<uint16_t>(top[x*2]) + top[x*2+1] + bottom[x*2] + bottom[x*2+1])/4 );
    }
  }
}

#if PYRAMID_SIMD_ENABLE
#ifdef __SSE2__
static void image_halfsample16_cpu_sse2(const cv::Mat & in, cv::Mat & out) {
  const unsigned char * in_data = in.data;
  unsigned char * out_data = out.data;
  const unsigned char* nextRow = in_data + in.cols;
  const int w = in.cols;
  const int h = in.rows;
  int sw = w >> 4;
  int sh = h >> 1;
  // [SLAM zero-to-hero] 0x00FF in each 16-bit lane keeps the lane's low byte, which on a
  // little-endian CPU is the even pixel of each pair.
  __m128i m = _mm_set1_epi16(0x00FF);
  for (int i=0; i<sh; i++) {
    for (int j=0; j<sw; j++) {
      __m128i here = _mm_load_si128(reinterpret_cast<const __m128i*>(in_data));
      __m128i next = _mm_load_si128(reinterpret_cast<const __m128i*>(nextRow));
      // [SLAM zero-to-hero] (v & m) = even pixels, (v >> 1 byte) & m = odd pixels, as 16-bit
      // lanes; their sum is 8 horizontal pair sums per row. Add the two rows, >> 2: 8 averages.
#if STRICT_CONFORMANCE
      here = _mm_add_epi16(_mm_and_si128(here,m),_mm_and_si128(_mm_srli_si128(here,1), m));
      next = _mm_add_epi16(_mm_and_si128(next,m),_mm_and_si128(_mm_srli_si128(next,1), m));
      here = _mm_srli_epi16(_mm_add_epi16(here,next),2);
#else
      here = _mm_avg_epu8(here,next);
      next = _mm_and_si128(_mm_srli_si128(here,1), m);
      here = _mm_and_si128(here,m); 
      here = _mm_avg_epu16(here, next);
#endif /* STRICT_CONFORMANCE */
      /*
       * Note to future self:
       * with GCC-7, the proper unaligned store is not available.
       */
      // [SLAM zero-to-hero] packus narrows the 16-bit lanes to bytes (all are <= 255 already), and
      // _mm_storel_epi64 stores the low 8 bytes (movq, no alignment needed). The output is written
      // as one contiguous stream, so out must have no row padding either (true for a fresh Mat).
      _mm_storel_epi64((__m128i*)out_data, _mm_packus_epi16(here,here));
      in_data += 16;
      nextRow += 16;
      out_data += 8;
    }
    in_data += w;
    nextRow += w;
  }
}
#endif /* __SSE2__ */

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
void image_halfsample_cpu_neon(const cv::Mat & in, cv::Mat & out) {
  const int in_stride = in.step.p[0];
  const int out_stride = out.step.p[0];
  // [SLAM zero-to-hero] Pitfall: y visits every even row below in.rows. With an odd height the
  // last pass reads row in.rows and writes row out.rows, one row past each buffer (the scalar
  // loop stops at out.rows = in.rows / 2, the SSE2 one uses h >> 1). Example: the 240x135
  // level of a 1920x1080 frame, on a Jetson. vld2_u8 loads 16 bytes and de-interleaves them
  // into even and odd pixels; vaddl / vaddw widen to 16 bits while adding; vshrn shifts by 2 and
  // narrows.
  for (int y = 0; y < in.rows; y += 2) {
    const uint8_t * in_top = in.data + y*in_stride;
    const uint8_t * in_bottom = in.data + (y+1)*in_stride;
    uint8_t * out_data = out.data + (y >> 1)*out_stride;
    for (int x = in.cols; x > 0 ; x-=16, in_top += 16, in_bottom += 16, out_data += 8) {
      uint8x8x2_t top  = vld2_u8( (const uint8_t *)in_top );
      uint8x8x2_t bottom = vld2_u8( (const uint8_t *)in_bottom );
      uint16x8_t sum = vaddl_u8( top.val[0], top.val[1] );
      sum = vaddw_u8( sum, bottom.val[0] );
      sum = vaddw_u8( sum, bottom.val[1] );
      uint8x8_t final_sum = vshrn_n_u16(sum, 2);
      vst1_u8(out_data, final_sum);
    }
  }
}
#endif /* defined(__ARM_NEON__) || defined(__ARM_NEON) */
#endif /* PYRAMID_SIMD_ENABLE */

void pyramid_display(const std::vector<cv::Mat> & pyramid) {
  for(std::size_t l=0;l<pyramid.size();++l) {
    std::string subframe_title("Image pyramid (");
    subframe_title += std::to_string(pyramid[l].cols);
    subframe_title += "x";
    subframe_title += std::to_string(pyramid[l].rows);
    subframe_title += ")";
    cv::imshow(subframe_title.c_str(), pyramid[l]);
    cv::waitKey();
  }
}

} // namespace vilib