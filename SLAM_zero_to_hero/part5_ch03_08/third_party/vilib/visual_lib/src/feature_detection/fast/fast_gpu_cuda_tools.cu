/*
 * FAST feature detector utilities in CUDA
 * fast_gpu_cuda_tools.cu
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
// This file computes the FAST corner response of one image: one float per pixel, 0 for "no corner",
// the corner score otherwise. detector_base_gpu_cuda_tools.cu then keeps the best corner per grid
// cell. A pixel with value c is a FAST-N corner (N = min_arc_length, 9 in fast_benchmark) if N
// contiguous pixels of the 16-pixel Bresenham circle of radius 3 around it are all darker than
// c - t or all brighter than c + t (t = threshold, 10 in fast_benchmark).
//
// Mapping: one thread per pixel, 2D blocks of 128 threads (4 warps): 32x4, or 16x8 when the width
// is not a multiple of 32. EuRoC's 752x480 gets 16x8 blocks in a 47x60 grid, one thread per pixel.
// Memory:
//  - Image: global memory, pitched (cudaMallocPitch in Subframe: every row starts on an aligned
//    address), read through a const __restrict__ pointer. Neighbouring threads read neighbouring
//    bytes, so each read of a warp touches only a few 32-byte sectors. Every byte is read by up to
//    17 threads (as a centre or as a circle point); the repeats are served from cache. Upstream
//    tried shared memory and found it slower: "the amount of data reuse is not enough to overcome
//    the shm loading/synchronization overheads" (fast_gpu_config.h).
//  - Response: global memory, pitched floats. Each thread stores 0, and a corner then its score.
//  - Lookup table: 8 KB in global memory (cudaMalloc in fast_gpu.cpp), read at a per-thread
//    address. Not constant memory, which serializes when the lanes of a warp read different
//    addresses (fast_gpu_config.h).
//  - Registers: the 16 circle pixels (px[16]), both masks, the score. This file uses no shared
//    memory, no textures and no constant memory.
// Techniques:
//  1. An exact pre-test on two opposite pairs of circle pixels lets most threads exit after reading
//     2 or 4 of them (fast_gpu_prechecks).
//  2. Branch-free classification: the sign bit of a float difference says darker or brighter, and
//     `cond ? (1<<i) : 0` compiles without a branch (the sign bit is shifted straight into bit i),
//     so all lanes run the same instructions.
//  3. Fully unrolled loops with compile-time indices: circle offsets reduce to pitch expressions
//     computed once per thread, and px[16] stays in registers instead of local memory.
//  4. The arc test is one table lookup (65536 bits, one per 16-bit mask) instead of a loop whose
//     trip count depends on the data and would make the lanes of a warp diverge. The read of one
//     table word is issued as soon as its address is known, so its latency overlaps the last
//     compares (the other word: see the note at the load).
//  5. Only corners compute a score, and the score type is a template parameter: each kernel
//     instantiation holds one score's code and no run-time switch.
// Error checks: CMakeLists.txt passes -DNDEBUG to nvcc, so CUDA_KERNEL_CHECK() (cuda_common.h) is
// empty and a failed launch goes unreported. Use compute-sanitizer (README, exercise 2) or your own
// cudaGetLastError() when you change a kernel. No __launch_bounds__ is given, so nvcc chooses the
// register count; nvcc -Xptxas -v prints it.
#include "vilib/feature_detection/config.h"
#include "vilib/feature_detection/fast/fast_gpu_cuda_tools.h"
#include "vilib/cuda_common.h"

namespace vilib {

// [SLAM zero-to-hero] Offset of circle point i from the centre, in bytes. Every caller passes a
// compile-time i (a literal, or the index of a fully unrolled loop), so the compiler resolves the
// if-chain: each offset becomes a fixed expression in the run-time pitch (e.g. 3*pitch - 1),
// computed once per thread, and the reads use a few row addresses plus small immediate column
// offsets. Upstream found this faster than reading the
// offsets from a table in constant memory (fast_gpu_config.h). Point 0 is 3 rows below the centre,
// 4 is left, 8 above, 12 right, and consecutive indices are neighbours on the circle: bit i of the
// masks below is point i, so "contiguous on the circle" means "contiguous bits".
__inline__ __device__ int bresenham_circle_offset_pitch(const int & i,
                                                        const int & pitch,
                                                        const int & pitch2,
                                                        const int & pitch3) {
  /*
   * Note to future self and others:
   * this function is only should be called in for loops that are unrolled.
   * Due to unrollment, the if else structure disappears, and the offsets get
   * substituted.
   *
   * Order within the circle:
   *
   *      7 8  9
   *    6       10
   *  5           11
   *  4     x     12
   *  3           13
   *   2        14
   *     1 0 15
   */
  int offs=0;
  if(i==0)
    offs=pitch3;
  else if(i==1)
    offs=pitch3-1;
  else if(i==2)
    offs=pitch2-2;
  else if(i==3)
    offs=pitch-3;
  else if(i==4)
    offs=-3;
  else if(i==5)
    offs=-pitch-3;
  else if(i==6)
    offs=-pitch2-2;
  else if(i==7)
    offs=-pitch3-1;
  else if(i==8)
    offs=-pitch3;
  else if(i==9)
    offs=-pitch3+1;
  else if(i==10)
    offs=-pitch2+2;
  else if(i==11)
    offs=-pitch+3;
  else if(i==12)
    offs=3;
  else if(i==13)
    offs=pitch+3;
  else if(i==14)
    offs=pitch2+2;
  else if(i==15)
    offs=pitch3+1;
  return offs;
}

// [SLAM zero-to-hero] The arc test as a loop. As shipped (FAST_GPU_USE_LOOKUP_TABLE 1) it only
// fills the lookup table. The 16-bit mask is copied into both halves of a 32-bit word, so an arc
// that wraps from point 15 to point 0 becomes one contiguous run of ones. __popc (count of ones)
// is one instruction (POPC), __clz (count of leading zeros) two (FLO, find leading one, then 31
// minus it). The loop runs once per run of ones, a
// number that differs from pixel to pixel: inside the corner kernel the lanes of a warp would leave
// it at different times, and the warp would run until its slowest lane is done.
__inline__ __device__ unsigned char fast_gpu_is_corner(const unsigned int & address,
                                                       const int & min_arc_length) {
  int ones = __popc(address);
  if(ones < min_arc_length) { // if we dont have enough 1-s in the address, dont even try
    return 0;
  }
  unsigned int address_dup = address|(address<<16); //duplicate the low 16-bits at the high 16-bits
  while(ones > 0) {
    address_dup <<= __clz(address_dup); // shift out the high order zeros
    int lones = __clz(~address_dup); // count the leading ones
    if(lones >= min_arc_length) {
      return 1;
    }
    address_dup <<= lones; // shift out the high order ones
    ones -= lones;
  }
  return 0;
}

// [SLAM zero-to-hero] Pre-test. It returns 1 to REJECT the pixel. Points 4 and 12 (left, right) are
// opposite: the other 14 points form two runs of 7, so every arc of 8 or more contiguous points
// contains 4 or 12. If both are "similar" (between c - t and c + t), no such arc can be all darker
// or all brighter, and the pixel is not a corner. Points 0 and 8 (below, above) give a second
// chance to reject. The test is exact for N >= 8, so for vilib's 9 to 12, but would drop real
// corners for N <= 7, and the kernel runs it whatever min_arc_length is (see fast_gpu.h).
__inline __device__ unsigned int fast_gpu_prechecks(const float & c_t,
                                                    const float & ct,
                                                    const unsigned char * image_ptr,
                                                    const int & image_pitch,
                                                    const int & image_pitch2,
                                                    const int & image_pitch3) {
  /*
   * Note to future self:
   * using too many prechecks of course doesnt help
   */
  // (-3,0) (3,0) -> 4,12
  float px0 = (float)image_ptr[bresenham_circle_offset_pitch(4,image_pitch,image_pitch2,image_pitch3)];
  float px1 = (float)image_ptr[bresenham_circle_offset_pitch(12,image_pitch,image_pitch2,image_pitch3)];
  // [SLAM zero-to-hero] The four sign bits are all 0 only when both pixels are similar. A bitwise
  // OR of them gives one branch instead of four.
  if((signbit(px0-c_t)|signbit(px1-c_t)|signbit(ct-px0)|signbit(ct-px1))==0) {
    return 1;
  }
  // (0,3), (0,-3) -> 0, 8
  px0 = (float)image_ptr[bresenham_circle_offset_pitch(0,image_pitch,image_pitch2,image_pitch3)];
  px1 = (float)image_ptr[bresenham_circle_offset_pitch(8,image_pitch,image_pitch2,image_pitch3)];
  if((signbit(px0-c_t)|signbit(px1-c_t)|signbit(ct-px0)|signbit(ct-px1))==0) {
    return 1;
  }
  return 0;
}

// [SLAM zero-to-hero] The lookup table is built once per FASTGPU object (in its constructor), so
// the divergent loop above costs nothing per frame. Bit layout (FAST_GPU_USE_LOOKUP_TABLE_BITBASED
// 1, as shipped): 2048 words of 32 bits. Bit i of word x answers "is mask (i << 11) | x a corner?",
// so a 16-bit mask m is bit (m >> 11) of word (m & 0x7FF). 65536 bits are 8 KB, against 64 KB for
// one byte per mask; likely chosen so that more of the table stays in cache (upstream: "it did not
// help significantly", fast_gpu_config.h). Each thread computes one word; consecutive threads store
// consecutive words, a coalesced write.
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
__global__ void fast_gpu_calculate_lut_kernel(unsigned int * __restrict__ d_corner_lut,
                                              const int min_arc_length) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // 11 bits come from here
  unsigned int output = 0;

  #pragma unroll
  for (int i=0;i<32;++i) {
    // the top 5 bits come from here
    // in total, 2^16 combinations
    const int unified_address = (i<<11) | x;
    output |= fast_gpu_is_corner(unified_address,min_arc_length)<<i;
  }
  d_corner_lut[x] = output;
}

__host__ void fast_gpu_calculate_lut(unsigned int * d_corner_lut,
                                     const int & min_arc_length,
                                     cudaStream_t stream) {
  // [SLAM zero-to-hero] 2048 threads in 8 blocks of 256. Nothing here waits for the kernel; the
  // first detect() runs after it because both are queued on the same stream (see fast_gpu.cpp).
  // every thread writes 4 bytes: in total 64kbits get written
  kernel_params_t p = cuda_gen_kernel_params_1d(2048,256);
  fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid,p.threads_per_block,0,stream>>>
                                                   ((unsigned int*)d_corner_lut,
                                                    min_arc_length);
  CUDA_KERNEL_CHECK();
}
#else
__global__ void fast_gpu_calculate_lut_kernel(unsigned char * __restrict__ d_corner_lut,
                                              const int min_arc_length) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // all 16 bits come from here
  d_corner_lut[x] = fast_gpu_is_corner(x,min_arc_length);
}

__host__ void fast_gpu_calculate_lut(unsigned char * d_corner_lut,
                                     const int & min_arc_length,
                                     cudaStream_t stream) {
  // every thread writes a byte: in total 64kB gets written
  kernel_params_t p = cuda_gen_kernel_params_1d(64*1024,256);
  fast_gpu_calculate_lut_kernel<<<p.blocks_per_grid,p.threads_per_block,0,stream>>>
                                                   (d_corner_lut,
                                                    min_arc_length);
  CUDA_KERNEL_CHECK();
}
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

// [SLAM zero-to-hero] The kernel's corner test again, for another threshold, from the 16 pixel
// values already in registers: no image reads. Only the MAX_THRESHOLD score calls it.
__inline__ __device__ int fast_gpu_is_corner_quick(
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                  const unsigned int * __restrict__ d_corner_lut,
#else
                                  const unsigned char * __restrict__ d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                  const float * __restrict__ px,
                                  const float & center_value,
                                  const float & threshold,
                                  unsigned int & dark_diff_address,
                                  unsigned int & bright_diff_address
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                  ,
                                  unsigned int & dark_diff_bit,
                                  unsigned int & dark_diff_data,
                                  unsigned int & bright_diff_bit,
                                  unsigned int & bright_diff_data
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                ) {
  const float ct = center_value + threshold;
  const float c_t = center_value - threshold;
  dark_diff_address = 0;
  bright_diff_address = 0;

#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  dark_diff_bit = 0;
  bright_diff_bit = 0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

  #pragma unroll 16
  for(int i=0;i<16;++i) {
    int darker = signbit(px[i]-c_t);
    int brighter = signbit(ct-px[i]);
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
    if(i<11) {
      dark_diff_address   += darker?(1<<i):0;
      bright_diff_address += brighter?(1<<i):0;
    } else {
      if(i==11) {
        // initiate a readout
        dark_diff_data = d_corner_lut[dark_diff_address];
        bright_diff_data = d_corner_lut[bright_diff_address];
      }
      dark_diff_bit   += darker?(1<<(i-11)):0;
      bright_diff_bit += brighter?(1<<(i-11)):0;
    }
#else
    dark_diff_address   += signbit(px[i]-c_t)?(1<<i):0;
    bright_diff_address += signbit(ct-px[i])?(1<<i):0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
  }
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  return (dark_diff_data&(1<<dark_diff_bit)) | (bright_diff_data&(1<<bright_diff_bit));
#else
  return (d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]);
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
}

// [SLAM zero-to-hero] SCORE is a template parameter, so every `if(SCORE == ...)` in the kernel is
// decided at compile time: each of the three instantiations holds one score's code, with no
// run-time branch and no registers spent on the other two. The host function below picks the
// instantiation with a switch. With a plain `if` (not C++17 `if constexpr`) every branch must still
// compile in every instantiation.
template <fast_score SCORE>
__global__ void fast_gpu_calc_corner_response_kernel(
                                       const int image_width,
                                       const int image_height,
                                       const int image_pitch,
                                       const unsigned char * __restrict__ d_image,
                                       const int horizontal_border,
                                       const int vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                       const unsigned int * __restrict__ d_corner_lut,
#else
                                       const unsigned char * __restrict__ d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                       const float threshold,
                                       const int min_arc_length,
                                       const int response_pitch_elements,
                                       float * __restrict__ d_response) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // thread id X
  const int y = blockDim.y*blockIdx.y + threadIdx.y; // thread id Y
  // [SLAM zero-to-hero] The grid is rounded up to whole blocks, so threads past the right or bottom
  // edge do nothing. With 16x8 blocks on 752x480 there are none.
  if(x < image_width && y < image_height) {
    const int resp_offset = y*response_pitch_elements + x;
    // [SLAM zero-to-hero] Every pixel of the image gets a value, border and rejected pixels
    // included: 0 here, overwritten by the score if the pixel is a corner. The buffer is reused
    // every frame (and starts uninitialized from cudaMallocPitch), so this store clears the last
    // frame's scores and lets the grid NMS read any neighbour without bounds checks. Threads with
    // consecutive x store consecutive floats: coalesced.
    d_response[resp_offset] = 0.0f;
    // [SLAM zero-to-hero] These borders are FASTGPU's det_*_border_: at least 3, the circle radius,
    // so the 16 reads never leave the image and need no checks of their own. They are normally one
    // pixel (the NMS radius) closer to the edge than the region the NMS searches: 7 here for
    // fast_benchmark's border of 8 (fast_gpu.cpp).
    if((x >= horizontal_border) &&
       (y >= vertical_border) &&
       (x < (image_width - horizontal_border)) &&
       (y < (image_height - vertical_border))) {
      const unsigned char * d_image_ptr = d_image + y*image_pitch + x;
      const float c = (float)(*d_image_ptr);
      const float ct = c + threshold;
      const float c_t = c - threshold;
      /*
       * Note to future self:
       * we need to create 2 differences for each of the 16 pixels
       * have 1 lookup table, and look-up both values
       *
       * c_t stands for: c - threshold (epsilon)
       * ct stands for : c + threshold (epsilon)
       *
       * Label of px:
       * - darker  if   px < c_t              (1)
       * - similar if   c_t <= px <= ct      (2)
       * - brighter if  ct < px             (3)
       *
       * Darker diff: px - c_t
       * sign will only give 1 in case of (1), and 0 in case of (2),(3)
       *
       * Similarly, brighter diff: ct - px
       * sign will only give 1 in case of (3), and 0 in case of (2),(3)
       */
      unsigned int dark_diff_address = 0;
      unsigned int bright_diff_address = 0;
#if FAST_GPU_USE_LOOKUP_TABLE && FAST_GPU_USE_LOOKUP_TABLE_BITBASED
      unsigned int dark_diff_bit = 0;
      unsigned int bright_diff_bit = 0;
      unsigned int dark_diff_data = 0;
      unsigned int bright_diff_data = 0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */

      // [SLAM zero-to-hero] Computed once per thread with shifts and an add; upstream measured that
      // this helps (fast_gpu_config.h).
      // Precalculate pitches
      const int image_pitch2 = image_pitch << 1;
      const int image_pitch3 = image_pitch + (image_pitch << 1);

      // [SLAM zero-to-hero] Early exit for most pixels. A warp whose 32 pixels are all rejected
      // retires here; in a warp with survivors, the rejected lanes sit idle until the others finish
      // (divergence). Returning early is safe because this kernel has no __syncthreads() and no
      // warp shuffle that every thread must reach; the grid NMS kernel has both and cannot do this.
      // Upstream's note: -10 usec for the pre-test (fast_gpu_config.h). The source reads those
      // pixels again below (the TODO), but nvcc already reuses the four values: the course build
      // loads 17 bytes per surviving thread (centre + 16), not 21.
      // Do a coarse corner check
      // TODO: I could use the results of the prechecks afterwards
      if(fast_gpu_prechecks(c_t,ct,d_image_ptr,image_pitch,image_pitch2,image_pitch3)) {
        return;
      }

      // [SLAM zero-to-hero] px[16] lives in registers only because the loop is fully unrolled and
      // every index is a compile-time constant. A per-thread array indexed at run time goes to
      // local memory, which is off-chip (cached) and much slower. Unrolling also removes the
      // `if(i<11)` and `if(i==11)` tests, and the `?:` below compile to branch-free shifts and
      // masks, not branches.
      float px[16];
      #pragma unroll 16
      for(int i=0;i<16;++i) {
        int image_ptr_offset = bresenham_circle_offset_pitch(i,image_pitch,image_pitch2,image_pitch3);
        px[i] = (float)d_image_ptr[image_ptr_offset];
        // [SLAM zero-to-hero] signbit(px - c_t) is 1 exactly when px < c - t: a subtract and a
        // sign-bit read, no compare-and-branch. Pixels are whole numbers in float, so with a
        // whole-number threshold (10 in fast_benchmark) this matches the integer compares of
        // Rosten's CPU code exactly.
        int darker = signbit(px[i]-c_t);
        int brighter = signbit(ct-px[i]);
#if FAST_GPU_USE_LOOKUP_TABLE && FAST_GPU_USE_LOOKUP_TABLE_BITBASED
        if(i<11) {
          dark_diff_address   += darker?(1<<i):0;
          bright_diff_address += brighter?(1<<i):0;
        } else {
          // [SLAM zero-to-hero] An early load, a form of software prefetch: points 0-10 fix the low
          // 11 bits, which select the table word, so the source requests both words here, before
          // points 11-15 are compared. A GPU thread does not stall when it issues a load, only when
          // an instruction first uses the loaded register. nvcc may move loads: in the course build
          // it keeps one word's load here and moves the other below the `||` after the loop, so
          // that load is issued only when the first test fails.
          if(i==11) {
            // initiate a readout
            dark_diff_data = d_corner_lut[dark_diff_address];
            bright_diff_data = d_corner_lut[bright_diff_address];
          }
          dark_diff_bit   += darker?(1<<(i-11)):0;
          bright_diff_bit += brighter?(1<<(i-11)):0;
        }
#else
        dark_diff_address   += signbit(px[i]-c_t)?(1<<i):0;
        bright_diff_address += signbit(ct-px[i])?(1<<i):0;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
      }
      // [SLAM zero-to-hero] The table read replaces the arc loop: the same work for every lane, so
      // no control divergence. The price is a gather: the 32 lanes read up to 32 different table
      // words, which can take several memory transactions (upstream: "serialized accesses to global
      // memory", fast_gpu_config.h). The table is small (8 KB) and read every frame, so it likely
      // stays cached. It is read through const __restrict__, which lets nvcc use the read-only data
      // path, as __ldg() would.
      // Look up these addresses, whether they qualify for a corner
      // If any of these qualify for a corner, it is a corner candidate, yaay
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
      if((dark_diff_data&(1<<dark_diff_bit)) || (bright_diff_data&(1<<bright_diff_bit))) {
#else
      if(d_corner_lut[dark_diff_address] || d_corner_lut[bright_diff_address]) {
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#else
      if(fast_gpu_is_corner(dark_diff_address,min_arc_length) ||
         fast_gpu_is_corner(bright_diff_address,min_arc_length)) {
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        // [SLAM zero-to-hero] Only corners, a small fraction of the pixels, get here. In a warp
        // with one corner lane, the other 31 lanes wait (masked off) while it computes the score.
        /*
         * Note to future self:
         * Only calculate the score once we determined that the pixel is considered
         * a corner. This policy gave better results than computing the score
         * for every pixel
         */
        if(SCORE == SUM_OF_ABS_DIFF_ALL) {
          float response = 0.0f;
          #pragma unroll 16
          for(int i=0;i<16;++i) {
            response += fabsf(px[i]-c);
          }
          d_response[resp_offset] = response;
        } else if(SCORE == SUM_OF_ABS_DIFF_ON_ARC) {
          // [SLAM zero-to-hero] The score fast_benchmark uses, the same as Rosten's
          // fast9_corner_score_old (vilib's CPU reference for this score): the larger of the sums
          // of |px - c| - t over the brighter and over the darker circle points. Every brighter
          // (darker) point counts, not only those of the arc. With a whole-number threshold every
          // term is a whole number and the sums stay far below 2^24, so the float adds are exact
          // and the GPU score equals the CPU score: ties are ties on both sides. The masks are
          // tested with ?: (selects), no branches.
          float response_bright = 0.0f;
          float response_dark   = 0.0f;
          #pragma unroll 16
          for(int i=0;i<16;++i) {
            float absdiff = fabsf(px[i]-c)-threshold;
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
            if(i<11) {
              // address
              response_dark += (dark_diff_address&(1<<i))?absdiff:0.0f;
              response_bright += (bright_diff_address&(1<<i))?absdiff:0.0f;
            } else {
              // bits
              response_dark += (dark_diff_bit&(1<<(i-11)))?absdiff:0.0f;
              response_bright += (bright_diff_bit&(1<<(i-11)))?absdiff:0.0f;
            }
#else
            response_dark   += (dark_diff_address&(1<<i))?absdiff:0.0f;
            response_bright += (bright_diff_address&(1<<i))?absdiff:0.0f;
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
          }
          d_response[resp_offset] = fmaxf(response_bright,response_dark);
        } else if(SCORE == MAX_THRESHOLD) {
          // [SLAM zero-to-hero] Rosten's 2008 score, as in fast9_corner_score_new: the largest
          // threshold at which the pixel is still a corner. 7-8 rounds over [t + 1, 255], each
          // rerunning the 16 compares from registers plus two table reads. The trip count is almost
          // the same for every lane, so the warp diverges little.
          // Binary search for the maximum threshold value with which the given
          // point is still a corner
          float min_thr = threshold + 1;
          float max_thr = 255.0f;
          while(min_thr <= max_thr) {
            float med_thr = floorf((min_thr + max_thr)*0.5f);
            // try out med_thr as a new threshold
            if(fast_gpu_is_corner_quick(d_corner_lut,
                                        px,
                                        c,
                                        med_thr,
                                        dark_diff_address,
                                        bright_diff_address
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                        ,
                                        dark_diff_bit,
                                        dark_diff_data,
                                        bright_diff_bit,
                                        bright_diff_data
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
                                      )) {
              // still a corner
              min_thr = med_thr + 1.0f;
            } else {
              // not a corner anymore
              max_thr = med_thr - 1.0f;
            }
          }
          d_response[resp_offset] = max_thr;
        }
      }
    }
  }
}

__host__ void fast_gpu_calc_corner_response(const int image_width,
                                            const int image_height,
                                            const int image_pitch,
                                            const unsigned char * d_image,
                                            const int horizontal_border,
                                            const int vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
                                            const unsigned int * d_corner_lut,
#else
                                            const unsigned char * d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                            const float threshold,
                                            const int min_arc_length,
                                            const fast_score score,
                                            const int response_pitch_elements,
                                            float * d_response,
                                            cudaStream_t stream) {
  // [SLAM zero-to-hero] 128 threads (4 warps) per block: 32x4 when the width is a multiple of 32,
  // otherwise 16x8. Likely why: fewer idle threads at the right edge. 752 = 47 x 16, while 32-wide
  // blocks would leave half of the last block column idle. The grid covers the whole image, borders
  // included, because every pixel stores its response.
  // Note: I'd like to launch 128 threads / thread block
  std::size_t threads_per_x = (image_width%32 == 0)?32:16;
  std::size_t threads_per_y = 128/threads_per_x;
  kernel_params_t p = cuda_gen_kernel_params_2d(image_width,
                                                image_height,
                                                threads_per_x,
                                                threads_per_y);
  // [SLAM zero-to-hero] The run-time score picks a compile-time instantiation. stream is
  // DetectorBaseGPU::stream_, 0 unless setStream() was called. This file is compiled with
  // --default-stream per-thread, so stream 0 here is the calling host thread's default stream, not
  // the legacy default stream that the .cpp files use (README, "Two default streams").
  switch(score) {
    case SUM_OF_ABS_DIFF_ALL:
      fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ALL> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
          image_width,
          image_height,
          image_pitch,
          d_image,
          horizontal_border,
          vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
          d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
          threshold,
          min_arc_length,
          response_pitch_elements,
          d_response
        );
      break;
    case SUM_OF_ABS_DIFF_ON_ARC:
      fast_gpu_calc_corner_response_kernel<SUM_OF_ABS_DIFF_ON_ARC> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
        image_width,
        image_height,
        image_pitch,
        d_image,
        horizontal_border,
        vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
        d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        threshold,
        min_arc_length,
        response_pitch_elements,
        d_response
      );
      break;
    case MAX_THRESHOLD:
      fast_gpu_calc_corner_response_kernel<MAX_THRESHOLD> <<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
        image_width,
        image_height,
        image_pitch,
        d_image,
        horizontal_border,
        vertical_border,
#if FAST_GPU_USE_LOOKUP_TABLE
        d_corner_lut,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
        threshold,
        min_arc_length,
        response_pitch_elements,
        d_response
      );
      break;
  }
  CUDA_KERNEL_CHECK();
}

} // namespace vilib
