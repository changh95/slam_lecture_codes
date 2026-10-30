/*
 * Detector Base GPU CUDA tools
 * detector_base_gpu_cuda_tools.cu
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
// Grid non-maximum suppression (NMS). Input: the float response image of one pyramid level
// (fast_gpu_cuda_tools.cu). Output: for each grid cell, the strongest response that is strictly
// greater than its 8 neighbours, as a position, a score and a level. Called once per level; the
// per-cell arrays keep the best over all levels. A VIO front end wants its features spread over the
// image, and one corner per cell gives that, with a result of fixed size.
//
// Mapping: one thread block per cell. Grid = the cell grid (24 x 15 = 360 blocks for 752x480 and
// 32-px cells). Block = the cell width at this level times as many rows as fit in 128 threads: 32x4
// at level 0. Warp w is then thread row w, and each thread walks down one column of the cell, 4
// rows per step (8 pixels at level 0).
// Memory:
//  - Response image: global, pitched, read only. A warp reads 32 consecutive floats (128 bytes) per
//    access, coalesced. Each value is read by up to 9 threads; the repeats come from cache.
//  - Registers: the suppressed value and each thread's best (score, x, y). The response image is
//    never written, unlike in the unused regular NMS below.
//  - Shared memory: 4 floats per warp (dynamic, size set at launch), to combine the warps.
//  - Global per-cell output (d_pos, d_score, d_level): written by one thread per block.
// Techniques:
//  1. A fully unrolled 8-neighbour loop; nms_offset() reduces to one pitch expression per
//     neighbour, as the FAST circle does.
//  2. Branch-free suppression: copysignf turns each comparison into a factor of 0 or 1.
//  3. Warp-level argmax with __shfl_down_sync: 5 steps, registers only, no __syncthreads().
//  4. Block-level step in shared memory: lane 0 of each warp writes its result, __syncthreads(),
//     then thread 0 combines the warps.
//  5. strictly_greater is a template parameter; the host picks the instantiation.
//  6. One result slot per cell. The output size is known before the kernel runs, so there is no
//     atomic counter, no stream compaction and no count to read back first; each slot has exactly
//     one writer, so no atomics at all. The host copies 16 bytes per cell (5,760 bytes for 360
//     cells) instead of the 1.4 MB response image (detector_base_gpu.cpp).
// Size: 360 blocks of 4 warps are about 2 blocks per SM on the RTX 5090's 170 SMs (README): far
// fewer warps than an SM can keep in flight to hide memory latency. Upstream timed this kernel on a
// GTX 960M and a Jetson TX2 (see the note in the kernel); the TX2 has 2 SMs, so there the same grid
// fills the GPU many times over. README, "What the numbers say", point 4, shows the same effect for
// the pyramid kernels: for work this small, launches and synchronizations cost more than the kernel
// code.
// Error checks: CUDA_KERNEL_CHECK() is empty (-DNDEBUG), as in fast_gpu_cuda_tools.cu.
#include "vilib/feature_detection/detector_base_gpu_cuda_tools.h"
#include "vilib/feature_detection/config.h"
#include "vilib/cuda_common.h"

namespace vilib {

// [SLAM zero-to-hero] All 32 lanes take part in every shuffle below. That needs full warps, and
// every lane must reach the shuffle; see the reduction in the grid kernel.
  // Warp preliminaries
#define WARP_MASK                            0xFFFFFFFF
// Use precalculated indices
/*
 * Note to future self: this is just a coding convenience,
 * because the loop is unrolled, there's no speed impact
 */
#define USE_PRECALCULATED_INDICES            1

#if USE_PRECALCULATED_INDICES
// [SLAM zero-to-hero] Not called anywhere in vilib; nms_offset() below is the one in use.
__inline__ __device__ void nms_offset_pos(const int & i, float & x_offset, float & y_offset) {
  /*
   * 3x3 (n=1):
   * 7 0 1
   * 6 x 2
   * 5 4 3
   */
   if(i==0) {
     x_offset = 0.0f;
     y_offset = -1.0f;
   }
   if(i==1) {
     x_offset = 1.0f;
     y_offset = -1.0f;
   }
   if(i==2) {
     x_offset = 1.0f;
     y_offset = 0.0f;
   }
   if(i==3) {
     x_offset = 1.0f;
     y_offset = 1.0f;
   }
   if(i==4) {
     x_offset = 0.0f;
     y_offset = 1.0f;
   }
   if(i==5) {
     x_offset = -1.0f;
     y_offset = 1.0f;
   }
   if(i==6) {
     x_offset = -1.0f;
     y_offset = 0.0f;
   }
   if(i==7) {
     x_offset = -1.0f;
     y_offset = -1.0f;
   }
 #if (DETECTOR_BASE_NMS_SIZE >= 5)
  /*
   * 5x5 (n=2):
   * 23 8  9  10 11
   * 22 7  0  1  12
   * 21 6  x  2  13
   * 20 5  4  3  14
   * 19 18 17 16 15
   */
   if(i==8) {
     x_offset = -1.0f;
     y_offset = -2.0f;
   }
   if(i==9) {
     x_offset = 0.0f;
     y_offset = -2.0f;
   }
   if(i==10) {
     x_offset = 1.0f;
     y_offset = -2.0f;
   }
   if(i==11) {
     x_offset = 2.0f;
     y_offset = -2.0f;
   }
   if(i==12) {
     x_offset = 2.0f;
     y_offset = -1.0f;
   }
   if(i==13) {
     x_offset = 2.0f;
     y_offset = 0.0f;
   }
   if(i==14) {
     x_offset = 2.0f;
     y_offset = 1.0f;
   }
   if(i==15) {
     x_offset = 2.0f;
     y_offset = 2.0f;
   }
   if(i==16) {
     x_offset = 1.0f;
     y_offset = 2.0f;
   }
   if(i==17) {
     x_offset = 0.0f;
     y_offset = 2.0f;
   }
   if(i==18) {
     x_offset = -1.0f;
     y_offset = 2.0f;
   }
   if(i==19) {
     x_offset = -2.0f;
     y_offset = 2.0f;
   }
   if(i==20) {
     x_offset = -2.0f;
     y_offset = 1.0f;
   }
   if(i==21) {
     x_offset = -2.0f;
     y_offset = 0.0f;
   }
   if(i==22) {
     x_offset = -2.0f;
     y_offset = -1.0f;
   }
   if(i==23) {
     x_offset = -2.0f;
     y_offset = -2.0f;
   }
#if (DETECTOR_BASE_NMS_SIZE >= 7)
  /*
   * 7x7 (n=3)
   * 47 24 25 26 27 28 29
   * 46 23 8  9  10 11 30
   * 45 22 7  0  1  12 31
   * 44 21 6  x  2  13 32
   * 43 20 5  4  3  14 33
   * 42 19 18 17 16 15 34
   * 41 40 39 38 37 36 35
   */
   if(i==24) {
     x_offset = -2.0f;
     y_offset = -3.0f;
   }
   if(i==25) {
     x_offset = -1.0f;
     y_offset = -3.0f;
   }
   if(i==26) {
    x_offset = 0.0f;
    y_offset = -3.0f;
  }
  if(i==27) {
    x_offset = 1.0f;
    y_offset = -3.0f;
  }
  if(i==28) {
    x_offset = 2.0f;
    y_offset = -3.0f;
  }
  if(i==29) {
    x_offset = 3.0f;
    y_offset = -3.0f;
  }
  if(i==30) {
    x_offset = 3.0f;
    y_offset = -2.0f;
  }
  if(i==31) {
    x_offset = 3.0f;
    y_offset = -1.0f;
  }
  if(i==32) {
    x_offset = 3.0f;
    y_offset = 0.0f;
  }
  if(i==33) {
    x_offset = 3.0f;
    y_offset = 1.0f;
  }
  if(i==34) {
    x_offset = 3.0f;
    y_offset = 2.0f;
  }
  if(i==35) {
    x_offset = 3.0f;
    y_offset = 3.0f;
  }
  if(i==36) {
    x_offset = 2.0f;
    y_offset = 3.0f;
  }
  if(i==37) {
    x_offset = 1.0f;
    y_offset = 3.0f;
  }
  if(i==38) {
    x_offset = 0.0f;
    y_offset = 3.0f;
  }
  if(i==39) {
    x_offset = -1.0f;
    y_offset = 3.0f;
  }
  if(i==40) {
    x_offset = -2.0f;
    y_offset = 3.0f;
  }
  if(i==41) {
    x_offset = -3.0f;
    y_offset = 3.0f;
  }
  if(i==42) {
    x_offset = -3.0f;
    y_offset = 2.0f;
  }
  if(i==43) {
    x_offset = -3.0f;
    y_offset = 1.0f;
  }
  if(i==44) {
    x_offset = -3.0f;
    y_offset = 0.0f;
  }
  if(i==45) {
    x_offset = -3.0f;
    y_offset = -1.0f;
  }
  if(i==46) {
    x_offset = -3.0f;
    y_offset = -2.0f;
  }
  if(i==47) {
    x_offset = -3.0f;
    y_offset = -3.0f;
  }
#if (DETECTOR_BASE_NMS_SIZE >= 9)
  /* 
   * 9x9 (n=4)
   * 79 48 49 50 51 52 53 54 55
   * 78 47 24 25 26 27 28 29 56 
   * 77 46 23 8  9  10 11 30 57
   * 76 45 22 7  0  1  12 31 58
   * 75 44 21 6  x  2  13 32 59
   * 74 43 20 5  4  3  14 33 60
   * 73 42 19 18 17 16 15 34 61
   * 72 41 40 39 38 37 36 35 62
   * 71 70 69 68 67 66 65 64 63
   */
  if(i==48) {
    x_offset = -3.0f;
    y_offset = -4.0f;
  }
  if(i==49) {
    x_offset = -2.0f;
    y_offset = -4.0f;
  }
  if(i==50) {
    x_offset = -1.0f;
    y_offset = -4.0f;
  }
  if(i==51) {
    x_offset = 0.0f;
    y_offset = -4.0f;
  }
  if(i==52) {
    x_offset = 1.0f;
    y_offset = -4.0f;
  }
  if(i==53) {
    x_offset = 2.0f;
    y_offset = -4.0f;
  }
  if(i==54) {
    x_offset = 3.0f;
    y_offset = -4.0f;
  }
  if(i==55) {
    x_offset = 4.0f;
    y_offset = -4.0f;
  }
  if(i==56) {
    x_offset = 4.0f;
    y_offset = -3.0f;
  }
  if(i==57) {
    x_offset = 4.0f;
    y_offset = -2.0f;
  }
  if(i==58) {
    x_offset = 4.0f;
    y_offset = -1.0f;
  }
  if(i==59) {
    x_offset = 4.0f;
    y_offset = 0.0f;
  }
  if(i==60) {
    x_offset = 4.0f;
    y_offset = 1.0f;
  }
  if(i==61) {
    x_offset = 4.0f;
    y_offset = 2.0f;
  }
  if(i==62) {
    x_offset = 4.0f;
    y_offset = 3.0f;
  }
  if(i==63) {
    x_offset = 4.0f;
    y_offset = 4.0f;
  }
  if(i==64) {
    x_offset = 3.0f;
    y_offset = 4.0f;
  }
  if(i==65) {
    x_offset = 2.0f;
    y_offset = 4.0f;
  }
  if(i==66) {
    x_offset = 1.0f;
    y_offset = 4.0f;
  }
  if(i==67) {
    x_offset = 0.0f;
    y_offset = 4.0f;
  }
  if(i==68) {
    x_offset = -1.0f;
    y_offset = 4.0f;
  }
  if(i==69) {
    x_offset = -2.0f;
    y_offset = 4.0f;
  }
  if(i==70) {
    x_offset = -3.0f;
    y_offset = 4.0f;
  }
  if(i==71) {
    x_offset = -4.0f;
    y_offset = 4.0f;
  }
  if(i==72) {
    x_offset = -4.0f;
    y_offset = 3.0f;
  }
  if(i==73) {
    x_offset = -4.0f;
    y_offset = 2.0f;
  }
  if(i==74) {
    x_offset = -4.0f;
    y_offset = 1.0f;
  }
  if(i==75) {
    x_offset = -4.0f;
    y_offset = 0.0f;
  }
  if(i==76) {
    x_offset = -4.0f;
    y_offset = -1.0f;
  }
  if(i==77) {
    x_offset = -4.0f;
    y_offset = -2.0f;
  }
  if(i==78) {
    x_offset = -4.0f;
    y_offset = -3.0f;
  }
  if(i==79) {
    x_offset = -4.0f;
    y_offset = -4.0f;
  }
#endif /* (DETECTOR_BASE_NMS_SIZE >= 9) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 7) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 5) */
}

// [SLAM zero-to-hero] Offset of neighbour i, in floats. Only called with the index of a fully
// unrolled loop, so the if-chain reduces to one pitch expression per neighbour, as for the FAST
// circle. With
// DETECTOR_BASE_NMS_SIZE 3 (config.h) only i = 0..7 occur: the 8 neighbours, clockwise from the one
// above.
__inline__ __device__ int nms_offset(const int & i, const int & pitch) {
  int offs=0;
  /*
   * 3x3 (n=1):
   * 7 0 1
   * 6 x 2
   * 5 4 3
   */
  if(i==0)
    offs=-pitch;
  if(i==1)
    offs=-pitch+1;
  if(i==2)
    offs=+1;
  if(i==3)
    offs=pitch+1;
  if(i==4)
    offs=pitch;
  if(i==5)
    offs=pitch-1;
  if(i==6)
    offs= -1;
  if(i==7)
    offs=-pitch-1;
#if (DETECTOR_BASE_NMS_SIZE >= 5)
  /*
   * 5x5 (n=2):
   * 23 8  9  10 11
   * 22 7  0  1  12
   * 21 6  x  2  13
   * 20 5  4  3  14
   * 19 18 17 16 15
   */
  if(i==8)
    offs=-(pitch<<1)-1;
  if(i==9)
    offs=-(pitch<<1);
  if(i==10)
    offs=-(pitch<<1)+1;
  if(i==11)
    offs=-(pitch<<1)+2;
  if(i==12)
    offs=-pitch+2;
  if(i==13)
    offs=+2;
  if(i==14)
    offs=pitch+2;
  if(i==15)
    offs=(pitch<<1)+2;
  if(i==16)
    offs=(pitch<<1)+1;
  if(i==17)
    offs=(pitch<<1);
  if(i==18)
    offs=(pitch<<1)-1;
  if(i==19)
    offs=(pitch<<1)-2;
  if(i==20)
    offs=pitch-2;
  if(i==21)
    offs=-2;
  if(i==22)
    offs=-pitch-2;
  if(i==23)
    offs=-(pitch<<1)-2;
#if (DETECTOR_BASE_NMS_SIZE >= 7)
  /*
   * 7x7 (n=3)
   * 47 24 25 26 27 28 29
   * 46 23 8  9  10 11 30
   * 45 22 7  0  1  12 31
   * 44 21 6  x  2  13 32
   * 43 20 5  4  3  14 33
   * 42 19 18 17 16 15 34
   * 41 40 39 38 37 36 35
   */
  if(i==24)
    offs=-((pitch<<1)+pitch)-2;
  if(i==25)
    offs=-((pitch<<1)+pitch)-1;
  if(i==26)
    offs=-((pitch<<1)+pitch);
  if(i==27)
    offs=-((pitch<<1)+pitch)+1;
  if(i==28)
    offs=-((pitch<<1)+pitch)+2;
  if(i==29)
    offs=-((pitch<<1)+pitch)+3;
  if(i==30)
    offs=-(pitch<<1)+3;
  if(i==31)
    offs=-pitch+3;
  if(i==32)
    offs=3;
  if(i==33)
    offs=pitch+3;
  if(i==34)
    offs=(pitch<<1)+3;
  if(i==35)
    offs=((pitch<<1)+pitch)+3;
  if(i==36)
    offs=((pitch<<1)+pitch)+2;
  if(i==37)
    offs=((pitch<<1)+pitch)+1;
  if(i==38)
    offs=((pitch<<1)+pitch);
  if(i==39)
    offs=((pitch<<1)+pitch)-1;
  if(i==40)
    offs=((pitch<<1)+pitch)-2;
  if(i==41)
    offs=((pitch<<1)+pitch)-3;
  if(i==42)
    offs=(pitch<<1)-3;
  if(i==43)
    offs=pitch-3;
  if(i==44)
    offs=-3;
  if(i==45)
    offs=-pitch-3;
  if(i==46)
    offs=-(pitch<<1)-3;
  if(i==47)
    offs=-((pitch<<1)+pitch)-3;
#if (DETECTOR_BASE_NMS_SIZE >= 9)
  /* 
   * 9x9 (n=4)
   * 79 48 49 50 51 52 53 54 55
   * 78 47 24 25 26 27 28 29 56 
   * 77 46 23 8  9  10 11 30 57
   * 76 45 22 7  0  1  12 31 58
   * 75 44 21 6  x  2  13 32 59
   * 74 43 20 5  4  3  14 33 60
   * 73 42 19 18 17 16 15 34 61
   * 72 41 40 39 38 37 36 35 62
   * 71 70 69 68 67 66 65 64 63
   */
  if(i==48)
    offs=-(pitch<<2)-3;
  if(i==49)
    offs=-(pitch<<2)-2;
  if(i==50)
    offs=-(pitch<<2)-1;
  if(i==51)
    offs=-(pitch<<2);
  if(i==52)
    offs=-(pitch<<2)+1;
  if(i==53)
    offs=-(pitch<<2)+2;
  if(i==54)
    offs=-(pitch<<2)+3;
  if(i==55)
    offs=-(pitch<<2)+4;
  if(i==56)
    offs=-((pitch<<1)+pitch)+4;
  if(i==57)
    offs=-(pitch<<1)+4;
  if(i==58)
    offs=-pitch+4;
  if(i==59)
    offs=4;
  if(i==60)
    offs=pitch+4;
  if(i==61)
    offs=(pitch<<1)+4;
  if(i==62)
    offs=((pitch<<1)+pitch)+4;
  if(i==63)
    offs=(pitch<<2)+4;
  if(i==64)
    offs=(pitch<<2)+3;
  if(i==65)
    offs=(pitch<<2)+2;
  if(i==66)
    offs=(pitch<<2)+1;
  if(i==67)
    offs=(pitch<<2);
  if(i==68)
    offs=(pitch<<2)-1;
  if(i==69)
    offs=(pitch<<2)-2;
  if(i==70)
    offs=(pitch<<2)-3;
  if(i==71)
    offs=(pitch<<2)-4;
  if(i==72)
    offs=((pitch<<1)+pitch)-4;
  if(i==73)
    offs=(pitch<<1)-4;
  if(i==74)
    offs=pitch-4;
  if(i==75)
    offs=-4;
  if(i==76)
    offs=-pitch-4;
  if(i==77)
    offs=-(pitch<<1)-4;
  if(i==78)
    offs=-((pitch<<1)+pitch)-4;
  if(i==79)
    offs=-(pitch<<2)-4;
#endif /* (DETECTOR_BASE_NMS_SIZE >= 9) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 7) */
#endif /* (DETECTOR_BASE_NMS_SIZE >= 5) */
  return offs;
}
#endif /* USE_PRECALCULATED_INDICES */

// [SLAM zero-to-hero] Not used: processResponse() calls the grid version (`#if 1`), and this one
// was kept "for comparison purposes in the paper". It shows what the grid kernel avoids. It writes
// 0 into d_response while neighbouring threads may still read that pixel, so a neighbour sees the
// old score or 0 depending on timing: a data race in global memory. And its `break` lets the lanes
// of a warp leave the loop at different times. The grid kernel keeps its results in registers and
// suppresses without branches.
__global__ void detector_base_gpu_regular_nms_kernel(const int image_width_m_borders,
                                                     const int image_height_m_borders,
                                                     const int horizontal_border,
                                                     const int vertical_border,
                                                     const int response_pitch_elements,
                                                     float * __restrict__ d_response) {
  const int x = blockDim.x*blockIdx.x + threadIdx.x; // thread id X
  const int y = blockDim.y*blockIdx.y + threadIdx.y; // thread id Y
  if(x < image_width_m_borders && y < image_height_m_borders) {
    float * d_response_ptr = d_response + (y+vertical_border)*response_pitch_elements + x + horizontal_border;
    const float center_value = d_response_ptr[0];

#if USE_PRECALCULATED_INDICES == 0
    int x=0;
    int y=-1;
    int dx=1;
    int dy=0;
    bool next=false;
#endif /* USE_PRECALCULATED_INDICES */

    // Spiral generation
    #pragma unroll
    for(int i=0;i<(DETECTOR_BASE_NMS_SIZE*DETECTOR_BASE_NMS_SIZE-1);++i) {
#if USE_PRECALCULATED_INDICES
      const int j = nms_offset(i,response_pitch_elements);
#else
      const int j = y*response_pitch_elements + x;
#endif /* USE_PRECALCULATED_INDICES */

      if(d_response_ptr[j] > center_value) {
        d_response_ptr[0] = 0.0f;
        break;
      }

#if USE_PRECALCULATED_INDICES == 0
      if((x>0 && x==y) || (x==-y) || next) {
        int tmp = dx;
        dx = -dy;
        dy = tmp;
        next = false;
      } else if(x<0 && x==y) {
        next = true;
      }
      x=x+dx;
      y=y+dy;
#endif /* USE_PRECALCULATED_INDICES */
    }
  }
}

__host__ void detector_base_gpu_regular_nms(const int image_width,
                                            const int image_height,
                                            const int horizontal_border,
                                            const int vertical_border,
                                            const int response_pitch_elements,
                                            float * d_response,
                                            cudaStream_t stream) {
  const int image_width_m_borders = image_width-2*horizontal_border;
  const int image_height_m_borders = image_height-2*vertical_border;
  // Note: I'd like to have 128 threads / thread block
  const int horizontal_threads = ((image_width_m_borders%32) == 0)?32:16;
  const int vertical_threads = 128 / horizontal_threads;
  kernel_params_t p_nms = cuda_gen_kernel_params_2d(image_width_m_borders,
                                                    image_height_m_borders,
                                                    horizontal_threads,
                                                    vertical_threads);
  // [SLAM zero-to-hero] Upstream bug in this unused path: the kernel's second argument is its
  // height bound, but image_width_m_borders is passed twice.
  detector_base_gpu_regular_nms_kernel<<<p_nms.blocks_per_grid,p_nms.threads_per_block,0,stream>>>(
                                     image_width_m_borders,
                                     image_width_m_borders,
                                     horizontal_border,
                                     vertical_border,
                                     response_pitch_elements,
                                     d_response);
  CUDA_KERNEL_CHECK();
}

// [SLAM zero-to-hero] strictly_greater is a template parameter, so `if(strictly_greater)` in the
// loop is resolved at compile time and each instantiation holds one comparison formula.
template<bool strictly_greater>
__global__ void detector_base_gpu_grid_nms_kernel(const int level,
                                                  const int min_level,
                                                  const int image_width,
                                                  const int image_height,
                                                  const int horizontal_border,
                                                  const int vertical_border,
                                                  const int cell_size_width,
                                                  const int cell_size_height,
                                                  const int response_pitch_elements,
                                                  const float * __restrict__ d_response,
                                                  float2 * __restrict__ d_pos,
                                                  float * __restrict__ d_score,
                                                  int * __restrict__ d_level) {
  // [SLAM zero-to-hero] Block (bx, by) is cell (bx, by). x is the thread's column in the image; a
  // thread only ever scans that column, so max_x below is simply x. lane_id and warp_id come from
  // the linear thread index; with blockDim.x = 32 (level 0), warp w is thread row w.
  // Various identifiers
  const int x = cell_size_width * blockIdx.x + threadIdx.x;
  const int y = cell_size_height * blockIdx.y + threadIdx.y;
  const int cell_id = gridDim.x * blockIdx.y + blockIdx.x;
  const int thread_id = threadIdx.x + blockDim.x * threadIdx.y;
  const int lane_id = thread_id & 0x1F;
  const int warp_id = thread_id >> 5;
  const int warp_cnt = (blockDim.x * blockDim.y + 31) >> 5;
  // Selected maximum response
  float max_x = static_cast<float>(x);
  float max_y = 0.f;
  float max_resp = 0.0f;

  // [SLAM zero-to-hero] Threads outside the image (for a width of 752, the right half of the last
  // cell column) skip the scan but do not return: they must still reach the shuffles and the
  // __syncthreads() below, and they take part with max_resp = 0.
  if(x < image_width && y < image_height) {
    // [SLAM zero-to-hero] Clears this cell's score once per frame, on the first level detected;
    // later levels compare against it. The same thread reads it back at the end, so no barrier is
    // needed.
    if(threadIdx.x == 0 && threadIdx.y == 0) {
      // the very first thread in the threadblock, initializes the cell score
      if(level == min_level) {
        d_score[cell_id] = 0.0f;
      }
    }

    /*
     * Note to future self:
     * basically, we perform NMS on every line in the cell just like a regular NMS, BUT
     * we go in a spiral and check if any of the neigbouring values is higher than our current one.
     * If it is higher, than we set our current value to zero.
     * We DO NOT write to the response buffer, we keep everything in registers. Also,
     * we do not use if-else for checking values, we use my signbit trick. This latter is amazing, because
     * it completely avoids warp divergence.
     * Then once all threads in a warp figured out whether the value they are looking at was supressed
     * or not, they reduce with warp-level intrinsics to lane 0.
     */

    // Maximum value
    // Location: x,y -> x is always threadIdx.x
    int max_y_tmp = 0;

    // [SLAM zero-to-hero] Threads in the left and right border columns idle. For the top border the
    // whole block starts lower instead (y_offset): in the top cell row, with a border of 8 and 4
    // thread rows, the threads cover rows 8..31 in 6 steps, with no step spent on border rows. The
    // bottom border ends the loop (gy < image_height_m_border).
    // border remains the same irrespective of pyramid level
    int image_width_m_border = image_width-horizontal_border;
    int image_height_m_border = image_height-vertical_border;
    if(x >= horizontal_border && x < image_width_m_border) {
      // we want as few idle threads as possible, hence we shift them according to y
      // note: we shift down all lines within the block
      int cell_top_to_border = vertical_border-(cell_size_height*blockIdx.y);
      int y_offset = cell_top_to_border>0?cell_top_to_border:0;
      int gy = y + y_offset;
      int box_line = threadIdx.y + y_offset;
      const float * d_response_ptr = d_response + gy*response_pitch_elements + x;
      int response_offset = blockDim.y*response_pitch_elements;
      for(; (box_line < cell_size_height) && (gy < image_height_m_border); box_line += blockDim.y
        , d_response_ptr += response_offset
        , gy += blockDim.y) {
        // acquire the center response value
        float center_value = d_response_ptr[0];

        // Perform spiral NMS
#if (USE_PRECALCULATED_INDICES == 0)
        int x=0;
        int y=-1;
        int dx=1;
        int dy=0;
        bool next=false;
#endif /* USE_PRECALCULATED_INDICES */

        #pragma unroll
        for(int i=0;i<(DETECTOR_BASE_NMS_SIZE*DETECTOR_BASE_NMS_SIZE-1);++i) {
#if USE_PRECALCULATED_INDICES
          int j = nms_offset(i,response_pitch_elements);
#else
          int j = y*response_pitch_elements+x;
#endif /* USE_PRECALCULATED_INDICES */

          // [SLAM zero-to-hero] Branch-free suppression; FASTGPU passes strictly_greater = true.
          // copysignf(1, n - c) is +1 when neighbour n >= c (n - c is +0.0 on a tie) and -1 when
          // n < c, so -0.5*(-1 + s) is 0 or 1, and the multiply zeroes c if any neighbour is at
          // least as large. Once 0, it stays 0. Ties suppress both pixels, as Rosten's
          // nonmax_suppression does (Compare is >=, rosten/nonmax.cpp), the CPU reference in
          // fast_benchmark. Every lane runs the same instructions for its 8 neighbours whatever the
          // data: no divergence.
          // Perform non-maximum suppression
          if(strictly_greater) {
            center_value *= -0.5f*(-1.0f+copysignf(1.0f,d_response_ptr[j]-center_value));
          } else {
            center_value *= 0.5f*(1.0f+copysignf(1.0f,center_value-d_response_ptr[j]));
          }
          /*
           * Note to future self:
           * Interestingly on Maxwell (960M), checking for equivalence with 0.0f, results
           * in better runtimes.
           * However, on Pascal (Tegra X2) this increases the runtime, hence we opted for
           * not checking it for now.
           */
#if 0
          if(center_value==0.0f) break; // we should check it on the Jetson
#endif /* 0 */

#if (USE_PRECALCULATED_INDICES == 0)
          if((x>0 && x==y) || (x==-y) || next) {
            // if we reached the limit, change how we proceed
            // essentially we change the direction
            int tmp = dx;
            dx = -dy;
            dy = tmp;
            next = false;
          } else if(x<0 && x==y) {
            next = true;
          }
          x = x+dx;
          y = y+dy;
#endif /* USE_PRECALCULATED_INDICES */
        }

        // [SLAM zero-to-hero] Running best of this thread's column, in registers. `>` keeps the
        // topmost of equal scores. Two register moves, which the compiler can predicate instead of
        // branching.
        // NMS is over, is this value greater than the previous one?
        if(center_value > max_resp) {
           max_resp = center_value;
           max_y_tmp = gy;
        }
      }
    }
    // Perform conversion
    max_y = static_cast<float>(max_y_tmp);
  }

  // [SLAM zero-to-hero] Warp-level argmax. At each step lane i takes lane i + offset's
  // (score, x, y) if that score is strictly greater; after offsets 16, 8, 4, 2, 1, lane 0 holds the
  // warp's best. 15 shuffles, all in registers: no shared memory and no barrier inside a warp.
  // Lanes whose source would be past lane 31 get their own value back; only lane 0's result is
  // used. Ties do not go to the smallest x: at each step the lower lane of a pair keeps its value,
  // so the last step (offset 1) prefers an even lane, the step before it a lane with bit 1 clear,
  // and so on. The winner is the tied lane with the smallest bit-reversed index (lanes 3 and 5
  // tied: lane 5 wins). The shared-memory step below prefers the lower warp. So a cell's
  // equal-score corners go to the lowest warp (thread row), then that bit-reversed lane order,
  // then the topmost row: not the raster order (topmost row, then smallest x)
  // in which Rosten's CPU path keeps the first of equal corners. Hence the README's 224 cells where
  // CPU and GPU pick different corners with the same score.
  // Full warps only: with 32-px cells, from level 3 on a block has 16 threads or fewer, and the
  // offset-16 step then reads lanes that do not exist, whose values CUDA leaves undefined.
  // fast_benchmark detects on level 0 only (128-thread blocks).
  // Reduce the maximum location to thread 0 within each warp
  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    float max_resp_new = __shfl_down_sync(WARP_MASK, max_resp, offset);
    float max_x_new    = __shfl_down_sync(WARP_MASK, max_x, offset);
    float max_y_new    = __shfl_down_sync(WARP_MASK, max_y, offset);
    if(max_resp_new > max_resp) {
      max_resp = max_resp_new;
      max_x = max_x_new;
      max_y = max_y_new;
    }
  }

  // [SLAM zero-to-hero] Block-level step through shared memory, the on-chip memory all warps of a
  // block can see. extern __shared__: the size comes from the launch (shm_mem_size below, 4 floats
  // per warp, 64 bytes at level 0). Slot 3 stays unused; likely kept so that the slot index is a
  // shift (warp_id << 2).
  // now each warp's lane 0 has the maximum value of its cell
  // reduce in shared memory
  // each warp's lane 0 writes to shm
  // resp, x, y, (level - not used)
  extern __shared__ float s[];
  float * s_data = s + (warp_id << 2);
  float scale = static_cast<float>(1<<level);

  if(lane_id == 0) {
    s_data[0] = max_resp;
    s_data[1] = max_x;
    s_data[2] = max_y;
  }
  // [SLAM zero-to-hero] Barrier: thread 0 must not read a slot before that warp's lane 0 has
  // written it. Every thread of the block reaches this line (it is outside every condition), as
  // __syncthreads() requires; a barrier that only some threads reach can hang or misbehave.
  __syncthreads();
  // [SLAM zero-to-hero] Thread 0 is lane 0 of warp 0 and already holds warp 0's result, so it folds
  // in the other warps (3 at level 0) one by one; for so few entries a serial loop is enough.
  // threadId x & y 0 reduces the warp results
  if(threadIdx.x == 0 && threadIdx.y == 0) {
    s_data = s + 4; // skip self results
    for(int i=1;i<warp_cnt;i++,s_data += 4) {
      float max_resp_s = s_data[0];
      float max_x_s    = s_data[1];
      float max_y_s    = s_data[2];
      if(max_resp_s > max_resp) {
        max_resp = max_resp_s;
        max_x = max_x_s;
        max_y = max_y_s;
      }
    }

    // [SLAM zero-to-hero] One thread per cell touches the global result, each cell belongs to
    // exactly one block, and the levels are separate launches ordered by the stream: plain loads
    // and stores suffice, no atomics. `<` keeps the earlier level's corner on equal scores.
    // Positions are scaled to level-0 pixels (scale = 2^level).
    // Exercise 2 (README), a hint: which of the three per-cell arrays does this kernel write for
    // every cell, and which only for some? Then look at which values processGrid()
    // (detector_base_gpu.cpp) reads, and when.
    if(d_score[cell_id] < max_resp) {
      d_score[cell_id] = max_resp;
      d_pos[cell_id].x = max_x * scale;
      d_pos[cell_id].y = max_y * scale;
      d_level[cell_id] = level;
    }
  }
}

__host__ void detector_base_gpu_grid_nms(const int image_level,
                                         const int min_image_level,
                                         const int image_width,
                                         const int image_height,
                                         const int horizontal_border,
                                         const int vertical_border,
                                         const int cell_size_width,
                                         const int cell_size_height,
                                         const int horizontal_cell_num,
                                         const int vertical_cell_num,
                                         const bool strictly_greater,
                                         /* pitch in bytes/sizeof(float) */
                                         const int response_pitch_elements,
                                         const float * d_response,
                                         float2 * d_pos,
                                         float * d_score,
                                         int * d_level,
                                         cudaStream_t stream) {
  // [SLAM zero-to-hero] Block = one cell at this level: (cell width >> level) threads across and up
  // to 128 threads in all, 32x4 at level 0. Grid = the cell grid, the same at every level, so block
  // (bx, by) always handles cell (bx, by). Dynamic shared memory: 16 bytes per warp.
  const int cell_size_width_level  = (cell_size_width  >> image_level);
  const int cell_size_height_level = (cell_size_height >> image_level);
  int target_threads_per_block = 128;
  kernel_params_t p;
  p.threads_per_block.x = cell_size_width_level;
  p.threads_per_block.y = max(1,min(target_threads_per_block/cell_size_width_level,cell_size_height_level));
  p.threads_per_block.z = 1;
  p.blocks_per_grid.x = horizontal_cell_num;
  p.blocks_per_grid.y = vertical_cell_num;
  p.blocks_per_grid.z = 1;
  // shared memory allocation
  int launched_warp_count = (p.threads_per_block.x*p.threads_per_block.y*p.threads_per_block.z+32-1)/32;
  std::size_t shm_mem_size = launched_warp_count*4*sizeof(float);

  // [SLAM zero-to-hero] Both instantiations are compiled; the host picks one through a function
  // pointer, which <<< >>> accepts like a kernel name. Stream 0 is the per-thread default stream in
  // this file (--default-stream per-thread), as in fast_gpu_cuda_tools.cu.
  decltype(&detector_base_gpu_grid_nms_kernel<false>) kernel;
  if(strictly_greater) {
    kernel = detector_base_gpu_grid_nms_kernel<true>;
  } else {
    kernel = detector_base_gpu_grid_nms_kernel<false>;
  }
  kernel<<<p.blocks_per_grid,p.threads_per_block,shm_mem_size,stream>>>(
                                                image_level,
                                                min_image_level,
                                                image_width,
                                                image_height,
                                                horizontal_border,
                                                vertical_border,
                                                cell_size_width_level,
                                                cell_size_height_level,
                                                response_pitch_elements,
                                                d_response,
                                                d_pos,
                                                d_score,
                                                d_level);
  CUDA_KERNEL_CHECK();
}

} // namespace vilib
