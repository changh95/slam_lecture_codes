/*
 * Functions for creating image pyramids on the GPU
 * pyramid_gpu.cu
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
// Each level halves the one before: out(x, y) = (a + b + c + d) >> 2 over the 2x2 block at
// (2x, 2y). The shift rounds down, exactly like pyramid_cpu.cpp, so the GPU and CPU pyramids are
// identical; cv::resize(INTER_AREA) rounds to nearest (README, "1. Image pyramid"). An odd last
// column or row of the input is dropped.
// Mapping: one thread computes N consecutive output pixels of one row, N = 4, 2 or 1 (the largest
// that divides the output width), in blocks of 64x2 threads (32x2 when a row needs at most 32
// threads). A warp is 32 neighbouring threads of one row. For EuRoC's 752x480 frames:
//   level 1  376x240  N = 4  94 threads per row  2 x 120 blocks
//   level 2  188x120  N = 4  47 threads per row  1 x 60 blocks
//   level 3   94x60   N = 2  47 threads per row  1 x 30 blocks
//   level 4   47x30   N = 1  47 threads per row  1 x 15 blocks
// Memory: global memory only, one pitched buffer per level (cudaMallocPitch, from PyramidPool).
// No shared memory: every input byte feeds exactly one output pixel, so threads share nothing.
// Techniques:
//  1. Pitched rows: every row starts on an address aligned by the driver, so the 256 bytes a warp
//     reads from an input row (N = 4) cover the fewest 32-byte sectors, and the uchar2 / uchar4
//     casts are aligned on every row.
//  2. Several outputs per thread: N = 4 needs a quarter of the threads and does the index math
//     once per 4 pixels. The vector types were meant to widen the memory accesses as well, but in
//     the course image they do not (see the note in the kernel).
//  3. Read-only data path: const + __restrict__ lets nvcc use non-coherent loads
//     (LDG.E.*.CONSTANT in the SASS). The N iterations re-read the same sectors; those repeats
//     normally hit L1, so L2 and DRAM see every byte once.
//  4. No tail code: the host picks N per level so that no thread's pixels cross the row end; the
//     kernel's bounds check only matters for the surplus threads of the last blocks.
//  5. One launch per level, all in one stream, with no host synchronization in between: stream
//     order alone makes level l wait for level l - 1.
// The cost is fixed overhead, not the kernel body. README point 4: the four kernels move about
// 0.6 MB, about 0.3 us at the RTX 5090's bandwidth, yet the stage takes 18.5 us; and levels 2 to 4
// have only 60, 30 and 15 blocks for 170 SMs. CUDA Graphs, or one kernel that builds several
// levels from a tile in shared memory, would attack that; vilib does neither.
#include <opencv2/imgproc.hpp>
#include "vilib/preprocess/pyramid.h"
#include "vilib/cuda_common.h"

namespace vilib {
// [SLAM zero-to-hero] The texture path is switched off. It would sample the centre of each 2x2
// block with linear filtering, so the texture unit averages the 4 pixels. But the value goes
// through normalized float and back (/ 255, then * 255 and a truncating cast), so it is not
// guaranteed to match the integer CPU code bit for bit. Likely why it is off: the integer kernel
// is exact and needs no texture setup per level. Texture references (texture<>,
// cudaBindTexture2D) were removed in CUDA 12, so USE_TEXTURE_MEMORY 1 with USE_TEXTURE_OBJECTS 0
// no longer compiles with the course's CUDA 12.8.
#define USE_TEXTURE_MEMORY  0

#define USE_TEXTURE_OBJECTS 0

#if (USE_TEXTURE_MEMORY == 1) && (USE_TEXTURE_OBJECTS == 0)
static texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> d_image_in_tex;
#endif /* USE_TEXTURE_MEMORY */

#if USE_TEXTURE_MEMORY
template <typename T, const int N>
static __global__ void image_halfsample_gpu_tex_kernel(T * __restrict__ d_image_out,
#if USE_TEXTURE_OBJECTS
                                                       cudaTextureObject_t d_image_in_tex,
#endif /* USE_TEXTURE_OBJECTS */
                                                       const unsigned int width_dst_px,
                                                       const unsigned int height_dst_px,
                                                       const unsigned int pitch_dst_px) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if((x < width_dst_px) && (y < height_dst_px)) {
    const int dst = y*pitch_dst_px/N + x; //every thread writes N bytes. the next row starts at pitch_Dst_px/N
    float src_x = N*2.0f*x + 1.0f;        //every thread reads in 2x4 bytes
    float src_y = 2.0f*y + 1.0f;
    #pragma unroll N
    for(int i=0;i<N;++i,src_x += 2.0f) {
      *(((unsigned char*)(d_image_out+dst))+i) = (unsigned char)(255.0f*
#if USE_TEXTURE_OBJECTS
      tex2D<float>
#else
      tex2D
#endif /* USE_TEXTURE_OBJECTS */
        (d_image_in_tex,src_x,src_y));
    }
  }
}
#else
// [SLAM zero-to-hero] T is the type that holds one thread's N output pixels (uchar4 for N = 4), so
// d_image_out + dst points at the thread's first output byte. The input is viewed as uchar2, one
// horizontal pixel pair per element. width_px is the output width in units of T (pixels / N).
template <typename T, const int N>
__global__ void image_halfsample_gpu_kernel(const uchar2 * __restrict__ d_image_in,
                                 const unsigned int pitch_src_px,
                                 T * __restrict__ d_image_out,
                                 const unsigned int width_px,
                                 const unsigned int height_px,
                                 const unsigned int pitch_dst_px) {
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  // [SLAM zero-to-hero] The grid is rounded up to whole blocks; this idles the surplus threads.
  if((x < width_px) && (y < height_px)) {
    // [SLAM zero-to-hero] Output index in units of T: a row is pitch_dst_px / N elements long,
    // which needs a pitch divisible by N. cudaMallocPitch's pitch is; a packed row (pitch = width)
    // is too, because N divides the width.
    const int dst  = y*pitch_dst_px/N + x; //every thread writes N bytes. the next row starts at pitch_dst_px/N
    // [SLAM zero-to-hero] Input index in units of uchar2. Output row y needs input rows 2y and
    // 2y + 1. Row 2y starts at byte 2y * pitch, which is uchar2 index y * pitch, and the next row
    // is pitch / 2 uchar2 further on. This needs an even source pitch: always true for pitched
    // memory, but a packed buffer of odd width (Subframe's LINEAR or UNIFIED types) would break it.
    int src_top    = y*pitch_src_px + x*N; //every thread reads in Nx2 bytes
    int src_bottom = y*pitch_src_px + x*N + (pitch_src_px/2);
    // [SLAM zero-to-hero] N unrolled iterations, each reads one pixel pair from the top and one
    // from the bottom row and writes one output byte. The sum is at most 4 * 255 = 1020; >> 2
    // rounds down. What the compiler made of it in the course image (CUDA 12.8, sm_120;
    // `cuobjdump -sass /usr/local/lib/libvilib.so | grep -E 'Function|LDG|STG'`): for N = 4,
    // 2 LDG.E.U16 + 12 LDG.E.U8 loads and 4 STG.E.U8 stores per thread. The uchar2 / uchar4 types
    // did not turn into 8-byte loads or a 4-byte store: a warp-wide load moves 32 or 64 bytes
    // spread over 256, a store 32 bytes spread over 128. An experiment: load one uint2 per row and
    // store one uchar4 (2 loads + 1 store; pitched rows keep them aligned), then time it. At
    // 752x480 launch overhead dominates (README point 4), so expect a small difference.
    #pragma unroll N
    for(int i=0;i<N;++i) {
      const uchar2 t2 = d_image_in[src_top++];
      const uchar2 b2 = d_image_in[src_bottom++];
      *(((unsigned char*)(d_image_out+dst))+i) = (unsigned char)(((unsigned int)t2.x + (unsigned int)t2.y + (unsigned int)b2.x + (unsigned int)b2.y)>>2);
    }
  }
}
#endif /* USE_TEXTURE_MEMORY */

// [SLAM zero-to-hero] Host code: runs on the CPU and only enqueues one kernel launch, then returns
// long before the GPU has run it.
static inline __host__ void pyramid_create_level_gpu(const unsigned char * d_img_src,
                                                      unsigned char * d_img_dst,
                                                      std::size_t & img_src_pitch,
                                                      std::size_t & img_dst_pitch,
                                                      std::size_t & img_src_width,
                                                      std::size_t & img_dst_width,
                                                      std::size_t & img_src_height,
                                                      std::size_t & img_dst_height,
                                                      cudaStream_t stream) {
#if USE_TEXTURE_MEMORY
#if USE_TEXTURE_OBJECTS
  cudaTextureObject_t tex_object;
  cudaResourceDesc tex_res;
  memset(&tex_res,0,sizeof(cudaResourceDesc));
  tex_res.resType = cudaResourceTypePitch2D;
  tex_res.res.pitch2D.width = img_src_width;
  tex_res.res.pitch2D.height = img_src_height;
  tex_res.res.pitch2D.pitchInBytes = img_src_pitch;
  tex_res.res.pitch2D.devPtr = (void*)d_img_src;
  tex_res.res.pitch2D.desc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);

  cudaTextureDesc tex_desc;
  memset(&tex_desc,0,sizeof(cudaTextureDesc));
  tex_desc.normalizedCoords = 0;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.readMode = cudaReadModeNormalizedFloat;

  // create texture object on the fly
  CUDA_API_CALL(cudaCreateTextureObject(&tex_object,&tex_res,&tex_desc,NULL));
#else
  // Bind the texture memory
  d_image_in_tex.addressMode[0] = cudaAddressModeClamp;
  d_image_in_tex.addressMode[1] = cudaAddressModeClamp;
  d_image_in_tex.filterMode     = cudaFilterModeLinear;
  d_image_in_tex.normalized     = 0;
  d_image_in_tex.channelDesc    = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
  CUDA_API_CALL(cudaBindTexture2D(NULL,&d_image_in_tex,d_img_src,&d_image_in_tex.channelDesc,img_src_width,img_src_height,img_src_pitch));
#endif /* USE_TEXTURE_OBJECTS */
#else
  // Unused parameters
  (void)img_src_width;
  (void)img_src_height;
#endif /* USE_TEXTURE_MEMORY */

  // Use the most efficient vectorized version
  // [SLAM zero-to-hero] N = v is the largest of 4, 2, 1 that divides the output width, so every
  // thread's pixels lie inside the row and the kernel needs no tail code. The price: a width of
  // 4k + 2 runs the whole row with N = 2 and an odd width with N = 1 (EuRoC's level 4 is 47 wide),
  // instead of N = 4 plus a short tail. Blocks are thread_num_x x 2 threads, where thread_num_x is
  // one row of threads rounded up to whole warps and capped at 64. Likely why the blocks are this
  // small (128 threads): on vilib's Jetson targets (TX2: 2 SMs, Xavier: 8 SMs) even the coarse
  // levels still get several blocks to spread over the SMs.
  for(unsigned int v=4;v>0;v=v/2) {
    if(img_dst_width % v == 0) {
      const unsigned int img_dst_width_n = img_dst_width/v;
      const unsigned int thread_num_x = min(64,((img_dst_width_n+32-1)/32)*32);
      const kernel_params_t p = cuda_gen_kernel_params_2d(img_dst_width_n,img_dst_height,thread_num_x,2);
      // [SLAM zero-to-hero] Launches below: <<<grid, block, 0 bytes of dynamic shared memory,
      // stream>>>. This .cu file is compiled with --default-stream per-thread, so stream == 0 here
      // is the calling host thread's own default stream, whereas a 0 passed to a copy in a .cpp
      // file (storage/opencv.cpp) is the legacy default stream (README, "Two default streams").
      // No error check follows a launch (and CUDA_KERNEL_CHECK is compiled out by -DNDEBUG).
      switch(v) {
        case 1:
#if USE_TEXTURE_MEMORY
          image_halfsample_gpu_tex_kernel<uchar1,1><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar1*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar1,1><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar1*)d_img_dst,
            (unsigned int)img_dst_width,
            (unsigned int)img_dst_height,
            (unsigned int)img_dst_pitch
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
        case 2:
#if USE_TEXTURE_MEMORY
          // [SLAM zero-to-hero] Pitfall in this disabled branch (here and in case 4): it passes
          // img_dst_width, not img_dst_width_n, but the grid is sized for img_dst_width_n. The
          // kernel's x test then lets the surplus threads of the last block through, and they
          // write past the row's last pixel.
          image_halfsample_gpu_tex_kernel<uchar2,2><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar2,2><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar2*)d_img_dst,
            (unsigned int)(img_dst_width_n),
            (unsigned int)(img_dst_height),
            (unsigned int)(img_dst_pitch)
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
        case 4:
#if USE_TEXTURE_MEMORY
          image_halfsample_gpu_tex_kernel<uchar4,4><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar4*)d_img_dst,
            #if USE_TEXTURE_OBJECTS
            tex_object,
            #endif
            img_dst_width,
            img_dst_height,
            img_dst_pitch
          );
#else
          image_halfsample_gpu_kernel<uchar4,4><<<p.blocks_per_grid,p.threads_per_block,0,stream>>>(
            (uchar2*)d_img_src,
            (unsigned int)img_src_pitch,
            (uchar4*)d_img_dst,
            (unsigned int)(img_dst_width_n),
            (unsigned int)(img_dst_height),
            (unsigned int)(img_dst_pitch)
          );
#endif /* USE_TEXTURE_MEMORY */
          break;
      }
      break;
    }
  }

#if USE_TEXTURE_MEMORY
#if USE_TEXTURE_OBJECTS
  // Destroy the created texture object
  CUDA_API_CALL(cudaDestroyTextureObject(tex_object));
#else
  // Unbind the texture memory
  CUDA_API_CALL(cudaUnbindTexture(d_image_in_tex));
#endif /* USE_TEXTURE_OBJECTS */
#endif /* USE_TEXTURE_MEMORY */
}

// [SLAM zero-to-hero] One launch per level, back to back in one stream, with no synchronization
// in between: level l reads level l - 1, and stream order alone guarantees that it is complete.
// Both overloads return once the launches are queued; whoever needs the result waits for the
// stream (the exercise ends each timed block with cudaDeviceSynchronize()).
__host__ void pyramid_create_gpu(std::vector<unsigned char *> & d_images,
                                 std::vector<std::size_t> & width,
                                 std::vector<std::size_t> & height,
                                 std::vector<std::size_t> & pitch,
                                 unsigned int levels,
                                 cudaStream_t stream) {
  for(std::size_t l=1;l<levels;++l) {
    pyramid_create_level_gpu(d_images[l-1],
                             d_images[l],
                             pitch[l-1],
                             pitch[l],
                             width[l-1],
                             width[l],
                             height[l-1],
                             height[l],
                             stream);
  }
}

__host__ void pyramid_create_gpu(std::vector<std::shared_ptr<Subframe>> & d_subframes,
                                 cudaStream_t stream) {
  for(std::size_t l=1;l<d_subframes.size();++l) {
    pyramid_create_level_gpu(d_subframes[l-1]->data_,
                             d_subframes[l]->data_,
                             d_subframes[l-1]->pitch_,
                             d_subframes[l]->pitch_,
                             d_subframes[l-1]->width_,
                             d_subframes[l]->width_,
                             d_subframes[l-1]->height_,
                             d_subframes[l]->height_,
                             stream);
  }
}

__host__ void pyramid_display(const std::vector<std::shared_ptr<Subframe>> & subframes) {
  for(std::size_t l=0;l<subframes.size();++l) {
    subframes[l]->display();
  }
}

} // namespace vilib
