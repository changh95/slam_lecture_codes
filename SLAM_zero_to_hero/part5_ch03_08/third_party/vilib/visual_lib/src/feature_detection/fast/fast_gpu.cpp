/*
 * FAST feature detector on the GPU
 * fast_gpu.cpp
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
// Host side of the CUDA FAST detector. The constructor allocates the corner lookup table and fills
// it on the GPU, once. Each detect() then queues:
//  1. one corner response kernel per pyramid level (fast_gpu_cuda_tools.cu),
//  2. one grid NMS kernel per level (processResponse(), detector_base_gpu_cuda_tools.cu),
//  3. one copy of the per-cell results into pinned host memory, and a wait for it (processGrid()).
// fast_benchmark detects on level 0 only: 2 kernels, a 5,760-byte copy and one synchronization per
// frame, 21.2 us median on the RTX 5090 (README). Launches are asynchronous: the host queues both
// kernels back to back and waits only in copyGridToHost(), while the GPU runs them in stream order,
// so the NMS starts after the response image it reads is complete.
// Nothing is allocated per frame: the lookup table here, and the response images and the result
// grid in DetectorBaseGPU, are allocated in the constructors and reused.
// Borders: the circle needs 3 pixels (MINIMUM_BORDER). The response is also computed one NMS radius
// closer to the edge than the NMS searches, so the 3x3 NMS of the outermost searched pixel compares
// against real scores. With fast_benchmark's border of 8, the NMS searches [8, width - 8) and the
// response is computed on [7, width - 7), and the same for rows.
// Checks: CMakeLists.txt compiles this file with -DNDEBUG, so the assert below and the error check
// in CUDA_API_CALL (cuda_common.h) compile to nothing. The BENCHMARK_* macros are empty as well
// (BENCHMARK_ENABLE is 0 in benchmark.h); fast_benchmark times detect() with easy_profiler.
#include "vilib/feature_detection/fast/fast_gpu.h"
#include "vilib/feature_detection/fast/fast_gpu_cuda_tools.h"
#include "vilib/feature_detection/detector_benchmark.h"
#include "vilib/cuda_common.h"
#include "vilib/timer.h"

namespace vilib {
// Copy back the response images (as debug)
#define SAVE_RESPONSE                 0
// We need a minimal border, because of the Bresenham circle
#define MINIMUM_BORDER                3
// Modify MINIMUM_BORDER if NONMAXIMUM_SUPPRESSION_SIZE requires more border
#if ((DETECTOR_BASE_NMS_SIZE/2) > MINIMUM_BORDER)
#undef MINIMUM_BORDER
#define MINIMUM_BORDER                (DETECTOR_BASE_NMS_SIZE/2)
#endif /* (NONMAXIMUM_SUPPRESSION_SIZE/2) > MINIMUM_BORDER */

FASTGPU::FASTGPU(const std::size_t image_width,
                 const std::size_t image_height,
                 const std::size_t cell_size_width,
                 const std::size_t cell_size_height,
                 const std::size_t min_level,
                 const std::size_t max_level,
                 const std::size_t horizontal_border,
                 const std::size_t vertical_border,
                 const float threshold,
                 const int min_arc_length,
                 fast_score score) :
  DetectorBaseGPU(image_width,
                  image_height,
                  cell_size_width,
                  cell_size_height,
                  min_level,
                  max_level,
                  std::max(static_cast<std::size_t>(MINIMUM_BORDER),horizontal_border),
                  std::max(static_cast<std::size_t>(MINIMUM_BORDER),vertical_border),
                  true),
  threshold_(threshold),
  min_arc_length_(min_arc_length),
  score_(score),
  // [SLAM zero-to-hero] The response region: one NMS radius (DETECTOR_BASE_NMS_SIZE/2 = 1) closer
  // to the edge than the NMS border passed to DetectorBaseGPU above, but never under 3.
  det_horizontal_border_(std::max(MINIMUM_BORDER,static_cast<int>(horizontal_border) - static_cast<int>(DETECTOR_BASE_NMS_SIZE/2))),
  det_vertical_border_(std::max(MINIMUM_BORDER,static_cast<int>(vertical_border) - static_cast<int>(DETECTOR_BASE_NMS_SIZE/2))) {
  // [SLAM zero-to-hero] Compiled out (-DNDEBUG), and it matters: the GPU pre-test is exact only for
  // arcs of 8 or more (fast_gpu_cuda_tools.cu), so a min_arc_length of 7 or less silently loses
  // corners.
  assert(min_arc_length >= 9 && min_arc_length <= 12);
  // LUT for corner response
#if FAST_GPU_USE_LOOKUP_TABLE
#if FAST_GPU_USE_LOOKUP_TABLE_BITBASED
  // [SLAM zero-to-hero] 2048 words x 32 bits: one bit per 16-bit darker/brighter mask.
  CUDA_API_CALL(cudaMalloc((void**)&d_corner_lut_,8*1024));
#else
  CUDA_API_CALL(cudaMalloc((void**)&d_corner_lut_,64*1024));
#endif /* FAST_GPU_USE_LOOKUP_TABLE_BITBASED */
  // [SLAM zero-to-hero] Built on the GPU, asynchronously; nothing waits for it here. Stream order
  // puts it before the first detect(): both use stream_, which with stream_ = 0 and the kernels
  // compiled with --default-stream per-thread means the same host thread's default stream.
  // calculate the lookup table
  fast_gpu_calculate_lut(d_corner_lut_,min_arc_length_,stream_);
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
}

FASTGPU::~FASTGPU(void) {
#if FAST_GPU_USE_LOOKUP_TABLE
  // release the lookup table
  CUDA_API_CALL(cudaFree(d_corner_lut_));
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
}

void FASTGPU::detectBase(const std::vector<std::shared_ptr<Subframe>> & pyramid) {
  /*
   * FAST corner detection process:
   * - the image is already in GPU memory
   * - determine corner points
   * - apply non-maximum suppression on them
   *   and populate the feature grid
   * - copy the feature grid back to host memory
   */
  BENCHMARK_START_DEVICE(DetectorBenchmark,CRF,stream_);
  // [SLAM zero-to-hero] One launch per level, each into its own response image, all queued without
  // waiting. The pyramid is already in device memory (the caller uploaded it), so no image data
  // crosses PCIe here.
  for(std::size_t level=min_level_;level<pyramid.size() && level<max_level_;++level) {
    std::size_t level_resp = level-min_level_;
    fast_gpu_calc_corner_response(pyramid[level]->width_,
                                  pyramid[level]->height_,
                                  pyramid[level]->pitch_,
                                  pyramid[level]->data_,
                                  det_horizontal_border_,
                                  det_vertical_border_,
#if FAST_GPU_USE_LOOKUP_TABLE
                                  d_corner_lut_,
#endif /* FAST_GPU_USE_LOOKUP_TABLE */
                                  threshold_,
                                  min_arc_length_,
                                  score_,
                                  responses_[level_resp].pitch_elements_,
                                  responses_[level_resp].data_,
                                  stream_);
  }
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,CRF,stream_);
  // [SLAM zero-to-hero] Debug aid: SAVE_RESPONSE 1 writes each response image to a TIFF. It
  // synchronizes and copies the full float image back (1.4 MB at 752x480): the transfer the
  // per-cell grid exists to avoid.
#if SAVE_RESPONSE
  saveResponses("fast_gpu");
#endif /* SAVE_RESPONSE */
  // Perform non-maximum suppression on the response images
  processResponse();
}

// [SLAM zero-to-hero] fast_benchmark calls this overload: processGrid() copies the per-cell results
// back and fills getPoints() and getGrid() on the host. The overload below hands the raw host
// arrays (position, score and level per cell) to a callback instead.
void FASTGPU::detect(const std::vector<std::shared_ptr<Subframe>> & pyramid) {
  detectBase(pyramid);
  processGrid();
}

void FASTGPU::detect(const std::vector<std::shared_ptr<Subframe>> & pyramid,
                     std::function<void(const std::size_t &  /* cell count */,
                                        const float *  /* pos */,
                                        const float *  /* score */,
                                        const int *  /* level */)> callback) {
  detectBase(pyramid);
  processGridCustom(callback);
}

} // namespace vilib
