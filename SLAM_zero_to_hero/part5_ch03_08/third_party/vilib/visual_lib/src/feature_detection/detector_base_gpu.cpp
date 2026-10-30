/*
 * Base class for GPU feature detectors
 * detector_base_gpu.cpp
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
// Host side shared by the GPU detectors (FASTGPU, HarrisGPU). It owns the device buffers, launches
// the grid NMS for each pyramid level and brings the per-cell results back.
// Memory, all allocated once in the constructor and reused every frame (cudaMalloc and cudaFree are
// slow and can synchronize the device):
//  - One response image per level, pitched floats (cudaMallocPitch): rows start on aligned
//    addresses, so the warps' row accesses coalesce. The FAST kernel writes every pixel each frame.
//  - The feature grid, 16 bytes per cell (x, y, score, level), as a struct of arrays, in one device
//    buffer and one pinned host buffer of the same layout, so the whole grid is one copy: 5,760
//    bytes for 752x480 and 32-px cells (360 cells), against 1.4 MB for the response image. It is
//    the only data a detection sends back to the host.
//  - Pinned host memory (cudaMallocHost) is page-locked, so the GPU's copy engine writes it
//    directly by DMA. With pageable memory the driver stages the copy through a pinned buffer of
//    its own, and a device-to-host cudaMemcpyAsync returns only once the copy is done.
// Synchronization: the kernels are queued and the host goes on; copyGridToHost() is the only place
// a detection waits for the GPU. stream_ is 0 unless setStream() is called. This file is compiled
// by the C++ compiler, without --default-stream per-thread, so 0 here is the legacy default stream,
// while the kernels in the .cu files go to the per-thread one. The legacy stream waits for the
// per-thread streams, so the copy still runs after the kernels (README, "Two default streams").
// Error checks: with -DNDEBUG, CUDA_API_CALL() is the bare call (cuda_common.h), so a failed
// allocation or copy goes unreported.
#include <opencv2/highgui.hpp>
#include "vilib/cuda_common.h"
#include "vilib/feature_detection/detector_base_gpu.h"
#include "vilib/feature_detection/detector_base_gpu_cuda_tools.h"
#include "vilib/feature_detection/detector_benchmark.h"

namespace vilib {

DetectorBaseGPU::DetectorBaseGPU(const std::size_t image_width,
                                 const std::size_t image_height,
                                 const std::size_t cell_size_width,
                                 const std::size_t cell_size_height,
                                 const std::size_t min_level,
                                 const std::size_t max_level,
                                 const std::size_t horizontal_border,
                                 const std::size_t vertical_border,
                                 const bool strictly_greater) :
  DetectorBase(image_width,
               image_height,
               cell_size_width,
               cell_size_height,
               min_level,
               max_level,
               horizontal_border,
               vertical_border),
  strictly_greater_(strictly_greater),
  stream_(0) {
  // [SLAM zero-to-hero] Both asserts are compiled out (-DNDEBUG). The first keeps the level-0 NMS
  // block a multiple of 32 threads, so its warps are full, as the shuffles need. The second makes
  // the cell height divide evenly at every level detected.
  assert((cell_size_width%32) == 0);
  assert((cell_size_height%static_cast<std::size_t>(pow(2,max_level-1))) == 0);
  /*
   * Preallocate response buffers (atm 1 for each level)
   *
   * Note to future self and others:
   * because we don't allocate space for anything below min_level_,
   * if min_level_ is not zero, then all the consecutive levels will be shifted
   * in the responses_ vector.
   */
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t pitch_bytes;
    float * data;
    std::size_t width = (image_width/(1<<l));
    std::size_t height = (image_height/(1<<l));
    // [SLAM zero-to-hero] pitch_bytes comes back rounded up for alignment. FeatureResponse also
    // keeps it in floats (pitch_elements_), the unit the kernels index in.
    CUDA_API_CALL(cudaMallocPitch((void**)&data,
                                  &pitch_bytes,
                                  width*sizeof(float),
                                  height));
    struct FeatureResponse response(width,height,pitch_bytes,data);
    responses_.push_back(response);
  }
  /*
   * Preallocate FeaturePoint struct of arrays
   * Note to future self:
   * we use SoA, because of the efficient bearing vector calculation
   * float x_
   * float y_:                                      | 2x float
   * float score_:                                  | 1x float
   * int level_:                                    | 1x int
   */
  // [SLAM zero-to-hero] Layout, N = cell count: [x0 y0 x1 y1 ... (N float2)][N scores][N levels],
  // 16 bytes per cell. In each array, cell i is at index i (cell_id = row * n_cols + col in the
  // kernel). One allocation per side, so one copy moves everything.
  const std::size_t nr_4bytes = 0 + 4;
  const std::size_t bytes_per_featurepoint = sizeof(float)*nr_4bytes;
  feature_cell_count_ = n_cols_ * n_rows_;
  feature_grid_bytes_ = feature_cell_count_*bytes_per_featurepoint;
  CUDA_API_CALL(cudaMalloc((void**)&d_feature_grid_,feature_grid_bytes_));
  // [SLAM zero-to-hero] Pinned (page-locked) host memory: the target of the one copy per frame.
  CUDA_API_CALL(cudaMallocHost((void**)&h_feature_grid_,feature_grid_bytes_));
  // Initialize pointers
  // Device & Host pointers
  h_pos_ = h_feature_grid_;
  d_pos_ = (float2*)(d_feature_grid_ );
  h_score_ = (h_feature_grid_ + feature_cell_count_*2);
  d_score_ = (d_feature_grid_ + feature_cell_count_*2);
  h_level_ = (int*)(h_feature_grid_ + feature_cell_count_*3);
  d_level_ = (int*)(d_feature_grid_ + feature_cell_count_*3);
}

void DetectorBaseGPU::setStream(cudaStream_t stream) {
  stream_ = stream;
}

cudaStream_t DetectorBaseGPU::getStream(void) {
  return stream_;
}

DetectorBaseGPU::~DetectorBaseGPU(void) {
  // Free response buffers
  for(std::size_t l=0;l<responses_.size();++l) {
    CUDA_API_CALL(cudaFree(responses_[l].data_));
  }
  // Free grid buffers
  CUDA_API_CALL(cudaFree(d_feature_grid_));
  CUDA_API_CALL(cudaFreeHost(h_feature_grid_));
}

// [SLAM zero-to-hero] The one synchronization of a detection: queue the copy of the whole grid,
// then block until it is done; the legacy stream waits for the per-thread stream first (see
// above), so both kernels have finished too. After that the
// host reads the arrays directly. 5,760 bytes in fast_benchmark, part of the README's 21.2 us
// median for detect().
// Exercise 2 (README): initcheck reports this copy as reading partly uninitialized device memory.
// Hint: look at which per-cell values the grid NMS kernel writes for a cell without a corner
// (detector_base_gpu_cuda_tools.cu), and at which values processGrid() reads, and when.
void DetectorBaseGPU::copyGridToHost(void) {
  CUDA_API_CALL(cudaMemcpyAsync(h_feature_grid_,
                                d_feature_grid_,
                                feature_grid_bytes_,
                                cudaMemcpyDeviceToHost,
                                stream_));
  CUDA_API_CALL(cudaStreamSynchronize(stream_));
}

// [SLAM zero-to-hero] Debug path (SAVE_RESPONSE in fast_gpu.cpp): copies a whole response image
// back, 1.4 MB at 752x480, the transfer the per-cell grid avoids. The 2D copy drops the row
// padding: the cv::Mat rows are width floats, the device rows pitch_bytes_.
void DetectorBaseGPU::copyResponseTo(const std::size_t level, cv::Mat & response) const {
  assert(level < responses_.size());
  assert(level >= min_level_);
  assert(level < max_level_);
  std::size_t level_resp = level-min_level_;
  CUDA_API_CALL(cudaMemcpy2DAsync(response.data,
                                  response.step,
                                  responses_[level_resp].data_,
                                  responses_[level_resp].pitch_bytes_,
                                  response.cols*sizeof(float),
                                  response.rows,
                                  cudaMemcpyDeviceToHost,
                                  stream_));
  CUDA_API_CALL(cudaStreamSynchronize(stream_));
}

// [SLAM zero-to-hero] One grid NMS launch per level, all queued on stream_ without waiting. They
// run one after another, so each level's kernel sees the per-cell results of the levels before it.
// `#if 1` selects the grid version; the regular NMS in the #else only zeroes the non-maxima in the
// response image and fills no grid.
void DetectorBaseGPU::processResponse(void) {
  // Perform GRID NMS on the responses
  BENCHMARK_START_DEVICE(DetectorBenchmark,NMS,stream_);
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t l_resp = l-min_level_;
#if 1
    detector_base_gpu_grid_nms(l,
                               min_level_,
                               responses_[l_resp].width_,
                               responses_[l_resp].height_,
                               horizontal_border_,
                               vertical_border_,
                               cell_size_width_,
                               cell_size_height_,
                               n_cols_,
                               n_rows_,
                               strictly_greater_,
                               /* wpitch in bytes/sizeof(float) */
                               responses_[l_resp].pitch_elements_,
                               responses_[l_resp].data_,
                               d_pos_,
                               d_score_,
                               d_level_,
                               stream_);
#else
    // Just for comparison purposes in the paper
    detector_base_gpu_regular_nms(responses_[l_resp].width_,
                                  responses_[l_resp].height_,
                                  horizontal_border_,
                                  vertical_border_,
                                  /* pitch in bytes/sizeof(float) */
                                  responses_[l_resp].pitch_elements_,
                                  responses_[l_resp].data_,
                                  stream_);
#endif /* 0 */
  }
  BENCHMARK_STOP_DEVICE(DetectorBenchmark,NMS,stream_);
}

void DetectorBaseGPU::saveResponses(const char * prefix) const {
  if(stream_ == NULL) {
    CUDA_API_CALL(cudaDeviceSynchronize());
  } else {
    CUDA_API_CALL(cudaStreamSynchronize(stream_));
  }
  for(std::size_t l=min_level_;l<max_level_;++l) {
    std::size_t l_resp = l-min_level_;
    cv::Mat response = cv::Mat(responses_[l_resp].height_,responses_[l_resp].width_,CV_32FC1);
    copyResponseTo(l,response);
    std::string filename = prefix;
    filename += "_l";
    filename += std::to_string(l);
    filename += ".tiff";
    // save response image
    cv::imwrite(filename.c_str(),response);
  }
}

// [SLAM zero-to-hero] What FASTGPU::detect() ends with: the copy back, then a host loop over the
// cells (360 in fast_benchmark) that fills getPoints() and getGrid().
void DetectorBaseGPU::processGrid(void) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  for(std::size_t i=0;i<feature_cell_count_;++i) {
    if(h_score_[i] > 0.0f) {
        keypoints_[i] = FeaturePoint((double)h_pos_[2*i],
                                     (double)h_pos_[2*i+1],
                                     (double)h_score_[i],
                                     (int)h_level_[i]);
      grid_.setOccupied(i);
    }
  }
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

void DetectorBaseGPU::processGridAndThreshold(float quality_level) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  // Find the maximum score in the grid
  // Note: this could be found also on the GPU side easily
  const float max_element = *(std::max_element(h_score_,h_score_+feature_cell_count_));
  const float threshold = max_element*quality_level;
  for(std::size_t i=0;i<feature_cell_count_;++i) {
    if(h_score_[i] > threshold) {
      keypoints_[i] = FeaturePoint((double)h_pos_[2*i],
                                    (double)h_pos_[2*i+1],
                                    (double)h_score_[i],
                                    (int)h_level_[i]);
      grid_.setOccupied(i);
    }
  }
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

void DetectorBaseGPU::processGridCustom(std::function<void(const std::size_t &,
                                                           const float *,
                                                           const float *,
                                                           const int *)> callback) {
  BENCHMARK_START_HOST(DetectorBenchmark,Download,true);
  copyGridToHost();
  callback(feature_cell_count_,h_pos_,h_score_,h_level_);
  BENCHMARK_STOP_HOST(DetectorBenchmark,Download);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,Pyramid);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,CRF);
  BENCHMARK_COLLECT_DEVICE(DetectorBenchmark,NMS);
}

} // namespace vilib
