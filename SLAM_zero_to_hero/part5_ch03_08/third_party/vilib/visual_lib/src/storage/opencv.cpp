/*
 * Functions for handling OpenCV types
 * opencv.cpp 
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
// Every copy between a cv::Mat and a Subframe ends here, as one cudaMemcpy2D(Async) call.
// The upload path of a frame: Frame -> preprocess_image() -> Subframe::copy_from(img, true,
// stream) -> opencv_copy_from_image_to_gpu() -> opencv_copy_from_image_common().
//  1. Why a 2D copy: the Mat's rows are packed (spitch = step = 752 bytes for EuRoC), while the
//     device rows are pitch_ bytes apart (dpitch), so the copy moves `height` rows of `width`
//     bytes and re-spaces them. How fast that is depends on the row width. README: in a side
//     experiment, rows that are a multiple of 128 bytes copied as fast as one contiguous block,
//     752-byte rows took 2-5x longer; a EuRoC frame took 94.7 us as one 2D copy from pinned
//     memory, 24.4 us as a contiguous copy plus a 2D copy on the device.
//  2. Pageable source (an ordinary cv::Mat): the copy engine can only read page-locked memory, so
//     the driver copies the frame on the CPU into a pinned staging buffer of its own and transfers
//     it from there. The CUDA docs say such an Async copy "might be synchronous"; it returns only
//     after the CPU has copied the frame out (README: 103.3 us per frame for vilib's upload).
//  3. Pinned source (cudaMallocHost, or cudaHostRegister below): the copy engine reads the Mat
//     directly, and the Async call returns at once. The Mat must then stay unchanged until the
//     stream has executed the copy.
//  4. Streams: this is a .cpp file, built without --default-stream per-thread, so stream 0 here
//     is the legacy default stream, while the pyramid kernels launched with 0 from pyramid_gpu.cu
//     run on the per-thread default stream. The legacy stream synchronizes with all blocking
//     streams, which keeps upload and kernels in order (README, "Two default streams").
//  5. Jetson: CPU and GPU share one DRAM, so this "upload" is a copy within the same memory. A
//     frame captured into unified or mapped pinned memory could be read by the kernels in place;
//     vilib still copies it, because its pyramids are always PITCHED_DEVICE_MEMORY.
//  6. No checks: nothing compares the Mat's size with the destination buffer, and CUDA_API_CALL's
//     error check is compiled out by -DNDEBUG (the call itself stays), so a frame larger than the
//     Subframe or a failed copy goes unnoticed.
#include <assert.h>
#include "vilib/storage/opencv.h"
#include "vilib/cuda_common.h"

namespace vilib {

static inline void opencv_copy_to_image_common(cv::Mat & dst_img,
                                               const unsigned char * src_img,
                                               unsigned int src_img_width,
                                               unsigned int src_img_height,
                                               unsigned int src_img_element_size,
                                               unsigned int src_img_pitch,
                                               bool async,
                                               cudaStream_t stream_num,
                                               enum cudaMemcpyKind dir) {
  // Initialize the destination image
  // TODO: currently we only support grayscale images
  assert(src_img_element_size == 1);
  if(dst_img.cols != (int)src_img_width ||
     dst_img.rows != (int)src_img_height) {
    dst_img = cv::Mat(src_img_height,src_img_width,CV_8U);
  }

  // Do the copying
  void * dst = (void*)dst_img.data;
  const void * src = (const void *)src_img;
  std::size_t dpitch = dst_img.step;
  std::size_t spitch = src_img_pitch;
  std::size_t width  = src_img_width*src_img_element_size;
  std::size_t height = src_img_height;
  if(async) {
    CUDA_API_CALL(cudaMemcpy2DAsync(dst,dpitch,
                                    src,spitch,
                                    width,height,
                                    dir,stream_num));
  } else {
    CUDA_API_CALL(cudaMemcpy2D(dst,dpitch,
                               src,spitch,
                               width,height,
                               dir));
  }
}

// [SLAM zero-to-hero] Exercise 1 is patched here. For dir == cudaMemcpyHostToDevice, replace the
// one host-to-device 2D copy by two copies in the same stream: a contiguous cudaMemcpyAsync of the
// whole Mat (rows * step bytes) into a device staging buffer, then a device-to-device
// cudaMemcpy2DAsync from it into dst with dpitch. Allocate the staging buffer once, not per frame
// (pyramid_pool.cpp explains why). Leave the host-to-host path (opencv_copy_from_image_to_host
// also ends here) as it is. pyramid_benchmark's upload_pinned_1D shows the two calls.
static inline void opencv_copy_from_image_common(const cv::Mat & src_img,
                                                 unsigned char * dst_img,
                                                 unsigned int dst_img_pitch,
                                                 bool async,
                                                 cudaStream_t stream_num,
                                                 enum cudaMemcpyKind dir) {
  //Do the copying
  const void * src = (const void *)src_img.data;
  void * dst = (void*)dst_img;
  std::size_t dpitch = dst_img_pitch;
  std::size_t spitch = src_img.step;
  std::size_t width  = src_img.cols * src_img.elemSize();
  std::size_t height = src_img.rows;
  // [SLAM zero-to-hero] The size comes from the Mat, not from the destination Subframe. With
  // async = false, cudaMemcpy2D runs in the legacy default stream and blocks the CPU (from
  // pageable memory the CUDA docs only promise that the data has been staged when it returns).
  // With async = true it is queued in stream_num; design notes 2 and 3 say what that means for
  // pageable and for pinned memory.
  if(async) {
    CUDA_API_CALL(cudaMemcpy2DAsync(dst,dpitch,
                                    src,spitch,
                                    width,height,
                                    dir,stream_num));
  } else {
    CUDA_API_CALL(cudaMemcpy2D(dst,dpitch,
                               src,spitch,
                               width,height,
                               dir));
  }
}

void opencv_copy_from_image_to_gpu(const cv::Mat & h_src_img,
                                   unsigned char * d_dst_img,
                                   unsigned int dst_img_pitch,
                                   bool async,
                                   cudaStream_t stream_num) {
  opencv_copy_from_image_common(h_src_img,
                                d_dst_img,
                                dst_img_pitch,
                                async,
                                stream_num,
                                cudaMemcpyHostToDevice);
}

void opencv_copy_from_image_to_host(const cv::Mat & h_src_img,
                                    unsigned char * h_dst_img,
                                    unsigned int dst_img_pitch) {
  opencv_copy_from_image_common(h_src_img,
                                h_dst_img,
                                dst_img_pitch,
                                false,
                                0,
                                cudaMemcpyHostToHost);
}

void opencv_copy_from_host_to_image(cv::Mat & h_dst_img,
                                    const unsigned char * h_src_img,
                                    unsigned int src_img_width,
                                    unsigned int src_img_height,
                                    unsigned int src_img_element_size,
                                    unsigned int src_img_pitch) {
  opencv_copy_to_image_common(h_dst_img,
                              h_src_img,
                              src_img_width,
                              src_img_height,
                              src_img_element_size,
                              src_img_pitch,
                              false,
                              0,
                              cudaMemcpyHostToHost);
}

void opencv_copy_from_gpu_to_image(cv::Mat & dst_img,
                                   const unsigned char * d_src_img,
                                   unsigned int src_img_width,
                                   unsigned int src_img_height,
                                   unsigned int src_img_element_size,
                                   unsigned int src_img_pitch,
                                   bool async,
                                   cudaStream_t stream_num) {
  opencv_copy_to_image_common(dst_img,
                              d_src_img,
                              src_img_width,
                              src_img_height,
                              src_img_element_size,
                              src_img_pitch,
                              async,
                              stream_num,
                              cudaMemcpyDeviceToHost);
}

// [SLAM zero-to-hero] cudaHostRegister page-locks an existing buffer in place, so a Mat you
// already have uploads like pinned memory without an extra host copy. Registering is a slow OS
// call: do it once per long-lived buffer (e.g. a ring of capture buffers), never per frame, and
// keep the Mat from reallocating while it is registered. vilib itself never calls it. On newer
// Jetsons, cudaDevAttrHostRegisterSupported tells whether the call works.
void opencv_register_image_in_pinned_memory(const cv::Mat & h_img) {
  /*
   * Note to future self and others:
   * on ARM platforms (like the Jetson TX2) the caching behaviour of memory areas
   * cannot be changed on the fly, hence this call does not work and returns:
   * "operation not supported".
   */
  CUDA_API_CALL(cudaHostRegister(h_img.data,
                                 h_img.total()*h_img.elemSize(),
                                 cudaHostRegisterPortable));
}

void opencv_unregister_image_from_pinned_memory(const cv::Mat & h_img) {
  CUDA_API_CALL(cudaHostUnregister(h_img.data));
}

} // namespace vilib
