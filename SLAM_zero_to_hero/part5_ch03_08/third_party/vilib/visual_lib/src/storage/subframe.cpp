/*
 * Class for holding a particular representation of an input image
 * subframe.cpp
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
// A Subframe is one image buffer plus its geometry: width_, height_, and pitch_, the distance in
// bytes from one row to the next. One class covers five kinds of memory, chosen per buffer.
// vilib's pyramids use PITCHED_DEVICE_MEMORY (IMAGE_PYRAMID_MEMORY_TYPE in config.h).
//  1. PAGED_HOST_MEMORY (malloc): ordinary pageable memory. The OS may move or swap its pages, so
//     the GPU's copy engine cannot read it directly; the driver stages every copy to the GPU
//     through a pinned buffer of its own (see opencv.cpp).
//  2. PINNED_HOST_MEMORY (cudaMallocHost): page-locked, so the copy engine reads it directly and
//     an async copy from it really is asynchronous. Allocating it is slow (the OS has to lock the
//     pages), and pinning a lot of memory takes it away from the rest of the system.
//  3. UNIFIED_MEMORY (cudaMallocManaged): one pointer valid on the CPU and the GPU. On a desktop
//     GPU the driver migrates the pages to whichever side touches them. On a Jetson both use the
//     same DRAM, so this kind of buffer (or mapped pinned memory) is how a frame reaches the GPU
//     without an upload copy. vilib does not use it for its pyramids.
//  4. LINEAR_DEVICE_MEMORY (cudaMalloc): device memory with packed rows, pitch = width.
//  5. PITCHED_DEVICE_MEMORY (cudaMallocPitch): each row is padded to a pitch the driver picks, so
//     every row starts on an aligned address. Warps reading along a row then touch the fewest
//     32-byte sectors, and the pyramid kernel's uchar2 / uchar4 accesses are aligned on every row.
//     The price: the rows are no longer contiguous, so every copy from or to the host is a 2D copy
//     (cudaMemcpy2D*), and for EuRoC's 752-byte rows that copy is slow (README, "The upload path").
// Allocating and freeing are expensive driver calls, and cudaFree also waits until all work on
// the device has finished. So Subframes are made up front and recycled through SubframePool and
// PyramidPool; a new one is allocated in the frame loop only when the pool runs dry
// (subframe_pool.cpp). None of the allocation calls below is error-checked.
// copy_from() and copy_to() pick a copy by memory type; the cv::Mat versions end in opencv.cpp, a
// .cpp file, where a stream of 0 is the legacy default stream (README, "Two default streams").
#include <stdlib.h>
#include <string>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#ifdef ROS_SUPPORT
#include <sensor_msgs/image_encodings.h>
#endif /* ROS_SUPPORT */
#include "vilib/storage/subframe.h"
#include "vilib/storage/subframe_pool.h"
#include "vilib/storage/opencv.h"
#include "vilib/storage/ros.h"

namespace vilib {

Subframe::Subframe(std::size_t width,
                   std::size_t height,
                   std::size_t data_bytes,
                   MemoryType type) :
  width_(width),height_(height),data_bytes_(data_bytes),type_(type) {
  // perform the memory allocations
  switch(type) {
    case MemoryType::PAGED_HOST_MEMORY: {
      total_bytes_ = width*data_bytes*height; // packed: width * height * data_bytes
      data_ = (unsigned char *)malloc(total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::PINNED_HOST_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMallocHost((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY: {
      total_bytes_ = width*height*data_bytes; // packed: width * height * data_bytes
      cudaMalloc((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      // [SLAM zero-to-hero] pitch_ comes back in bytes, >= width * data_bytes. The pools assume it
      // is the same for every buffer of one width (see pyramid_pool.cpp). Print it to see what
      // your GPU picks for 752. Address pixel (x, y) as data_ + y * pitch_ + x * data_bytes_.
      cudaMallocPitch((void**)&data_,&pitch_,width*data_bytes,height);
      /*
       * Note to future self:
       * the returned pitch will be the calculated pitch in byte units
       */
      total_bytes_ = pitch_*height;
      break;
    }
    case MemoryType::UNIFIED_MEMORY: {
      total_bytes_ = width*data_bytes*height; // packed: width * height * data_bytes
      cudaMallocManaged((void**)&data_,total_bytes_);
      pitch_ = width*data_bytes;
      break;
    }
  }
}

Subframe::~Subframe(void) {
  // perform the memory deallocations
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY: {
      free(data_);
      break;
    }
    case MemoryType::PINNED_HOST_MEMORY: {
      cudaFreeHost(data_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      // [SLAM zero-to-hero] cudaFree first waits for all queued work on the device, so a kernel
      // still reading this buffer is safe, but the CPU stalls: one more reason not to free per
      // frame.
      cudaFree(data_);
      break;
    }
  }
}

void Subframe::copy_from(const cv::Mat & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO : support color!
  assert(h_img.channels() == 1);
  // [SLAM zero-to-hero] Host-side types get a plain synchronous host copy (async is ignored). The
  // device types get the upload in opencv.cpp: one cudaMemcpy2D(Async) from the cv::Mat into
  // data_ with this Subframe's pitch_. Frame uploads level 0 of every new pyramid this way.
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_image_to_host(h_img,
                                     data_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_image_to_gpu(h_img,
                                    data_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

#ifdef ROS_SUPPORT
void Subframe::copy_from(const sensor_msgs::ImageConstPtr & h_img,
                         bool async,
                         cudaStream_t stream_num) {
  // TODO: support color
  assert(h_img->encoding == sensor_msgs::image_encodings::MONO8);
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      ros_copy_from_image_to_host(h_img,
                                  1,
                                  data_,
                                  pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      ros_copy_from_image_to_gpu(h_img,
                                 1,
                                 data_,
                                 pitch_,
                                 async,
                                 stream_num);
      break;
    }
  }
}
#endif /* ROS_SUPPORT */

void Subframe::copy_to(cv::Mat & h_img,
                       bool async,
                       cudaStream_t stream_num) const {
  switch(type_) {
    case MemoryType::PAGED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::PINNED_HOST_MEMORY:
      /* fall-through */
    case MemoryType::UNIFIED_MEMORY: {
      opencv_copy_from_host_to_image(h_img,
                                     data_,
                                     width_,
                                     height_,
                                     data_bytes_,
                                     pitch_);
      break;
    }
    case MemoryType::LINEAR_DEVICE_MEMORY:
      /* fall-through */
    case MemoryType::PITCHED_DEVICE_MEMORY: {
      opencv_copy_from_gpu_to_image(h_img,
                                    data_,
                                    width_,
                                    height_,
                                    data_bytes_,
                                    pitch_,
                                    async,
                                    stream_num);
      break;
    }
  }
}

void Subframe::display(void) const {
  // copy image to a temporary buffer and display that
  std::string subframe_title("Subframe (");
  subframe_title += std::to_string(width_);
  subframe_title += "x";
  subframe_title += std::to_string(height_);
  subframe_title += ")";
  cv::Mat image;
  copy_to(image);
  cv::imshow(subframe_title.c_str(), image);
  cv::waitKey();
}

} // namespace vilib
