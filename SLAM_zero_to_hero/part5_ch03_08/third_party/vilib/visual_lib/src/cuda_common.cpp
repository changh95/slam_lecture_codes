/*
 * Common CUDA functionalities
 * cuda_common.cpp
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

#include <iostream>
#include "vilib/cuda_common.h"

namespace vilib {

bool cuda_initialize(void) {
  /*
   * Note to future self and others:
   * there's no explicit initialization of the CUDA runtime, hence when
   * the first cudaX...() call is made, the CUDA runtime gets initialized.
   * This behaviour can skew benchmark measurements.
   */
  int device_count;
  if(cudaGetDeviceCount(&device_count) != cudaSuccess) {
    std::cout << "Error: no CUDA-capable device detected" << std::endl;
    return false;
  }

  // [SLAM zero-to-hero] Spin: a host thread waiting in cudaDeviceSynchronize() and the like
  // busy-polls the GPU instead of sleeping. That gives the shortest wake-up after short kernels,
  // at the cost of a CPU core at 100% while it waits (on a Jetson, a core and power the rest of
  // the VIO pipeline could use). The exercise programs do not call cuda_initialize(). They keep
  // the default, cudaDeviceScheduleAuto, which also spins as long as the process has no more
  // active CUDA contexts than logical CPUs (true on the course machine) and yields otherwise.
  CUDA_API_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
  // .. we need to do something more meaningful to initialize the Runtime API..
  // [SLAM zero-to-hero] The first real CUDA call creates the context, which is slow. Doing it here
  // keeps that out of the first timed call; the exercise does the same with cudaFree(nullptr)
  // in cuda_bench::initGpu().
  unsigned char * temp;
  CUDA_API_CALL(cudaMalloc((void**)&temp,1));
  CUDA_API_CALL(cudaFree(temp));
  return true;
}

// [SLAM zero-to-hero] The grid is rounded up to whole blocks, so the last block in x and in y can
// hold threads outside the image: every kernel launched with these parameters must check its
// coordinates (pyramid_gpu.cu does: x < width && y < height).
kernel_params_t cuda_gen_kernel_params_2d(unsigned int width,
                                          unsigned int height,
                                          unsigned int tpb_x,
                                          unsigned int tpb_y) {
  assert(width > 0 && height > 0);
  kernel_params_t p;
  p.threads_per_block.x = tpb_x;
  p.threads_per_block.y = tpb_y;
  p.threads_per_block.z = 1;
  p.blocks_per_grid.x = (width + tpb_x - 1) / tpb_x;
  p.blocks_per_grid.y = (height + tpb_y - 1) / tpb_y;
  p.blocks_per_grid.z = 1;
  return p;
}

kernel_params_t cuda_gen_kernel_params_1d(unsigned int n,
                                          unsigned int tpb) {
 assert(n > 0);
 kernel_params_t p;
 p.threads_per_block.x = tpb;
 p.threads_per_block.y = 1;
 p.threads_per_block.z = 1;
 p.blocks_per_grid.x = (n + tpb-1)/tpb;
 p.blocks_per_grid.y = 1;
 p.blocks_per_grid.z = 1;
 return p;
}

} // namespace vilib
