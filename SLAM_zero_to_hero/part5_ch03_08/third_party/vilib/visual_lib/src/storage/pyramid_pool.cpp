/*
 * Class for handling the allocation of entire image pyramids efficiently
 * pyramid_pool.cpp
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
// PyramidPool hands out whole image pyramids, one Subframe per level, from a process-wide pool;
// SubframePool (subframe_pool.cpp) keeps one free list per level. Why a pool:
//  1. Device allocation is expensive. cudaMallocPitch is a driver call with an unpredictable cost
//     that can exceed the whole 18.5 us pyramid, and cudaFree waits until all work on the device
//     has finished, so freeing per frame would stall the CPU and drain the GPU's queue each frame.
//  2. So the buffers are allocated up front, by init() (or by the first get()), and recycled:
//     get() pops one Subframe per level, release() pushes them back. Per frame that costs a mutex
//     and a list operation per level, and no CUDA call.
//  3. A Frame owns its pyramid through shared_ptr<Subframe> and calls release() in its destructor.
//     A frame lives as long as a track holds it, so several pyramids are in use at once; vilib
//     preallocates IMAGE_PYRAMID_PREALLOCATION_ITEM_NUM = 10 (config.h). If a level runs out,
//     get_subframe() allocates a new Subframe (a slow frame) that the pool then keeps: the pool
//     grows to the peak number in use and never shrinks until deinit().
//  4. All pyramids share widths, heights and pitches, so one pyramid_descriptor_t (desc_)
//     describes them all. The tracker passes it to its kernels by value, next to each frame's data
//     pointers (Frame::getPyramidDescriptor). This assumes cudaMallocPitch returns the same pitch
//     for every buffer of a level; SubframePool records only the first one.
//  5. The pool does not track GPU work. A released buffer can be handed out again while kernels
//     queued earlier still read it, which is safe only because vilib issues everything in stream
//     order (one stream, or default streams that synchronize with each other). With streams of
//     your own, record an event when a frame is done and wait for it before reusing its buffers.
// Today the stream-ordered allocator (cudaMallocAsync / cudaFreeAsync, CUDA 11.2+) does this kind
// of pooling in the driver, and it is stream-aware. It was not an option on vilib's TX2, whose
// last JetPack (4.6) ships CUDA 10.2.
#include <assert.h>
#include "vilib/storage/pyramid_pool.h"
#include "vilib/storage/subframe.h"
#include "vilib/storage/subframe_pool.h"

namespace vilib {

bool PyramidPool::initialized_ = false;
std::vector<std::shared_ptr<SubframePool>> PyramidPool::pool_;
pyramid_descriptor_t PyramidPool::desc_;

void PyramidPool::init(std::size_t preallocated_item_num,
                       std::size_t original_width,
                       std::size_t original_height,
                       std::size_t data_bytes,
                       std::size_t levels,
                       Subframe::MemoryType memory_type) {
  /*
   * Note to future self:
   * make sure that the original image width & height are divisible by 2
   * for all levels.
   */
  // [SLAM zero-to-hero] -DNDEBUG compiles these asserts out. A size that does not divide still
  // gives floor(w / 2^l) per level, the sizes the pyramid kernels produce. Nothing checks that
  // levels <= MAX_IMAGE_PYRAMID_LEVELS (5, config.h); more would overflow desc_'s arrays.
  assert(levels > 0);
  assert((original_width %(1<<(levels-1))) == 0);
  assert((original_height%(1<<(levels-1))) == 0);
  for(unsigned int l=0;l<levels;++l) {
    std::size_t level_width  = original_width >> l;
    std::size_t level_height = original_height >> l;
    pool_.push_back(std::shared_ptr<SubframePool>(
                     new SubframePool(preallocated_item_num,
                                      level_width,
                                      level_height,
                                      data_bytes,
                                      memory_type)));
    // get description info
    desc_.w[l] = pool_[l]->get_width();
    desc_.h[l] = pool_[l]->get_height();
    desc_.p[l] = pool_[l]->get_pitch();
  }
  desc_.l = levels;
  initialized_ = true;
}

void PyramidPool::deinit(void) {
  pool_.clear();
  initialized_ = false;
}

void PyramidPool::get(std::size_t preallocated_item_num,
                      std::size_t original_width,
                      std::size_t original_height,
                      std::size_t data_bytes,
                      std::size_t levels,
                      Subframe::MemoryType memory_type,
                      std::vector<std::shared_ptr<Subframe>> & pyramid) {
  /*
   * Note to future self:
   * 0) Initialize the subframe pools if necessary
   *    Hence, the first frame processing might take longer than the rest
   * 1) Give back subframes for all pyramid levels
   */
  // [SLAM zero-to-hero] Pitfall: the pool is static, sized by whichever call comes first. A later
  // get() is not checked: another size or memory type receives buffers of the first kind, and a
  // larger level count indexes pool_ past its end. `pyramid` is appended to, not cleared.
  if(!initialized_) {
    init(preallocated_item_num,
         original_width,
         original_height,
         data_bytes,
         levels,
         memory_type);
  }
  for(std::size_t l=0;l<levels;++l) {
    pyramid.push_back(pool_[l]->get_subframe());
  }
}

void PyramidPool::release(std::vector<std::shared_ptr<Subframe>> & pyramid) {
  if(!initialized_) return;
  // [SLAM zero-to-hero] Returns pool_.size() levels, so `pyramid` must have exactly as many levels
  // as the pool. No wait for GPU work here (design note 5).
  for(std::size_t l=0;l<pool_.size();++l) {
    pool_[l]->return_subframe(pyramid[l]);
  }
}

} // namespace vilib
