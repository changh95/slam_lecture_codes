/*
 * Feature tracker utilities in CUDA
 * feature_tracker_cuda_tools.h
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

#pragma once

#include "vilib/common/types.h"

namespace vilib {
namespace feature_tracker_cuda_tools {

// [SLAM zero-to-hero] Both functions only enqueue a kernel on 'stream' and return; the caller
// synchronizes. Conventions the .cu relies on:
//  - d_indir_data[cx] is the slot of candidate cx (cx = 0 .. candidate_num - 1).
//  - d_in_ref_px, d_in_first_px, d_in_cur_px, d_in_cur_alpha_beta, d_in_cur_f and
//    d_in_cur_disparity point at fields of slot 0's 64-byte record (mapped host memory); slot bx
//    is bx * 64 bytes further.
//  - d_patch_data is passed as bytes and cast to FEATURE_TRACKER_REFERENCE_PATCH_TYPE (int) in
//    the .cu; it and d_hessian_data are device memory, one fixed block per slot.
//  - The descriptors are taken by reference here and passed by value to the kernels.
//  - The two bools pick a template instantiation; they are not passed to the kernels.
void update_tracks(const int candidate_num,
                   const bool affine_est_offset,
                   const bool affine_est_gain,
                   const int min_level,
                   const int max_level,
                   const image_pyramid_descriptor_t & pyramid_description,
                   const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                   const int * d_indir_data,
                   const float2 * d_in_ref_px,
                   unsigned char * d_patch_data,
                   float * d_hessian_data,
                   cudaStream_t stream);

void track_features(const bool affine_est_offset,
                    const bool affine_est_gain,
                    const int candidate_num,
                    const int min_level,
                    const int max_level,
                    const float min_update_squared,
                    const image_pyramid_descriptor_t & pyramid_description,
                    const pyramid_patch_descriptor_t & pyramid_patch_sizes,
                    const int * d_indir_data,
                    const unsigned char * d_patch_data,
                    const float * d_hessian_data,
                    const float2 * d_in_first_px,
                    float2 * d_in_cur_px,
                    float2 * d_in_cur_alpha_beta,
                    float4 * d_in_cur_f,
                    float  * d_in_cur_disparity,
                    cudaStream_t stream);

} // namespace feature_tracker_cuda_tools
} // namespace vilib
