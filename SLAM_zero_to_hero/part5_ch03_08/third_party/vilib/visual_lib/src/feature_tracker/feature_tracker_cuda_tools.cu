/*
 * CUDA kernels for the feature tracker
 * feature_tracker_cuda_tools.cu
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
// Pyramidal inverse compositional Lucas-Kanade (KLT), split into two kernels:
//  - update_tracks_kernel runs when a template is made: for each new feature, or for every track
//    on every frame if klt_template_is_first_observation is false. On each pyramid level it cuts
//    the template patch (plus a 1-pixel border for the gradients) out of the current frame,
//    builds H = sum(J J^T) from the template's gradients, inverts it, and stores patch and H^-1.
//  - track_features_kernel runs every frame for every live track: coarse to fine, Gauss-Newton
//    iterations that sample the new frame under the patch and update the position (and the
//    optional gain/offset) with the stored H^-1.
// Thread mapping: one warp (32 lanes) per feature. Blocks are (32, 2) for tracking and (32, 3)
// for templates; threadIdx.y picks the feature, threadIdx.x the lane. A 16x16 patch gives each
// lane 8 pixels (the warp covers 2 rows per step), an 8x8 patch 2 pixels (4 rows per step).
// Memory spaces:
//  - Image pyramid: pitched device memory (PyramidPool), read with plain byte loads. There are
//    no texture objects and no shared memory anywhere in this file.
//  - Template patches (one int per pixel) and H^-1 (10 floats per level): cudaMalloc'd device
//    memory, one fixed slot per feature, written by update_tracks_kernel, then only read.
//  - Per-feature state (start/result position, first position, gain/offset, disparity) and the
//    candidate -> slot list: mapped (zero-copy) pinned host memory, see feature_tracker_gpu.cpp.
//  - Pyramid and patch-size descriptors: kernel arguments passed by value, so they sit in the
//    constant parameter bank; all lanes read the same entry, which is a broadcast.
//  - Registers: the partial sums (Jres, H), the position and the gain/offset.
// Techniques:
//  1. Inverse compositional: J and H depend only on the template, so H^-1 is computed once per
//     template and level. An iteration is then "sample, residual, accumulate J*r, multiply by
//     H^-1". (The forward additive form would rebuild and invert H in every iteration.)
//  2. One warp per feature, reduced with __shfl_xor_sync: a 5-step butterfly in registers leaves
//     the same sum in every lane, so all lanes compute the same update and take the same
//     branches, with no shared memory and no __syncthreads().
//  3. Compile-time specialization: the gain/offset flags are template parameters, so each of
//     the four instantiations contains only its own arithmetic; the host picks one per launch.
//  4. NaN as an in-band flag: a level whose template did not fit gets a NaN in H^-1[0], a lost
//     track gets a NaN position. No separate status arrays.
// vilib's nvcc flags (CMakeLists.txt) include --ftz=true --prec-div=false --fmad=true, so the
// arithmetic is not bit-identical to a CPU loop. Together with the butterfly's summation order,
// this is likely why klt_reference.hpp lands within 0.008 px of the GPU (README), not exactly.
#include "vilib/feature_tracker/feature_tracker_cuda_tools.h"
#include "vilib/feature_tracker/config.h"
#include "vilib/cuda_common.h"

namespace vilib {
namespace feature_tracker_cuda_tools {

// Warp preliminaries
#define WARP_SIZE                       32
#define WARP_MASK                       0xFFFFFFFF
// Precalculating feature patches
#define CANDIDATES_PER_BLOCK_UPDATE     3
/*
 * Note to future self:
 * - interestingly with reference patch interpolation, the tracking performance
 *   degrades. -> DISABLE
 * - Int/Float does not seem to affect the tracking performance, but the float
 *   version is slower. -> INT
 */
// [SLAM zero-to-hero] 0: load_ref_patch cuts the template at integer pixels, around floor() of
// the position on each level. FAST positions are integers on level 0, so a first-observation
// template loses nothing there; on a coarser level l the position x / 2^l is truncated, and the
// finer levels correct that. With klt_template_is_first_observation = false the template is
// re-cut every frame at the latest sub-pixel position, its fraction is dropped each time, and
// the tracks drift up and to the left (README, "Previous-frame templates drift": -0.5 px in x
// and y on average against OpenCV's frame-to-frame KLT). "Int/Float" in the note above is the
// template's element type, FEATURE_TRACKER_REFERENCE_PATCH_TYPE (config.h).
#define REFERENCE_PATCH_INTERPOLATION   0
// [SLAM zero-to-hero] Features per block: 2 warps (64 threads) for tracking, 3 (96 threads) for
// the templates (CANDIDATES_PER_BLOCK_UPDATE above). The warps of a block share nothing, so
// grouping them only makes the blocks bigger. Likely why: an SM holds at most 32 resident
// blocks on Pascal/Volta (vilib's Jetson TX2/Xavier), so 32-thread blocks would cap it at 32 of
// its 64 warps. With ~141 tracked features (README) the tracking grid is 71 blocks, fewer than
// the RTX 5090's 170 SMs: most SMs run one block or none, and the kernel time is the latency of
// the slowest feature, not throughput. On the TX2's 2 SMs the same grid is ~36 blocks per SM,
// more than can be resident at once.
#define CANDIDATES_PER_BLOCK_TRACK      2

// [SLAM zero-to-hero] One warp aligns one feature on one pyramid level. Every lane gets the same
// arguments and keeps its own copy of cur_px / cur_alpha_beta. Because the reduced sums below
// are identical in all lanes, these copies stay identical too, and every branch on them (NaN,
// image border, convergence) is taken by the whole warp together. T is the template's element
// type (int, config.h); the two bools are compile-time, so the if(affine_...) tests below cost
// nothing at run time.
template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__device__ __inline__ void perform_lk(const float & min_update_squared,
                                      const int & img_width,
                                      const int & img_height,
                                      const int & img_pitch,
                                      const unsigned char * __restrict__ d_in_cur_img,
                                      const int & patch_size,
                                      const int & half_patch_size,
                                      const int & patch_stride,
                                      const T * ref_patch,
                                      const float * invH,
                                      float2 & cur_px,
                                      float2 & cur_alpha_beta,
                                      bool & converged,
                                      bool & go_to_next_level) {
  converged = false;
  go_to_next_level = false;

  // [SLAM zero-to-hero] Lane -> pixel mapping, computed once per level, outside the iterations.
  // Lane t starts at patch pixel (x, y) = (t % patch_size, t / patch_size), so the warp covers
  // 32 / patch_size whole rows per step: for a 16x16 patch lanes 0-15 take row y and lanes 16-31
  // row y+1. Each step moves down that many rows (it_offset in the image, it_ref_offset in the
  // template, whose rows are patch_size + 2 long). This needs 32 % patch_size == 0 and
  // patch_area >= 32, i.e. patch_size 8, 16 or 32 (see klt_patch_sizes in the options).
  // it_start is the lane's pixel (x - half, y - half) for a patch centred on pixel (0, 0); per
  // iteration only "+ u_r + v_r*img_pitch" moves it to the current position. patch_size comes
  // from a kernel argument, so pixels_per_thread is not a compile-time constant and the
  // "#pragma unroll" on the pixel loop cannot unroll it completely.
  // Reference patch & actual image
  const int patch_area = patch_size * patch_size;
  const int x = threadIdx.x % patch_size;
  const int y = threadIdx.x / patch_size;
  const T * it_ref_start = ref_patch + (y+1)*patch_stride + (x+1);
  const int it_ref_offset = patch_stride*WARP_SIZE/patch_size;
  const unsigned char * it_start = (const unsigned char*) d_in_cur_img -
                             (img_pitch + 1)*half_patch_size + threadIdx.x +
                             (img_pitch - patch_size)*(threadIdx.x/patch_size);
  const int it_offset = img_pitch*WARP_SIZE/patch_size;
  const int pixels_per_thread = patch_area/WARP_SIZE;

  // [SLAM zero-to-hero] At most FEATURE_TRACKER_MAX_ITERATION_COUNT (30, config.h) Gauss-Newton
  // steps per level. It is a macro, not the klt_max_iter option (which nothing reads), so the
  // trip count is a compile-time constant and "#pragma unroll" can unroll this loop completely.
  #pragma unroll
  for(int iter=0; iter<FEATURE_TRACKER_MAX_ITERATION_COUNT;++iter) {
    if(isnan(cur_px.x) || isnan(cur_px.y)) {
      break;
    }
    // [SLAM zero-to-hero] The bilinear taps reach from u_r - half to u_r + half (and the same in
    // v), so this one test keeps every read inside the image, with no per-pixel checks. On a
    // coarse level, leaving the image is not a failure: the next finer level gets a try. On the
    // finest level (min_level) there is none: converged was set to false on entry and stays
    // false, so the track is lost. This is how features that move out of the frame end.
    int u_r = floorf(cur_px.x);
    int v_r = floorf(cur_px.y);
    if(u_r < half_patch_size ||
       v_r < half_patch_size ||
       u_r >= (img_width-half_patch_size) ||
       v_r >= (img_height-half_patch_size)) {
      // don't change the state 'converged'
      go_to_next_level = true;
      break;
    }

    // [SLAM zero-to-hero] Bilinear interpolation in software. The warp is a pure translation, so
    // all pixels of the patch share one sub-pixel offset and the 4 weights are computed once per
    // iteration; each sample is then 4 byte loads, 4 int-to-float conversions and 4 multiply-adds.
    // Likely why not a texture with cudaFilterModeLinear: the texture unit keeps the weights in
    // 9-bit fixed point (8 fractional bits), which would quantize the sub-pixel position that the
    // solver is refining.
    // compute interpolation weights
    float subpix_x = cur_px.x-u_r;
    float subpix_y = cur_px.y-v_r;
    float wTL = (1.0f-subpix_x)*(1.0f-subpix_y);
    float wTR = subpix_x * (1.0f-subpix_y);
    float wBL = (1.0f-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    float Jres[4];
    #pragma unroll
    for(int i=0;i<4;++i) {
      Jres[i] = 0.0f;
    }

    const uint8_t * it = it_start + u_r + v_r*img_pitch;
    const T * it_ref = it_ref_start;

    // [SLAM zero-to-hero] Per pixel: one bilinear sample of the new image, the residual against
    // the (gain/offset-corrected) template, and J*r with J = the template's central-difference
    // gradient. J is recomputed from the stored patch in every iteration instead of being stored:
    // 4 extra int loads from cache lines the warp touches anyway, against 2 more floats per pixel
    // and level in device memory. The template is re-read from global memory in every iteration
    // (the TODO further down suggests shared memory); it is 1.3 KB per 16x16 level, so after the
    // first iteration these reads are served from cache.
    // Note: every thread computes (PATCH_SIZE*PATCH_SIZE/WARP_SIZE) pixels
    #pragma unroll
    for(int i=0;i<pixels_per_thread;++i,it+=it_offset,it_ref+=it_ref_offset) {
      // Note it cannot be read as uchar2, because it would require proper alignment
      // [SLAM zero-to-hero] Each of these 4 loads is one instruction for the whole warp: for a
      // 16x16 patch, 16 lanes read 16 consecutive bytes of one image row and the other 16 the
      // next row, i.e. 2-4 32-byte sectors per load, and the 4 taps largely overlap.
      float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[img_pitch] + wBR*it[img_pitch+1];
      float res = search_pixel - (1.0f+cur_alpha_beta.x)*(*it_ref) - cur_alpha_beta.y;
      Jres[0] += res * 0.5f * (it_ref[1] - it_ref[-1]);
      Jres[1] += res * 0.5f * (it_ref[patch_stride] - it_ref[-patch_stride]);

      // [SLAM zero-to-hero] Gain/offset model: I(x + p) ~ (1 + alpha) * T(x) + beta, with
      // cur_alpha_beta = (alpha, beta). Their Jacobians are T and 1, also template-only, so they
      // join the precomputed 3x3/4x4 H. These ifs test template parameters: the compiler keeps
      // one branch and drops the others.
      // If affine compensation is used,
      // set Jres with respect to affine parameters.
      if(affine_est_offset && affine_est_gain) {
        Jres[2] += res;
        Jres[3] += res*(*it_ref);
      } else if(affine_est_offset) {
        Jres[2] += res;
      } else if(affine_est_gain) {
        Jres[2] += res*(*it_ref);
      }
    }

    // [SLAM zero-to-hero] Butterfly reduction: in each step a lane adds the value of the lane
    // whose index differs in one bit (offset 16, 8, 4, 2, 1). After 5 steps every lane holds the
    // sum of all 32. Partners add the same two numbers and a + b == b + a exactly, so (by
    // induction over the steps) all lanes end with bit-identical sums. That matters: the update,
    // the NaN/border tests and the convergence break must be warp-uniform, because the full-mask
    // (WARP_MASK) shuffle in the next iteration needs all 32 lanes to arrive. Cost: 5 x 4
    // shuffles per iteration, register to register. __shfl_down_sync would leave the sum in lane
    // 0 only (calc_hessian uses that). Without gain/offset, Jres[2..3] stay 0 but are still in
    // this loop; whether nvcc drops their shuffles can be checked in the SASS.
    // Reduce it to all lanes
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
      #pragma unroll
      for(int i=0;i<4;++i) {
        Jres[i] += __shfl_xor_sync(WARP_MASK, Jres[i], offset);
      }
    }

    // [SLAM zero-to-hero] Every lane computes the update itself from its identical copy of the
    // sums; there is no broadcast from lane 0, despite the upstream comment below. invH is in
    // global memory and all 32 lanes load the same address: one transaction, broadcast to the
    // warp. Only the upper triangle of the symmetric H^-1 is stored (3, 6 or 10 floats).
    //update = Hinv * Jres
    //broadcast the computed values in the warp from lane 0
    float update[4];
    if(affine_est_offset && affine_est_gain) {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1] + invH[2] * Jres[2] + invH[3] * Jres[3];
      update[1] = invH[1] * Jres[0] + invH[4] * Jres[1] + invH[5] * Jres[2] + invH[6] * Jres[3];
      update[2] = invH[2] * Jres[0] + invH[5] * Jres[1] + invH[7] * Jres[2] + invH[8] * Jres[3];
      update[3] = invH[3] * Jres[0] + invH[6] * Jres[1] + invH[8] * Jres[2] + invH[9] * Jres[3];
    } else if(affine_est_gain || affine_est_offset) {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1] + invH[2] * Jres[2];
      update[1] = invH[1] * Jres[0] + invH[3] * Jres[1] + invH[4] * Jres[2];
      update[2] = invH[2] * Jres[0] + invH[4] * Jres[1] + invH[5] * Jres[2];
    } else {
      update[0] = invH[0] * Jres[0] + invH[1] * Jres[1];
      update[1] = invH[1] * Jres[0] + invH[2] * Jres[1];
    }

    // [SLAM zero-to-hero] Inverse compositional update for a translation: the step d was solved
    // for the template (T(x + d) ~ T + J d), so the image position moves by -d. Gain and offset
    // get plain additive Gauss-Newton steps. H ignores the (1 + alpha) factor on the template's
    // gradient, which is what keeps it constant.
    // Updating our estimates
    // Translational displacement
    cur_px.x -= update[0];
    cur_px.y -= update[1];

    // Illumination changes
    if(affine_est_offset && affine_est_gain) {
      cur_alpha_beta.x += update[3];
      cur_alpha_beta.y += update[2];
    } else if (affine_est_offset) {
      cur_alpha_beta.y += update[2];
    } else if(affine_est_gain) {
      cur_alpha_beta.x += update[2];
    }

    // [SLAM zero-to-hero] Converged when the step is below sqrt(klt_min_update_squared), measured
    // in this level's pixels, so the threshold is 2^level times coarser in full-resolution
    // pixels. Gain/offset steps do not count. No cost is evaluated: there is no step-size control.
    if(update[0]*update[0]+update[1]*update[1] < min_update_squared) {
      converged=true;
      break;
    }
  }
}

// [SLAM zero-to-hero] One warp per live track: blockIdx.x * 2 + threadIdx.y is the candidate
// index cx (the track's position in tracks_ on the host), and d_indir_data[cx] its slot bx in
// the per-feature buffers. The cx < candidate_num test depends only on threadIdx.y, so a warp is
// either fully in or fully out (an odd count leaves the last block's second warp idle).
template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__global__ void track_features_kernel(const int candidate_num,
                                      const int min_level,
                                      const int max_level,
                                      const float min_update_squared,
                                      const image_pyramid_descriptor_t pyramid_description,
                                      const pyramid_patch_descriptor_t  pyramid_patch_sizes,
                                      const int * __restrict__ d_indir_data,
                                      const T * __restrict__ d_patch_data,
                                      const float * __restrict__ d_hessian_data,
                                      const float2 * __restrict__ d_first_px,
                                      float2 * __restrict__ d_cur_px,
                                      float2 * __restrict__ d_cur_alpha_beta,
                                      float4 * __restrict__ d_cur_f,
                                      float  * __restrict__ d_cur_disparity) {
  const int cx = blockIdx.x * CANDIDATES_PER_BLOCK_TRACK + threadIdx.y; // candidate id
  const int pyramid_levels = max_level - min_level + 1; // number of pyramid levels computed
  if(cx < candidate_num) {
    // [SLAM zero-to-hero] d_indir_data and the 64-byte records below are mapped host memory. On a
    // desktop each read is a round trip over PCIe, and the record's address depends on bx, so
    // every warp starts with two dependent PCIe reads. All warps wait at the same time, so the
    // cost is paid about once per launch, not once per feature. On a Jetson it is the same DRAM.
    // Acquire buffer id for the candidate
    const int bx = d_indir_data[cx];

    // [SLAM zero-to-hero] Each pointer points at one field of slot 0's record, and slot bx is
    // 64 bytes further: bx << 3 float2s, bx << 2 float4s, bx << 4 floats. This hard-codes
    // METADATA_ELEMENT_BYTES (feature_tracker_gpu.cpp) a second time.
    // Initialize input and output references
    // Remark: struct size: 64 bytes
    // Tracking
    float2 & d_cur_px_bx = d_cur_px[bx<<3]; // level 0
    float2 & d_cur_alpha_beta_bx = d_cur_alpha_beta[bx<<3];
    // Bearing vector
    float4 & d_cur_f_bx = d_cur_f[bx<<2];
    // Disparity calculation
    const float2 & d_first_px_bx = d_first_px[bx<<3];
    float & d_cur_disparity_bx = d_cur_disparity[bx<<4];

    // [SLAM zero-to-hero] Slot bx owns pyramid_levels patches (max_area ints each, coarsest level
    // first) and pyramid_levels H^-1 blocks (10 floats each), in plain device memory.
    // Patch data
    const T * d_patch_data_bx = d_patch_data + pyramid_patch_sizes.max_area*pyramid_levels*bx; // points to max_level
    // Hessian data
    const float * d_hessian_data_bx = d_hessian_data + 10*pyramid_levels*bx; // points to max_level

    /*
     * Iterate through all the selected levels,
     * and refine the current patch location
     */
    // [SLAM zero-to-hero] Coarse to fine. cur_px is kept in level-0 pixels between levels: it is
    // divided by 2^level before the level and multiplied back in the loop increment, which also
    // runs on 'continue'. The loop goes on while the last level converged or left the image. One
    // level that runs out of iterations (or reaches a NaN position) stops it, and the track is
    // lost. The start, cur_px, is the track's result from the previous frame.
    bool converged = false;
    bool go_to_next_level = true;
    float2 cur_px = d_cur_px_bx;
    float2 cur_alpha_beta = d_cur_alpha_beta_bx;
    float scale = 1.0f;
    for(int level=max_level;
        (converged || go_to_next_level) && level>=min_level;
        cur_px.x *= scale,cur_px.y *= scale,--level,d_patch_data_bx+=pyramid_patch_sizes.max_area,d_hessian_data_bx+=10) {
      // scale & patch size
      scale = (float)(1<<level);
      const float inv_scale = 1.0f/scale;
      const int & patch_size = pyramid_patch_sizes.wh[level];
      const int & half_patch_size = patch_size >> 1;
      const int & patch_stride = patch_size + 2;

      // image size
      const int & d_in_img_width  = pyramid_description.desc.w[level];
      const int & d_in_img_height = pyramid_description.desc.h[level];
      const int & d_in_img_pitch  = pyramid_description.desc.p[level];

      // update the pixel positions according to the current level
      cur_px.x = cur_px.x * inv_scale;
      cur_px.y = cur_px.y * inv_scale;

      // TODO : maybe load it into shared memory later if size is small (8x8), also the inverse hessian!
      // [SLAM zero-to-hero] (TODO above) In the source, the template and H^-1 are read from global
      // memory in every iteration; shm_per_block in track_features() is 0. Nothing in the loop
      // writes memory, so nvcc may keep H^-1 in registers; the SASS shows which.
      // A NaN in H^-1[0] marks a level whose template did not fit, or whose H was singular in a
      // way that makes element 0 NaN (e.g. a flat patch, see klt_reference.hpp): skip the level,
      // keep the position. If that is the finest level and the one above converged, the track is
      // reported at that coarser level's precision. Other singular H give a NaN step and lose the
      // track. A nearly singular H (a straight edge) passes: unlike OpenCV's
      // calcOpticalFlowPyrLK (minEigThreshold), there is no minimum-eigenvalue test.
      // Check if the inverse hessian was computed for the level successfully
      if(isnan(d_hessian_data_bx[0])) {
        continue;
      }

      const unsigned char * d_in_cur_img = pyramid_description.data[level];
      // do the Lukas-Kanade on the actual level, using the reference patch
      perform_lk<T, affine_est_offset, affine_est_gain>(
                                 min_update_squared,
                                 d_in_img_width,
                                 d_in_img_height,
                                 d_in_img_pitch,
                                 d_in_cur_img,
                                 patch_size,
                                 half_patch_size,
                                 patch_stride,
                                 d_patch_data_bx,
                                 d_hessian_data_bx,
                                 cur_px,
                                 cur_alpha_beta,
                                 converged,
                                 go_to_next_level);
    }

    // [SLAM zero-to-hero] All lanes hold the same result; lane 0 alone writes it. Stores from all
    // 32 lanes would also reach memory once per field (when lanes of a warp store to the same
    // address, one of them does the write), so the if makes the single writer explicit rather
    // than saving PCIe traffic. A lost track gets a NaN built from its bit
    // pattern (0x7fffffff, a quiet NaN); the host tests x with std::isnan. Gain/offset are written
    // only for the estimated terms. The bearing vector d_cur_f is never computed (TODO below).
    if(threadIdx.x == 0) {
      if(converged) {
        // Point location
        d_cur_px_bx.x = cur_px.x;
        d_cur_px_bx.y = cur_px.y;
        // Alpha-beta estimation
        if(affine_est_gain) {
          d_cur_alpha_beta_bx.x = cur_alpha_beta.x;
        }
        if(affine_est_offset) {
          d_cur_alpha_beta_bx.y = cur_alpha_beta.y;
        }
        // Bearing vector
        // TODO
        (void)d_cur_f_bx;
        // Disparity
        d_cur_disparity_bx = sqrtf((cur_px.x - d_first_px_bx.x)*(cur_px.x - d_first_px_bx.x) +
                                   (cur_px.y - d_first_px_bx.y)*(cur_px.y - d_first_px_bx.y));
      } else {
        // Initialize output to be NAN
        d_cur_px_bx.x = __int_as_float(0x7fffffff);
        d_cur_px_bx.y = __int_as_float(0x7fffffff);
      }
    }
  }
}

// [SLAM zero-to-hero] Host wrapper: picks one of the four template instantiations from the two
// runtime flags (each kernel is compiled four times) and enqueues it on 'stream'. It returns as
// soon as the launch is queued; feature_tracker_gpu.cpp waits with cudaStreamSynchronize. The
// descriptors are copied into the kernel's parameter space at launch, so the caller's structs
// may change right after.
__host__ void track_features(const bool affine_est_offset,
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
                             cudaStream_t stream) {
  // Kernel parameters
  dim3 threads_per_block;
  threads_per_block.x = WARP_SIZE;
  threads_per_block.y = CANDIDATES_PER_BLOCK_TRACK;
  threads_per_block.z = 1;
  // [SLAM zero-to-hero] Ceil division. The caller launches only with candidate_num > 0: a grid of
  // 0 blocks is an invalid launch configuration.
  const int blocks_per_grid = (candidate_num + CANDIDATES_PER_BLOCK_TRACK - 1) / CANDIDATES_PER_BLOCK_TRACK;
  const int shm_per_block = 0; //CANDIDATES_PER_BLOCK_TRACK*(PATCH_STRIDE*(PATCH_SIZE+2))*sizeof(REFERENCE_PATCH_TYPE);
  // Launch kernel
  if(affine_est_offset && affine_est_gain) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,true><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else if(affine_est_offset) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,false><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else if(affine_est_gain) {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,true><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  } else {
    track_features_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,false><<<blocks_per_grid,threads_per_block,shm_per_block,stream>>>(
      candidate_num,
      min_level,
      max_level,
      min_update_squared,
      pyramid_description,
      pyramid_patch_sizes,
      d_indir_data,
      (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
      d_hessian_data,
      d_in_first_px,
      d_in_cur_px,
      d_in_cur_alpha_beta,
      d_in_cur_f,
      d_in_cur_disparity
    );
  }
  // [SLAM zero-to-hero] cudaGetLastError() catches launch errors only; errors inside the kernel
  // show up at the next synchronization. vilib builds with -DNDEBUG, which compiles this check
  // out, and makes CUDA_API_CALL() run its call unchecked (cuda_common.h).
  CUDA_KERNEL_CHECK();
}

/* Precalculate patches & precalculate inverse Hessians  --------------------- */

// [SLAM zero-to-hero] Cuts the (patch_size + 2)^2 template, the patch plus a 1-pixel border for
// the central differences, at integer pixels: patch pixel (0,0) is (floor(x) - half,
// floor(y) - half), so the test needs half + 1 pixels left/above and half right/below. It
// depends only on ref_px, the same in all lanes, so the whole warp returns together. The 32
// lanes then fill the patch in row-major order, 32 consecutive ints per step: coalesced writes,
// reads from the image rows those ids span (2-3 for a 16x16 patch, 4 for 8x8). The trip count
// depends on patch_size at run time.
template<typename T>
__device__ __inline__ bool load_ref_patch(const unsigned char * __restrict__ d_in_ref_img,
                                          const float2 & ref_px,
                                          const int & img_width,
                                          const int & img_height,
                                          const int & img_pitch,
                                          const int & patch_size,
                                          const int & half_patch_size,
                                          T * __restrict__ d_out_ref_patch) {
  // Calculate top left corner
  // and verify that the patch fits in the image
  float x_tl_f = ref_px.x - (half_patch_size+1);
  float y_tl_f = ref_px.y - (half_patch_size+1);

  int x_tl_i = floorf(x_tl_f);
  int y_tl_i = floorf(y_tl_f);
  // Note: on the right side of the rectangle, there's a +1 because of the
  //       intensity interpolation
  if(x_tl_i < 0 ||
     y_tl_i < 0 ||
#if REFERENCE_PATCH_INTERPOLATION
     (x_tl_i+patch_size+2)>=img_width ||
     (y_tl_i+patch_size+2)>=img_height
#else
     (x_tl_i+patch_size+1)>=img_width ||
     (y_tl_i+patch_size+1)>=img_height
#endif /* REFERENCE_PATCH_INTERPOLATION */
    ) {
    return false;
  }

#if REFERENCE_PATCH_INTERPOLATION
  float subpix_x = x_tl_f-x_tl_i;
  float subpix_y = y_tl_f-y_tl_i;
  float wTL = (1.0f-subpix_x)*(1.0f-subpix_y);
  float wTR = subpix_x*(1.0f-subpix_y);
  float wBL = (1.0f-subpix_x)*subpix_y;
  float wBR = 1.0f - wTL - wTR - wBL;
#endif /* REFERENCE_PATCH_INTERPOLATION */

  T * patch_ptr = d_out_ref_patch + threadIdx.x;
  #pragma unroll
  for(int id = threadIdx.x; id < (patch_size+2)*(patch_size+2); id += WARP_SIZE, patch_ptr += WARP_SIZE) {
    int x_no_offs = (id % (patch_size+2));
    int y_no_offs = (id / (patch_size+2));
    int xi = x_no_offs + x_tl_i;
    int yi = y_no_offs + y_tl_i;

    const unsigned char * ptr = d_in_ref_img + yi*img_pitch + xi;
    *patch_ptr =
#if REFERENCE_PATCH_INTERPOLATION
          (T)(wTL*((float)ptr[0]) +
              wBL*((float)ptr[img_pitch]) +
              wTR*((float)ptr[1]) +
              wBR*((float)ptr[img_pitch+1]));
#else
          (T)ptr[0];
#endif /* REFERENCE_PATCH_INTERPOLATION */
  }
  return true;
}


// [SLAM zero-to-hero] H = sum over the patch of J J^T, with the same lane -> pixel mapping as
// perform_lk. H[10] holds the upper triangle; after unrolling, every index into H and J is a
// constant, so both arrays stay in registers (a runtime index would push them to local memory).
// The reduction uses __shfl_down_sync: only lane 0 needs the total, because lane 0 alone inverts
// H (closed-form adjugate / determinant) and writes H^-1. The other 31 lanes idle meanwhile;
// this runs once per template and level, so the divergence costs little.
template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__device__ __inline__ void calc_hessian(const int & img_width,
                                        const int & img_height,
                                        const int & img_pitch,
                                        const T * ref_patch,
                                        const int & patch_size,
                                        const int & half_patch_size,
                                        const int & patch_stride,
                                        float * __restrict__ d_inv_hessian) {
  /*
   * We're exploiting the fact that H is going to be symmetric !
   * - When we estimate also affine offset AND gain:
   * J = [ 0 1 2 3 ]
   * H = | 0 1 2 3 |
   *     | x 4 5 6 |
   *     | x x 7 8 |
   *     | x x x 9 |
   *
   * - When we estimate only affine offset OR gain
   * J = [ 0 1 2 ]
   * H = | 0 1 2 |
   *     | x 3 4 |
   *     | x x 5 |
   *
   * - When we don't estimate the affine offset and gain
   * J = [ 0 1 ]
   * H = | 0 1 |
   *     | x 2 |
   */
  float H[10];
  float J[4];
  #pragma unroll
  for(int i=0;i<10;++i) {
    H[i] = 0.0f;
  }

  const int patch_area = patch_size * patch_size;
  const int x = threadIdx.x % patch_size;
  const int y = threadIdx.x / patch_size;
  const T * it_ref_start = ref_patch + (y+1)*patch_stride + (x+1); // +1 due to the 1-1-1-1 borders
  const T * it_ref = it_ref_start;
  const int it_ref_offset = patch_stride * WARP_SIZE / patch_size;
  const int pixels_per_thread = patch_area/WARP_SIZE;

  #pragma unroll
  for(int i=0;i<pixels_per_thread;++i,it_ref+=it_ref_offset) {
    // Compute J
    J[0] = 0.5f * (it_ref[1] - it_ref[-1]);
    J[1] = 0.5f * (it_ref[patch_stride] - it_ref[-patch_stride]);
    // Affine parameter estimation
    if(affine_est_offset && affine_est_gain) {
      J[2] = 1.0f;
      J[3] = it_ref[0];
    } else if(affine_est_offset) {
      J[2] = 1.0f;
    } else if(affine_est_gain) {
      J[2] = it_ref[0];
    }

    // H += J*J^Transpose (using the fact that J*J^T is going to be symmetric)
    if(affine_est_offset && affine_est_gain) {
      /*
       * H: 4x4 matrix
       *    0      1     2     3
       * 0  (0x0) * (0x1) (0x2) (0x3)
       * 1  x     (1x1) (1x2) (1x3)
       * 2  x     x     (2x2) (2x3)
       * 3  x     x     x     (3x3)
       */
       H[0] += J[0]*J[0];
       H[1] += J[0]*J[1];
       H[2] += J[0]*J[2];
       H[3] += J[0]*J[3];
       H[4] += J[1]*J[1];
       H[5] += J[1]*J[2];
       H[6] += J[1]*J[3];
       H[7] += J[2]*J[2];
       H[8] += J[2]*J[3];
       H[9] += J[3]*J[3];
    } else if(affine_est_offset || affine_est_gain) {
      /*
       * H: 3x3 matrix
       *    0      1     2
       * 0  (0x0) (0x1) (0x2)
       * 1  x     (1x1) (1x2)
       * 2  x     x     (2x2)
       */
       H[0] += J[0]*J[0];
       H[1] += J[0]*J[1];
       H[2] += J[0]*J[2];
       H[3] += J[1]*J[1];
       H[4] += J[1]*J[2];
       H[5] += J[2]*J[2];
    } else {
      /*
       img_cur_width_pitch_diff* H: 2x2 matrix
       *    0      1
       * 0  (0x0) (0x1)
       * 1  x     (1x1)
       */
      H[0] += J[0]*J[0];
      H[1] += J[0]*J[1];
      H[2] += J[1]*J[1];
    }
  }

  // Reduce it down to lane 0
  #pragma unroll
  for(int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    #pragma unroll
    for(int i=0;i<10;++i) {
      H[i] += __shfl_down_sync(WARP_MASK, H[i], offset);
    }
  }

  // [SLAM zero-to-hero] No pivoting and no conditioning test: a singular H gives 1/0 = inf, and
  // then NaN (0 * inf) or inf entries, which track_features_kernel deals with as noted there.
  // Calculate the inverse of H
  float inv_detH;
  if(threadIdx.x == 0) {
    if(affine_est_gain && affine_est_offset) {
      // Inverse of a symmetric 4x4 matrix
      inv_detH = 1.0f /
              (H[0]*H[4]*H[7]*H[9] + 2*H[0]*H[5]*H[6]*H[8] + H[1]*H[1]*H[8]*H[8]
              + 2*H[1]*H[2]*H[5]*H[9] + 2*H[1]*H[3]*H[6]*H[7]
              + 2*H[2]*H[3]*H[4]*H[8] + H[2]*H[2]*H[6]*H[6] + H[3]*H[3]*H[5]*H[5]
              - H[0]*H[4]*H[8]*H[8] - H[0]*H[5]*H[5]*H[9] - H[0]*H[6]*H[6]*H[7]
              - H[1]*H[1]*H[7]*H[9] - 2*H[1]*H[3]*H[5]*H[8] - 2*H[1]*H[2]*H[6]*H[8]
              - H[2]*H[2]*H[4]*H[9] - 2*H[2]*H[3]*H[5]*H[6] - H[3]*H[3]*H[4]*H[7]);
      d_inv_hessian[0] = (H[4]*H[7]*H[9] + 2*H[5]*H[6]*H[8] - H[4]*H[8]*H[8] - H[5]*H[5]*H[9] - H[6]*H[6]*H[7]) * inv_detH;
      d_inv_hessian[1] = (H[1]*H[8]*H[8] + H[2]*H[5]*H[9] + H[3]*H[6]*H[7] - H[1]*H[7]*H[9] - H[2]*H[6]*H[8] - H[3]*H[5]*H[8]) * inv_detH;
      d_inv_hessian[2] = (H[1]*H[5]*H[9] + H[2]*H[6]*H[6] + H[3]*H[4]*H[8] - H[1]*H[6]*H[8] - H[2]*H[4]*H[9] - H[3]*H[5]*H[6]) * inv_detH;
      d_inv_hessian[3] = (H[1]*H[6]*H[7] + H[2]*H[4]*H[8] + H[3]*H[5]*H[5] - H[1]*H[5]*H[8] - H[2]*H[6]*H[5] - H[3]*H[4]*H[7]) * inv_detH;
      d_inv_hessian[4] = (H[0]*H[7]*H[9] + 2*H[2]*H[3]*H[8] - H[0]*H[8]*H[8] - H[2]*H[2]*H[9] - H[3]*H[3]*H[7]) * inv_detH;
      d_inv_hessian[5] = (H[0]*H[6]*H[8] + H[1]*H[2]*H[9] + H[3]*H[3]*H[5] - H[0]*H[5]*H[9] - H[2]*H[3]*H[6] - H[1]*H[3]*H[8]) * inv_detH;
      d_inv_hessian[6] = (H[0]*H[5]*H[8] + H[2]*H[2]*H[6] + H[1]*H[3]*H[7] - H[0]*H[6]*H[7] - H[1]*H[2]*H[8] - H[2]*H[3]*H[5]) * inv_detH;
      d_inv_hessian[7] = (H[0]*H[4]*H[9] + 2*H[1]*H[3]*H[6] - H[0]*H[6]*H[6] - H[1]*H[1]*H[9] - H[3]*H[3]*H[4]) * inv_detH;
      d_inv_hessian[8] = (H[0]*H[5]*H[6] + H[1]*H[1]*H[8] + H[2]*H[3]*H[4] - H[0]*H[4]*H[8] - H[1]*H[2]*H[6] - H[1]*H[3]*H[5]) * inv_detH;
      d_inv_hessian[9] = (H[0]*H[4]*H[7] + 2*H[1]*H[2]*H[5] - H[0]*H[5]*H[5] - H[1]*H[1]*H[7] - H[2]*H[2]*H[4]) * inv_detH;
    } else if(affine_est_gain || affine_est_offset) {
      // Inverse of a symmetric 3x3 matrix
      inv_detH = 1.0f / (H[0]*H[3]*H[5] + 2*H[1]*H[4]*H[2] - H[0]*H[4]*H[4] - H[2]*H[3]*H[2] - H[1]*H[1]*H[5]);
      d_inv_hessian[0] = (H[3]*H[5] - H[4]*H[4]) * inv_detH;
      d_inv_hessian[1] = (H[2]*H[4] - H[1]*H[5]) * inv_detH;
      d_inv_hessian[2] = (H[1]*H[4] - H[2]*H[3]) * inv_detH;
      d_inv_hessian[3] = (H[0]*H[5] - H[2]*H[2]) * inv_detH;
      d_inv_hessian[4] = (H[1]*H[2] - H[0]*H[4]) * inv_detH;
      d_inv_hessian[5] = (H[0]*H[3] - H[1]*H[1]) * inv_detH;
    } else {
      // Inverse of a symmetric 2x2 matrix
      inv_detH = 1.0f / (H[0]*H[2] - H[1]*H[1]);
      d_inv_hessian[0] = H[2] * inv_detH;
      d_inv_hessian[1] = -1.0f * H[1] * inv_detH;
      d_inv_hessian[2] = H[0] * inv_detH;
    }
  }
}

// [SLAM zero-to-hero] One warp per new (or re-templated) track, 3 per block. For every level it
// cuts the template out of the current frame's pyramid into the track's slot of the patch
// buffer and stores H^-1 in its slot of the separate H^-1 buffer. The position is d_in_ref_px
// (the first observation by default), scaled to the level and truncated
// (REFERENCE_PATCH_INTERPOLATION is 0).
template <typename T, const bool affine_est_offset, const bool affine_est_gain>
__global__ void update_tracks_kernel(const int candidate_num,
                                     const int min_level,
                                     const int max_level,
                                     const image_pyramid_descriptor_t pyramid_description,
                                     const pyramid_patch_descriptor_t pyramid_patch_sizes,
                                     const int * __restrict__ d_indir_data,
                                     const float2 * __restrict__ d_in_ref_px,
                                     T * __restrict__ d_patch_data,
                                     float * __restrict__ d_hessian_data) {
  const int cx = blockIdx.x * CANDIDATES_PER_BLOCK_UPDATE + threadIdx.y; // candidate id
  const int pyramid_levels = max_level - min_level + 1; // number of pyramid levels computed
  if(cx < candidate_num) {
    // Get the buffer id
    const int bx = d_indir_data[cx];
    // Metadata array size: 64 bytes
    const float2 & ref_px_bx = d_in_ref_px[bx<<3];
    // Patch data
    T * d_patch_data_bx = d_patch_data + pyramid_patch_sizes.max_area*pyramid_levels*bx;
    // Hessian data
    float * d_hessian_data_bx = d_hessian_data + 10*pyramid_levels*bx;

    // For all the levels, precompute the interpolated patch, and the resulting inverse Hessian
    for(int level=max_level;
        level>=min_level;
        --level,d_patch_data_bx+=pyramid_patch_sizes.max_area,d_hessian_data_bx+=10) {
      const float scale = (float)(1<<level);
      const float inv_scale = 1.0f/scale;
      const float2 ref_px_scaled = {.x = ref_px_bx.x * inv_scale, .y = ref_px_bx.y * inv_scale};

      const unsigned char * img_ref = pyramid_description.data[level];
      const int img_width  = pyramid_description.desc.w[level];
      const int img_height = pyramid_description.desc.h[level];
      const int img_pitch  = pyramid_description.desc.p[level];

      const int patch_size = pyramid_patch_sizes.wh[level];
      const int half_patch_size = patch_size >> 1;
      const int patch_stride = patch_size + 2;

      // Create the reference patch if possible with borders
      if(load_ref_patch<T>(img_ref,
                           ref_px_scaled,
                           img_width,
                           img_height,
                           img_pitch,
                           patch_size,
                           half_patch_size,
                           d_patch_data_bx) == false) {
          // To notify the subsequent kernel of this behaviour,
          // use the 0th index Hessian
          d_hessian_data_bx[0] = __int_as_float(0x7fffffff);
          continue;
      }
      // [SLAM zero-to-hero] calc_hessian reads patch pixels that other lanes just wrote (the
      // gradient neighbours, and a different lane -> pixel mapping). __syncwarp() is a barrier for
      // the warp that also orders its memory accesses. Since Volta, the lanes of a warp are not
      // guaranteed to run in lockstep, so it is needed; __syncthreads() would also wait for the
      // block's other warps (other features) for no reason. All lanes reach it: the fit test
      // above is warp-uniform.
      __syncwarp();

      // Calculate the Hessian and its inverse
      calc_hessian<T,affine_est_offset,affine_est_gain>(img_width,
                                                        img_height,
                                                        img_pitch,
                                                        d_patch_data_bx,
                                                        patch_size,
                                                        half_patch_size,
                                                        patch_stride,
                                                        d_hessian_data_bx);
    }
  }
}

// [SLAM zero-to-hero] Same dispatch pattern as track_features(). Asynchronous: nothing in
// FeatureTrackerGPU::track() waits for this kernel before returning (see the note at step 01
// in feature_tracker_gpu.cpp).
__host__ void update_tracks(const int candidate_num,
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
                            cudaStream_t stream) {
  dim3 threads_per_block;
  threads_per_block.x = WARP_SIZE;
  threads_per_block.y = CANDIDATES_PER_BLOCK_UPDATE;
  threads_per_block.z = 1;
  const int blocks_per_grid = (candidate_num + CANDIDATES_PER_BLOCK_UPDATE - 1) / CANDIDATES_PER_BLOCK_UPDATE;
  // Launch kernel
  if(affine_est_offset && affine_est_gain) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,true><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else if(affine_est_offset) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,true,false><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else if(affine_est_gain) {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,true><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  } else {
    update_tracks_kernel<FEATURE_TRACKER_REFERENCE_PATCH_TYPE,false,false><<<blocks_per_grid,threads_per_block,0,stream>>>(
                                                          candidate_num,
                                                          min_level,
                                                          max_level,
                                                          pyramid_description,
                                                          pyramid_patch_sizes,
                                                          d_indir_data,
                                                          d_in_ref_px,
                                                          (FEATURE_TRACKER_REFERENCE_PATCH_TYPE*)d_patch_data,
                                                          d_hessian_data);
  }
  CUDA_KERNEL_CHECK();
}

} // namespace feature_tracker_cuda_tools
} // namespace vilib
