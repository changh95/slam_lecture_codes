/*
 * FAST feature detector configuration
 * fast_gpu_config.h
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

namespace vilib {

/*
 * Note to future self:
 * Lessons learnt from implementing FAST on GPU:
 * - recomputing offsets is faster then getting it from constant memory
 * - constant memory serializes accesses if not the same address is accessed within warp
 *   -> constant memory is not a good choice for lookup table
 * - shared memory: this is sad, but the amount of data reuse is not enough to overcome the shm loading/synchronization overheads
 *   after profiling, the arithmetic instructions (due to the complicated SHM loading,
 *   increased the kernel time :( -> you can still find the shared memory version in the commits before "GPU feature detector overhaul"
 * - precomputing the pitches help!
 * - literals are better than const memory accesses
 * - offset computing locally is better than constant memory (e.g. for NMS it was -20usec)
 * - precheck -10 usec (coarse corner response function)
 * - compute response only for the points where the bresenham circle passed the test (-13usec)
 * - texture memory helped during response calculation (-10usec, because the borders were not calculated anymore), but NMS got considerably worse with texture memory
 * - integer multiply is expensive, use the tenary operator if possible
 */
// [SLAM zero-to-hero] Upstream likely measured the notes above on the two GPUs named in the NMS
// kernel's note (detector_base_gpu_cuda_tools.cu): Maxwell (a GTX 960M) and Pascal (Jetson TX2).
// Two of them on a current GPU:
// "integer multiply is expensive" dates from Maxwell and Pascal, which had no native 32-bit integer
// multiply (it took several 16-bit XMAD instructions); Volta and later, the RTX 5090 included, have
// one (IMAD), so it matters less now. "constant memory serializes accesses" still holds: the
// constant cache serves one address at a time, so a warp reading 32 different table entries takes
// up to 32 rounds. That is why the lookup table below lives in global memory.
/*
 * Lookup table
 * Note to future self:
 * we could also do some bit magic with __clz().
 * Downside of a lookup table is the serialized accesses to global memory
 * -50usec from kernel runtime
 */
#define FAST_GPU_USE_LOOKUP_TABLE              1
/*
 * Lookup table bit-based
 * Note to future self:
 * bitbase lookup table:
 * lower 11 bit -> address of the 4 bytes
 * upper 5  bit (32 bits) -> we have a 1 where the combination is good
 * ok, sadly it did not help significantly
 */
#define FAST_GPU_USE_LOOKUP_TABLE_BITBASED     1
// [SLAM zero-to-hero] As shipped both are 1: one bit per 16-bit darker/brighter mask, so 65536 bits
// in 2048 words, 8 KB (cudaMalloc(8*1024) in fast_gpu.cpp); the byte version would be 64 KB. Either
// way the table sits in global memory. Setting FAST_GPU_USE_LOOKUP_TABLE to 0 does not compile as
// shipped: the MAX_THRESHOLD score in fast_gpu_cuda_tools.cu still passes d_corner_lut, which is
// then not a kernel parameter.

} // namespace vilib
