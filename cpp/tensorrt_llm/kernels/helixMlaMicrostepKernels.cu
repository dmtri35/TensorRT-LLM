/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/kernels/helixMlaMicrostepKernels.h"

#include <cmath>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

__global__ void packHelixMlaMicrostepsKernel(uint8_t const* q, uint8_t* qSteps,
    int32_t const* firstSparseOffsetsKv, int32_t const* packedMask, int32_t* stepKvLens, int32_t batchSize,
    int32_t seqLenQ, int32_t packedMaskSeqStride, int32_t packedMaskBlockStride, int32_t validMaskBlocks,
    size_t qRowBytes)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const step = blockIdx.y;
    if (batchIdx >= batchSize)
    {
        return;
    }

    uint8_t const* src = q + (static_cast<size_t>(batchIdx) * seqLenQ + step) * qRowBytes;
    uint8_t* dst = qSteps + (static_cast<size_t>(step) * batchSize + batchIdx) * qRowBytes;
    for (size_t byteIdx = threadIdx.x; byteIdx < qRowBytes; byteIdx += blockDim.x)
    {
        dst[byteIdx] = src[byteIdx];
    }

    if (threadIdx.x != 0)
    {
        return;
    }

    int32_t ownedCount = 0;
    int32_t const maskOffset = (batchIdx * packedMaskSeqStride + step) * packedMaskBlockStride;
    for (int32_t maskBlockIdx = 0; maskBlockIdx < validMaskBlocks; ++maskBlockIdx)
    {
        ownedCount += __popc(static_cast<uint32_t>(packedMask[maskOffset + maskBlockIdx]));
    }
    stepKvLens[static_cast<size_t>(step) * batchSize + batchIdx] = firstSparseOffsetsKv[batchIdx] + ownedCount;
}

__global__ void unpackHelixMlaMicrostepsKernel(uint8_t const* oSteps, uint8_t* o, float2 const* softmaxStatsSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const step = blockIdx.y;
    if (batchIdx >= batchSize)
    {
        return;
    }

    uint8_t const* src = oSteps + (static_cast<size_t>(step) * batchSize + batchIdx) * oRowBytes;
    uint8_t* dst = o + (static_cast<size_t>(batchIdx) * seqLenQ + step) * oRowBytes;
    for (size_t byteIdx = threadIdx.x; byteIdx < oRowBytes; byteIdx += blockDim.x)
    {
        dst[byteIdx] = src[byteIdx];
    }

    if (softmaxStats != nullptr && softmaxStatsSteps != nullptr)
    {
        float2 const* statsSrc = softmaxStatsSteps + (static_cast<size_t>(step) * batchSize + batchIdx) * numHeads;
        float2* statsDst = softmaxStats + (static_cast<size_t>(batchIdx) * seqLenQ + step) * numHeads;
        for (int32_t headIdx = threadIdx.x; headIdx < numHeads; headIdx += blockDim.x)
        {
            statsDst[headIdx] = statsSrc[headIdx];
        }
    }
}

__device__ float2 convertLseToHelixStat(float const lse)
{
    if (!isfinite(lse))
    {
        return make_float2(-INFINITY, 0.0F);
    }

    // Helix combines normalized partial outputs using sum * exp(max - global_max).
    // FlashMLA exposes log(sum(exp(scores))) for each row, so encode it as
    // max=logZ and sum=1 to produce the same correction factor.
    return make_float2(lse, 1.0F);
}

__global__ void convertFlashMlaLseToHelixStatsKernel(
    float const* softmaxLse, float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, int32_t numHeads)
{
    int32_t const totalStats = batchSize * seqLenQ * numHeads;
    for (int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalStats; idx += blockDim.x * gridDim.x)
    {
        float const lse = softmaxLse[idx];
        softmaxStats[idx] = convertLseToHelixStat(lse);
    }
}

__global__ void unpackHelixFlashMlaMicrostepsKernel(uint8_t const* oSteps, uint8_t* o, float const* softmaxLseSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads)
{
    int32_t const batchIdx = blockIdx.x;
    int32_t const step = blockIdx.y;
    if (batchIdx >= batchSize)
    {
        return;
    }

    uint8_t const* src = oSteps + (static_cast<size_t>(step) * batchSize + batchIdx) * oRowBytes;
    uint8_t* dst = o + (static_cast<size_t>(batchIdx) * seqLenQ + step) * oRowBytes;
    for (size_t byteIdx = threadIdx.x; byteIdx < oRowBytes; byteIdx += blockDim.x)
    {
        dst[byteIdx] = src[byteIdx];
    }

    if (softmaxStats != nullptr && softmaxLseSteps != nullptr)
    {
        float const* lseSrc = softmaxLseSteps + (static_cast<size_t>(step) * batchSize + batchIdx) * numHeads;
        float2* statsDst = softmaxStats + (static_cast<size_t>(batchIdx) * seqLenQ + step) * numHeads;
        for (int32_t headIdx = threadIdx.x; headIdx < numHeads; headIdx += blockDim.x)
        {
            statsDst[headIdx] = convertLseToHelixStat(lseSrc[headIdx]);
        }
    }
}

} // namespace

void invokePackHelixMlaMicrosteps(void const* q, void* qSteps, int32_t const* firstSparseOffsetsKv,
    int32_t const* packedMask, int32_t* stepKvLens, int32_t batchSize, int32_t seqLenQ, int32_t packedMaskSeqStride,
    int32_t packedMaskBlockStride, int32_t validMaskBlocks, size_t qRowBytes, cudaStream_t stream)
{
    static constexpr int kThreadsPerBlock = 256;
    dim3 const grid(batchSize, seqLenQ);
    packHelixMlaMicrostepsKernel<<<grid, kThreadsPerBlock, 0, stream>>>(static_cast<uint8_t const*>(q),
        static_cast<uint8_t*>(qSteps), firstSparseOffsetsKv, packedMask, stepKvLens, batchSize, seqLenQ,
        packedMaskSeqStride, packedMaskBlockStride, validMaskBlocks, qRowBytes);
}

void invokeUnpackHelixMlaMicrosteps(void const* oSteps, void* o, float2 const* softmaxStatsSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads, cudaStream_t stream)
{
    static constexpr int kThreadsPerBlock = 256;
    dim3 const grid(batchSize, seqLenQ);
    unpackHelixMlaMicrostepsKernel<<<grid, kThreadsPerBlock, 0, stream>>>(static_cast<uint8_t const*>(oSteps),
        static_cast<uint8_t*>(o), softmaxStatsSteps, softmaxStats, batchSize, seqLenQ, oRowBytes, numHeads);
}

void invokeConvertFlashMlaLseToHelixStats(
    float const* softmaxLse, float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, int32_t numHeads,
    cudaStream_t stream)
{
    if (softmaxLse == nullptr || softmaxStats == nullptr)
    {
        return;
    }

    static constexpr int kThreadsPerBlock = 256;
    int32_t const totalStats = batchSize * seqLenQ * numHeads;
    int32_t const numBlocks = (totalStats + kThreadsPerBlock - 1) / kThreadsPerBlock;
    convertFlashMlaLseToHelixStatsKernel<<<numBlocks, kThreadsPerBlock, 0, stream>>>(
        softmaxLse, softmaxStats, batchSize, seqLenQ, numHeads);
}

void invokeUnpackHelixFlashMlaMicrosteps(void const* oSteps, void* o, float const* softmaxLseSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads, cudaStream_t stream)
{
    static constexpr int kThreadsPerBlock = 256;
    dim3 const grid(batchSize, seqLenQ);
    unpackHelixFlashMlaMicrostepsKernel<<<grid, kThreadsPerBlock, 0, stream>>>(static_cast<uint8_t const*>(oSteps),
        static_cast<uint8_t*>(o), softmaxLseSteps, softmaxStats, batchSize, seqLenQ, oRowBytes, numHeads);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
