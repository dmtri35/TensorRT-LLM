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
#include "tensorrt_llm/kernels/helixMlaKernels.h"

#include <cmath>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

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

} // namespace

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
    if (totalStats <= 0)
    {
        return;
    }

    int32_t const numBlocks = (totalStats + kThreadsPerBlock - 1) / kThreadsPerBlock;
    convertFlashMlaLseToHelixStatsKernel<<<numBlocks, kThreadsPerBlock, 0, stream>>>(
        softmaxLse, softmaxStats, batchSize, seqLenQ, numHeads);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
