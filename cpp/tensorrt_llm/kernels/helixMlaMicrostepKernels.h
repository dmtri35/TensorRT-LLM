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
#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokePackHelixMlaMicrosteps(void const* q, void* qSteps, int32_t const* firstSparseOffsetsKv,
    int32_t const* packedMask, int32_t* stepKvLens, int32_t batchSize, int32_t seqLenQ, int32_t packedMaskSeqStride,
    int32_t packedMaskBlockStride, int32_t validMaskBlocks, size_t qRowBytes, cudaStream_t stream);

void invokeUnpackHelixMlaMicrosteps(void const* oSteps, void* o, float2 const* softmaxStatsSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads, cudaStream_t stream);

void invokeConvertFlashMlaLseToHelixStats(
    float const* softmaxLse, float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, int32_t numHeads,
    cudaStream_t stream);

void invokeUnpackHelixFlashMlaMicrosteps(void const* oSteps, void* o, float const* softmaxLseSteps,
    float2* softmaxStats, int32_t batchSize, int32_t seqLenQ, size_t oRowBytes, int32_t numHeads, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
