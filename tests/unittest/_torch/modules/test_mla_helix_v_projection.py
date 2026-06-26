# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.modules.attention import (
    MLA, _helix_softmax_correction)


def test_helix_v_projection_uses_all_cp_head_shards() -> None:
    num_tokens = 3
    cp_size = 2
    heads_per_cp = 2
    num_heads_tp = cp_size * heads_per_cp
    qk_nope_head_dim = 3
    v_head_dim = 2
    kv_lora_rank = 4

    partial_o = torch.randn(num_tokens,
                            num_heads_tp * kv_lora_rank,
                            dtype=torch.float32)
    kv_b_proj_weight = torch.randn(
        num_heads_tp * (qk_nope_head_dim + v_head_dim),
        kv_lora_rank,
        dtype=torch.float32)

    mla = object.__new__(MLA)
    mla.kv_b_proj = SimpleNamespace(weight=kv_b_proj_weight)
    mla.kv_lora_rank = kv_lora_rank
    mla.num_heads_tp = num_heads_tp
    mla.qk_nope_head_dim = qk_nope_head_dim
    mla.v_head_dim = v_head_dim
    mla.v_b_proj = torch.empty(heads_per_cp, v_head_dim, kv_lora_rank)

    projected = MLA._project_helix_latent_to_v(mla, partial_o)

    _, v_weight = kv_b_proj_weight.split(
        [num_heads_tp * qk_nope_head_dim, num_heads_tp * v_head_dim],
        dim=0)
    v_weight = v_weight.view(num_heads_tp, v_head_dim, kv_lora_rank)
    expected = torch.bmm(
        partial_o.view(num_tokens, num_heads_tp,
                       kv_lora_rank).transpose(0, 1),
        v_weight.transpose(1, 2),
    ).transpose(0, 1).reshape(num_tokens, num_heads_tp * v_head_dim)

    torch.testing.assert_close(projected, expected)


def test_helix_softmax_correction_matches_global_softmax_weights() -> None:
    num_tokens = 2
    cp_size = 3
    num_heads = 4
    local_rank = 1

    softmax_stats_by_rank = torch.randn(num_tokens, cp_size, num_heads, 2)
    softmax_stats_by_rank[..., 1] = torch.rand(num_tokens, cp_size, num_heads)
    gathered_softmax_stats = softmax_stats_by_rank.reshape(
        num_tokens, cp_size * num_heads, 2)
    local_softmax_stats = softmax_stats_by_rank[:, local_rank]

    correction = _helix_softmax_correction(local_softmax_stats,
                                           gathered_softmax_stats, cp_size)

    local_max = softmax_stats_by_rank[..., 0]
    local_sum = softmax_stats_by_rank[..., 1]
    global_max = local_max.max(dim=1).values
    denominator = (local_sum *
                   torch.exp(local_max - global_max[:, None, :])).sum(dim=1)
    expected = (
        local_sum[:, local_rank] *
        torch.exp(local_max[:, local_rank] - global_max) / denominator)

    torch.testing.assert_close(correction, expected)


def test_helix_softmax_correction_zeros_empty_global_rows() -> None:
    num_tokens = 2
    cp_size = 2
    num_heads = 3
    local_softmax_stats = torch.zeros(num_tokens, num_heads, 2)
    local_softmax_stats[..., 0] = float("-inf")
    gathered_softmax_stats = torch.zeros(num_tokens, cp_size * num_heads, 2)
    gathered_softmax_stats[..., 0] = float("-inf")

    correction = _helix_softmax_correction(local_softmax_stats,
                                           gathered_softmax_stats, cp_size)

    torch.testing.assert_close(correction, torch.zeros_like(correction))
