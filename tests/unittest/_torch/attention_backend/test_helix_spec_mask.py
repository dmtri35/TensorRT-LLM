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

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.attention_backend.trtllm import (
    _build_helix_spec_decoding_packed_mask,
    _build_helix_spec_decoding_position_offsets,
    _should_use_helix_spec_decoding_mask,
)


def _unpack_row(row: torch.Tensor, width: int) -> list[int]:
    value = int(row[0].item())
    return [idx for idx in range(width) if value & (1 << idx)]


def test_helix_spec_decoding_mask_uses_owned_suffix_order():
    seq_lens = torch.tensor([4, 4], dtype=torch.long)
    helix_is_inactive_rank = torch.tensor(
        [
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
        ],
        dtype=torch.bool,
    )

    packed_mask, owned_counts = _build_helix_spec_decoding_packed_mask(
        helix_is_inactive_rank, seq_lens)

    assert owned_counts.tolist() == [4, 2]
    assert _unpack_row(packed_mask[0, 0], 4) == [0]
    assert _unpack_row(packed_mask[0, 3], 4) == [0, 1, 2, 3]

    assert _unpack_row(packed_mask[1, 0], 4) == []
    assert _unpack_row(packed_mask[1, 1], 4) == [0]
    assert _unpack_row(packed_mask[1, 2], 4) == [0]
    assert _unpack_row(packed_mask[1, 3], 4) == [0, 1]


def test_helix_spec_decoding_position_offsets_are_materialized():
    position_offsets = _build_helix_spec_decoding_position_offsets(
        num_seqs=3, max_generation_length=4)

    assert position_offsets.tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    assert position_offsets.is_contiguous()
    assert position_offsets.stride(0) == 4


def test_helix_spec_decoding_mask_is_disabled_for_mla():
    assert not _should_use_helix_spec_decoding_mask(
        is_mla_enable=True,
        attention_input_type=AttentionInputType.generation_only,
        mask_ready=True,
        sm=100,
    )


def test_helix_spec_decoding_mask_metadata_is_enabled_for_mla_microstep(
        monkeypatch):
    monkeypatch.setenv("TRTLLM_HELIX_MLA_MTP_MICROSTEP", "1")

    assert _should_use_helix_spec_decoding_mask(
        is_mla_enable=True,
        attention_input_type=AttentionInputType.generation_only,
        mask_ready=True,
        sm=100,
    )
    assert not _should_use_helix_spec_decoding_mask(
        is_mla_enable=True,
        attention_input_type=AttentionInputType.generation_only,
        mask_ready=False,
        sm=100,
    )
    assert not _should_use_helix_spec_decoding_mask(
        is_mla_enable=True,
        attention_input_type=AttentionInputType.context_only,
        mask_ready=True,
        sm=100,
    )
