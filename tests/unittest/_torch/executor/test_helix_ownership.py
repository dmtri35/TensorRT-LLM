# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTP
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    _helix_count_owned_decode_indices, _helix_owns_decode_index,
    _helix_spec_overlap_reserve)
from tensorrt_llm._torch.speculative.mtp import MTPWorker


def test_helix_decode_index_owner_is_round_robin_by_block():
    tokens_per_block = 4
    cp_size = 3

    owners = [
        next(
            cp_rank for cp_rank in range(cp_size)
            if _helix_owns_decode_index(decode_index, tokens_per_block,
                                        cp_size, cp_rank))
        for decode_index in range(16)
    ]

    assert owners == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        0,
        0,
        0,
        0,
    ]


def test_helix_owned_count_matches_naive_scan_for_unaligned_ranges():
    tokens_per_block = 3
    cp_size = 4

    for cp_rank in range(cp_size):
        for start_index in range(12):
            for count in range(17):
                expected = sum(
                    1 for decode_index in range(start_index,
                                                start_index + count)
                    if _helix_owns_decode_index(decode_index,
                                                tokens_per_block, cp_size,
                                                cp_rank))

                assert _helix_count_owned_decode_indices(
                    start_index, count, tokens_per_block, cp_size,
                    cp_rank) == expected


def test_helix_mtp_owner_mask_keeps_local_context_tokens_active():

    class Mapping:

        cp_size = 2
        cp_rank = 0

        def has_cp_helix(self):
            return True

    worker = object.__new__(MTPWorker)
    worker.model_config = SimpleNamespace(mapping=Mapping())
    attn_metadata = SimpleNamespace(
        tokens_per_block=4,
        num_tokens=4,
        seq_lens_cuda=torch.tensor([4], dtype=torch.int32),
        helix_total_input_len=torch.tensor([10], dtype=torch.int32),
        helix_position_offsets=torch.empty(4, dtype=torch.int32),
        helix_is_inactive_rank=torch.empty(4, dtype=torch.bool),
    )

    owner_counts = worker._helix_draft_owner_mask(
        attn_metadata,
        torch.tensor([8, 9, 10, 11], dtype=torch.int32),
        batch_size=1,
    )

    assert owner_counts.item() == 4
    assert not attn_metadata.helix_is_inactive_rank.any()


def test_helix_mtp_owner_mask_uses_static_repeat_sizes(monkeypatch):

    class Mapping:

        cp_size = 2
        cp_rank = 0

        def has_cp_helix(self):
            return True

    real_repeat_interleave = torch.repeat_interleave

    def require_output_size(values, repeats, *args, **kwargs):
        if torch.is_tensor(repeats) and "output_size" not in kwargs:
            raise AssertionError(
                "tensor repeat counts must provide output_size")
        return real_repeat_interleave(values, repeats, *args, **kwargs)

    monkeypatch.setattr(torch, "repeat_interleave", require_output_size)

    worker = object.__new__(MTPWorker)
    worker.model_config = SimpleNamespace(mapping=Mapping())
    attn_metadata = SimpleNamespace(
        tokens_per_block=2,
        num_tokens=5,
        seq_lens_cuda=torch.tensor([2, 3], dtype=torch.int32),
        kv_lens_cuda=torch.tensor([0, 5], dtype=torch.int32),
        helix_total_input_len=torch.tensor([10, 20], dtype=torch.int32),
        helix_position_offsets=torch.empty(5, dtype=torch.int32),
        helix_is_inactive_rank=torch.empty(5, dtype=torch.bool),
        helix_zero_kv_mask=torch.empty(5, dtype=torch.bool),
    )

    owner_counts = worker._helix_draft_owner_mask(
        attn_metadata,
        torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32),
        batch_size=2,
    )

    assert torch.equal(owner_counts, torch.tensor([2, 2], dtype=torch.int32))
    assert torch.equal(attn_metadata.helix_position_offsets,
                       torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32))
    assert torch.equal(attn_metadata.helix_is_inactive_rank,
                       torch.tensor([False, False, False, False, True]))
    assert torch.equal(attn_metadata.helix_zero_kv_mask,
                       torch.tensor([True, True, False, False, False]))


def test_helix_mtp_spec_all_rank_counts_use_cp_shards():

    class Mapping:

        cp_size = 4

        def has_cp_helix(self):
            return True

    class Dist:

        def __init__(self):
            self.gathered_value = None

        def tp_cp_allgather(self, value):
            self.gathered_value = value
            return [value, value]

    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = Mapping()
    engine.dist = Dist()

    all_rank_counts = engine._get_spec_all_rank_num_tokens(
        spec_num_tokens=128,
        num_sequences=32,
    )

    assert engine.dist.gathered_value == [32, 8]
    assert all_rank_counts == [[32, 8], [32, 8]]


def test_deepseek_mtp_eh_proj_split_uses_projection_layout():
    mtp = object.__new__(DeepseekV3MTP)
    mtp.model_config = SimpleNamespace(
        mapping=SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            enable_attention_dp=False,
        ))
    mtp.eh_proj = SimpleNamespace(
        tp_mode=TensorParallelMode.ROW,
        tp_size=2,
        tp_rank=1,
    )
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    sliced = mtp._split_eh_proj_input(hidden_states)

    assert torch.equal(sliced, hidden_states[:, 8:])


def test_helix_spec_overlap_reserve_covers_shifted_verify_window():
    draft_len = 1
    tokens_per_block = 2
    cp_size = 3
    reserve = _helix_spec_overlap_reserve(draft_len, max_draft_len=1)

    assert reserve == 3

    for accepted_tokens in range(1, draft_len + 2):
        current_window_end = accepted_tokens + draft_len
        assert current_window_end <= reserve
        accepted_drafts = accepted_tokens - 1

        for cp_rank in range(cp_size):
            allocated = _helix_count_owned_decode_indices(
                0, reserve + 1, tokens_per_block, cp_size, cp_rank)
            rewound = _helix_count_owned_decode_indices(
                accepted_tokens, reserve - accepted_drafts, tokens_per_block,
                cp_size, cp_rank)
            kept = _helix_count_owned_decode_indices(
                0, accepted_tokens, tokens_per_block, cp_size, cp_rank)

            assert allocated - rewound == kept


def test_helix_spec_overlap_reserve_keeps_plain_decode_unchanged():
    assert _helix_spec_overlap_reserve(draft_len=0, max_draft_len=0) == 0
