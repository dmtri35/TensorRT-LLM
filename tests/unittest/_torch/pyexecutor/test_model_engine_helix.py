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

import torch

from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine,
    _get_num_tokens_after_helix_cp,
)


class _Mapping:

    def __init__(self, has_cp_helix: bool, cp_rank: int = 0, cp_size: int = 4):
        self._has_cp_helix = has_cp_helix
        self.cp_rank = cp_rank
        self.cp_size = cp_size

    def has_cp_helix(self):
        return self._has_cp_helix


class _Request:

    def __init__(self, total_input_len_cp: int,
                 py_helix_global_decode_len: int):
        self.total_input_len_cp = total_input_len_cp
        self.py_helix_global_decode_len = py_helix_global_decode_len


def test_get_num_tokens_after_helix_cp_localizes_token_count():
    assert _get_num_tokens_after_helix_cp(96, _Mapping(True)) == 24
    assert _get_num_tokens_after_helix_cp(65, _Mapping(True)) == 17


def test_get_num_tokens_after_helix_cp_preserves_token_count_without_helix():
    assert _get_num_tokens_after_helix_cp(96, _Mapping(False)) == 96


def test_helix_verify_token_params_starts_at_unsettled_decode_index():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True, cp_rank=2)
    request = _Request(total_input_len_cp=100,
                       py_helix_global_decode_len=5)

    positions, inactive_flags, num_active = engine._helix_verify_token_params(
        request, num_draft=3, tokens_per_block=2)

    assert positions == [105, 106, 107, 108]
    assert inactive_flags == [False, True, True, True]
    assert num_active == 1


def test_helix_verify_token_params_keeps_initial_decode_index():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True, cp_rank=0)
    request = _Request(total_input_len_cp=100,
                       py_helix_global_decode_len=0)

    positions, inactive_flags, num_active = engine._helix_verify_token_params(
        request, num_draft=2, tokens_per_block=2)

    assert positions == [100, 101, 102]
    assert inactive_flags == [False, False, True]
    assert num_active == 2


class _Dist:

    def tp_allgather(self, obj):
        return [obj] * 4


class _TorchCompileBackend:
    capture_num_tokens = [32, 64, 128]


class _AttentionMetadata:

    def __init__(self):
        self.kv_cache_manager = object()
        self.num_seqs = 2
        self.num_contexts = 1
        self.num_generations = 1
        self.num_ctx_tokens = 2
        self.num_tokens = 5
        self.num_chunked_ctx_requests = 0
        self.helix_position_offsets = torch.tensor([0, 1, 10, 11, 12],
                                                   dtype=torch.int)
        self.helix_is_inactive_rank = torch.tensor(
            [False, False, False, False, False], dtype=torch.bool)
        self.helix_total_input_len = torch.tensor([0, 0], dtype=torch.int)
        self.seq_lens_cuda = torch.tensor([2, 3], dtype=torch.int)
        self.tokens_per_block = 2
        self.kv_lens_cuda = torch.tensor([2, 10], dtype=torch.int)
        self.on_update_kv_lens_calls = 0

    def on_update_kv_lens(self):
        self.on_update_kv_lens_calls += 1


def test_helix_overlap_updates_kernel_position_offsets():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True, cp_rank=0, cp_size=2)
    engine.enable_spec_decode = True
    engine._disable_overlap_scheduler = False
    engine.guided_decoder = None
    engine.previous_pos_id_offsets_cuda = torch.tensor([2, 2, 2],
                                                       dtype=torch.int)
    engine.previous_kv_lens_offsets_cuda = torch.tensor([-99],
                                                        dtype=torch.int)

    metadata = _AttentionMetadata()
    metadata.helix_position_offsets = torch.tensor([0, 1, 101, 102, 103],
                                                   dtype=torch.int)
    metadata.helix_total_input_len = torch.tensor([0, 100], dtype=torch.int)
    metadata.helix_is_inactive_rank = torch.tensor(
        [False, False, False, True, True], dtype=torch.bool)
    metadata.kv_lens_cuda = torch.tensor([2, 11], dtype=torch.int)
    inputs = {
        'attn_metadata': metadata,
        'input_ids': torch.tensor([1, 2, 3, 4, 5], dtype=torch.int),
        'position_ids': torch.tensor([[0, 1, 101, 102, 103]],
                                     dtype=torch.int),
    }

    engine._preprocess_inputs(inputs)

    assert torch.equal(inputs['position_ids'],
                       torch.tensor([[0, 1, 103, 104, 105]], dtype=torch.int))
    assert torch.equal(metadata.helix_position_offsets,
                       torch.tensor([0, 1, 103, 104, 105], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank,
                       torch.tensor([False, False, True, False, False]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 13]))
    assert metadata.on_update_kv_lens_calls == 1

    engine._postprocess_inputs(inputs)

    assert torch.equal(inputs['position_ids'],
                       torch.tensor([[0, 1, 101, 102, 103]],
                                    dtype=torch.int))
    assert torch.equal(metadata.helix_position_offsets,
                       torch.tensor([0, 1, 101, 102, 103], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank,
                       torch.tensor([False, False, False, True, True]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 11]))


def test_helix_overlap_ignores_padded_input_rows():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True, cp_rank=0, cp_size=2)
    engine.enable_spec_decode = True
    engine._disable_overlap_scheduler = False
    engine.guided_decoder = None
    engine.previous_pos_id_offsets_cuda = torch.tensor([2, 2, 2, 0, 0, 0],
                                                       dtype=torch.int)
    engine.previous_kv_lens_offsets_cuda = torch.tensor([-99],
                                                        dtype=torch.int)

    metadata = _AttentionMetadata()
    metadata.helix_position_offsets = torch.tensor(
        [0, 1, 101, 102, 103, 0, 0, 0], dtype=torch.int)
    metadata.helix_total_input_len = torch.tensor([0, 100], dtype=torch.int)
    metadata.helix_is_inactive_rank = torch.tensor(
        [False, False, False, True, True, False, False, False],
        dtype=torch.bool)
    metadata.kv_lens_cuda = torch.tensor([2, 11], dtype=torch.int)
    inputs = {
        'attn_metadata':
        metadata,
        'input_ids':
        torch.tensor([1, 2, 3, 4, 5, 0, 0, 0], dtype=torch.int),
        'position_ids':
        torch.tensor([[0, 1, 101, 102, 103, 0, 0, 0]], dtype=torch.int),
    }

    engine._preprocess_inputs(inputs)

    assert torch.equal(
        inputs['position_ids'],
        torch.tensor([[0, 1, 103, 104, 105, 0, 0, 0]], dtype=torch.int))
    assert torch.equal(
        metadata.helix_position_offsets,
        torch.tensor([0, 1, 103, 104, 105, 0, 0, 0], dtype=torch.int))
    assert torch.equal(
        metadata.helix_is_inactive_rank,
        torch.tensor(
            [False, False, True, False, False, False, False, False]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 13]))

    engine._postprocess_inputs(inputs)

    assert torch.equal(
        inputs['position_ids'],
        torch.tensor([[0, 1, 101, 102, 103, 0, 0, 0]], dtype=torch.int))
    assert torch.equal(
        metadata.helix_position_offsets,
        torch.tensor([0, 1, 101, 102, 103, 0, 0, 0], dtype=torch.int))
    assert torch.equal(
        metadata.helix_is_inactive_rank,
        torch.tensor(
            [False, False, False, True, True, False, False, False]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 11]))


def test_helix_piecewise_padding_keeps_full_token_shape():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True)
    engine.enable_attention_dp = True
    engine.dist = _Dist()
    engine._torch_compile_backend = _TorchCompileBackend()
    engine._torch_compile_piecewise_cuda_graph = True

    padded_num_tokens, can_run_graph, all_rank_num_tokens = (
        engine._get_padding_params(
            total_num_tokens=96,
            num_ctx_requests=1,
            attn_all_rank_num_tokens=[24] * 4,
        ))

    assert padded_num_tokens == 128
    assert can_run_graph is True
    assert all_rank_num_tokens == [32] * 4
