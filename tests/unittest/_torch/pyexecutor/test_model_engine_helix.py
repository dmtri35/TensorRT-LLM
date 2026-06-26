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
from tensorrt_llm._torch.speculative.mtp import MTPEagleWorker


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


class _SpecDecMode:

    def has_draft_model(self):
        return False

    def is_mtp_one_model(self):
        return True


class _SpecConfig:
    spec_dec_mode = _SpecDecMode()


class _CudaGraphRunner:
    enabled = True


class _GenerationRequest:

    def __init__(self, request_id: int):
        self.py_request_id = request_id


class _ScheduledRequests:
    num_context_requests = 0

    def __init__(self, request_ids):
        self.generation_requests = [
            _GenerationRequest(request_id) for request_id in request_ids
        ]


class _ModelConfig:

    def __init__(self, mapping):
        self.mapping = mapping


class _MTPEagleWorker:

    def __init__(self, mapping):
        self.model_config = _ModelConfig(mapping)


class _HelixDraftMetadata:

    def __init__(self):
        self.num_tokens = 8
        self.num_contexts = 0
        self.seq_lens_cuda = torch.tensor([4, 4], dtype=torch.int64)
        self.helix_total_input_len = torch.tensor([0, 0], dtype=torch.int64)
        self.tokens_per_block = 1
        self.helix_position_offsets = torch.zeros(8, dtype=torch.int32)
        self.helix_is_inactive_rank = torch.ones(8, dtype=torch.bool)
        self.helix_zero_kv_mask = None


def test_get_num_tokens_after_helix_cp_localizes_token_count():
    assert _get_num_tokens_after_helix_cp(96, _Mapping(True)) == 24
    assert _get_num_tokens_after_helix_cp(65, _Mapping(True)) == 17


def test_get_num_tokens_after_helix_cp_preserves_token_count_without_helix():
    assert _get_num_tokens_after_helix_cp(96, _Mapping(False)) == 96


def test_helix_mtp_owner_mask_uses_first_draft_sequence_lengths():
    worker = _MTPEagleWorker(_Mapping(True, cp_rank=0, cp_size=4))
    metadata = _HelixDraftMetadata()
    position_ids = torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int64)

    owner_counts = MTPEagleWorker._helix_draft_owner_mask(
        worker, metadata, position_ids, batch_size=2)

    assert torch.equal(owner_counts, torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(metadata.helix_position_offsets[:6],
                       position_ids.to(torch.int32))
    assert torch.equal(
        metadata.helix_is_inactive_rank[:6],
        torch.tensor([False, True, True, False, True, True]))


def test_helix_mtp_incremental_update_requires_accepted_tokens_tensor():
    engine = object.__new__(PyTorchModelEngine)
    engine.spec_config = _SpecConfig()
    engine.mapping = _Mapping(True)
    engine.cuda_graph_runner = _CudaGraphRunner()
    engine.use_mrope = False
    engine.previous_request_ids = [123]
    engine.is_draft_model = False
    engine.model_is_wrapped = False
    engine.has_previous_device_draft = True

    scheduled_requests = _ScheduledRequests([123])
    new_tokens = torch.tensor([[1, 2, 3, 4]])
    next_draft_tokens = torch.tensor([[2, 3, 4]])

    assert not engine._can_use_incremental_update(
        scheduled_requests,
        new_tokens_device=new_tokens,
        next_draft_tokens_device=next_draft_tokens,
        num_accepted_tokens_device=None)
    assert engine._can_use_incremental_update(
        scheduled_requests,
        new_tokens_device=new_tokens,
        next_draft_tokens_device=next_draft_tokens,
        num_accepted_tokens_device=torch.tensor([1]))


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


def test_helix_overlap_updates_zero_kv_mask():
    engine = object.__new__(PyTorchModelEngine)
    engine.mapping = _Mapping(True, cp_rank=0, cp_size=2)
    engine.enable_spec_decode = True
    engine._disable_overlap_scheduler = False
    engine.guided_decoder = None
    engine.previous_pos_id_offsets_cuda = torch.tensor([2, 2],
                                                       dtype=torch.int)
    engine.previous_kv_lens_offsets_cuda = torch.tensor([-99],
                                                        dtype=torch.int)

    metadata = _AttentionMetadata()
    metadata.num_tokens = 4
    metadata.seq_lens_cuda = torch.tensor([2, 2], dtype=torch.int)
    metadata.helix_position_offsets = torch.tensor([0, 1, 102, 103],
                                                   dtype=torch.int)
    metadata.helix_total_input_len = torch.tensor([0, 100], dtype=torch.int)
    metadata.helix_is_inactive_rank = torch.tensor(
        [False, False, True, True], dtype=torch.bool)
    metadata.helix_zero_kv_mask = torch.tensor([False, False, True, True],
                                               dtype=torch.bool)
    metadata.kv_lens_cuda = torch.tensor([2, 0], dtype=torch.int)
    inputs = {
        'attn_metadata': metadata,
        'input_ids': torch.tensor([1, 2, 3, 4], dtype=torch.int),
        'position_ids': torch.tensor([[0, 1, 102, 103]], dtype=torch.int),
    }

    engine._preprocess_inputs(inputs)

    assert torch.equal(inputs['position_ids'],
                       torch.tensor([[0, 1, 104, 105]], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank,
                       torch.tensor([False, False, False, False]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 2]))
    assert torch.equal(metadata.helix_zero_kv_mask,
                       torch.tensor([False, False, False, False]))

    engine._postprocess_inputs(inputs)

    assert torch.equal(inputs['position_ids'],
                       torch.tensor([[0, 1, 102, 103]], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank,
                       torch.tensor([False, False, True, True]))
    assert torch.equal(metadata.kv_lens_cuda, torch.tensor([2, 0]))
    assert torch.equal(metadata.helix_zero_kv_mask,
                       torch.tensor([False, False, True, True]))


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
