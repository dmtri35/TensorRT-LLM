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

from tensorrt_llm._torch.speculative.mtp import MTPWorker


class _Metadata:

    def __init__(self):
        self.num_seqs = 2
        self._seq_lens = torch.tensor([4, 4, 0], dtype=torch.int)
        self._seq_lens_cuda = torch.tensor([4, 4, 0], dtype=torch.int)
        self.kv_lens_cuda = torch.tensor([10, 20, 30], dtype=torch.int)
        self.helix_position_offsets = torch.tensor([100, 101, 102],
                                                   dtype=torch.int)
        self.helix_is_inactive_rank = torch.tensor([False, True, False],
                                                   dtype=torch.bool)
        self._saved_tensors = {}
        self.on_update_called = False

    def prepare_for_spec_dec(self, *fields):
        for field in fields:
            value = getattr(self, field)
            self._saved_tensors[field] = value
            setattr(self, field, value.clone())

    def restore_from_spec_dec(self):
        for field, value in self._saved_tensors.items():
            setattr(self, field, value)
        self._saved_tensors.clear()

    def on_update(self):
        self.on_update_called = True


class _Mapping:
    cp_size = 2
    cp_rank = 0

    def has_cp_helix(self):
        return True


class _ModelConfig:

    def __init__(self):
        self.mapping = _Mapping()


class _HelixMetadata:

    def __init__(self):
        self.seq_lens_cuda = torch.tensor([2, 1], dtype=torch.int)
        self.helix_total_input_len = torch.tensor([100, 200], dtype=torch.int)
        self.helix_position_offsets = torch.zeros(8, dtype=torch.int)
        self.helix_is_inactive_rank = torch.ones(8, dtype=torch.bool)
        self.tokens_per_block = 2
        self.num_tokens = 5


def test_mtp_spec_dec_restore_preserves_kv_lens_cuda():
    worker = object.__new__(MTPWorker)
    metadata = _Metadata()

    worker._prepare_attn_metadata_for_spec_dec(metadata)
    metadata.kv_lens_cuda[:metadata.num_seqs] += 7
    metadata.helix_position_offsets += 7
    metadata.helix_is_inactive_rank.logical_not_()

    worker._restore_attn_metadata_from_spec_dec(metadata)

    assert torch.equal(metadata.kv_lens_cuda,
                       torch.tensor([10, 20, 30], dtype=torch.int))
    assert torch.equal(metadata.helix_position_offsets,
                       torch.tensor([100, 101, 102], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank,
                       torch.tensor([False, True, False], dtype=torch.bool))
    assert metadata.on_update_called


def test_helix_draft_owner_mask_uses_current_position_count():
    worker = object.__new__(MTPWorker)
    worker.model_config = _ModelConfig()
    metadata = _HelixMetadata()

    owner_counts = worker._helix_draft_owner_mask(
        metadata,
        torch.tensor([100, 101, 200], dtype=torch.int),
        batch_size=2)

    assert torch.equal(owner_counts, torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(metadata.helix_position_offsets[:3],
                       torch.tensor([100, 101, 200], dtype=torch.int))
    assert torch.equal(metadata.helix_is_inactive_rank[:3],
                       torch.tensor([False, False, False]))
