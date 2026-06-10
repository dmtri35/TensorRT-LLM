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

from tensorrt_llm._torch.modules import attention as attention_module
from tensorrt_llm._torch.modules.attention import maybe_slice_for_helix_cp


class _Metadata:

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens


class _Mapping:
    cp_size = 4
    cp_rank = 2
    enable_attention_dp = True

    def has_cp_helix(self):
        return True


def test_helix_mtp_residual_slice_can_be_forced_for_nonzero_layer_idx():
    tensor = torch.arange(96 * 2).reshape(96, 2)

    sliced = maybe_slice_for_helix_cp(tensor,
                                      _Metadata(num_tokens=96),
                                      _Mapping(),
                                      layer_idx=42,
                                      force_slice=True)

    assert torch.equal(sliced, tensor[48:72])


def test_helix_input_allgather_can_be_skipped_for_full_mtp_input(monkeypatch):
    tensor = torch.arange(96 * 2).reshape(96, 2)

    def fail_cp_allgather(*args, **kwargs):
        raise AssertionError("cp_allgather should not be called")

    monkeypatch.setattr(attention_module, "cp_allgather", fail_cp_allgather)

    output = attention_module._helix_cp_allgather_input(tensor,
                                                        _Metadata(96),
                                                        _Mapping(),
                                                        layer_idx=42,
                                                        input_is_full=True)

    assert output is tensor
