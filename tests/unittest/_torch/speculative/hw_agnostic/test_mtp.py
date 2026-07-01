import unittest
from types import SimpleNamespace

import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modules.attention import helix_cp_selective_reduce_rows
from tensorrt_llm._torch.speculative.eagle3 import MTPEagleWorker
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager, MTPSpecMetadata, MTPWorker
from tensorrt_llm.llmapi import MTPDecodingConfig


def unittest_name_func(testcase_func, param_num, param):
    name = param.args[0]
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(name),
    )


class TestMTPSampleAndAcceptDraftTokens(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")

    def test_helix_draft_owner_mask_accepts_long_seq_lens(self):
        mapping = SimpleNamespace(cp_size=2,
                                  cp_rank=1,
                                  has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)

        batch_size = 2
        num_tokens = 5
        attn_metadata = SimpleNamespace(
            num_tokens=num_tokens,
            num_contexts=0,
            tokens_per_block=2,
            seq_lens_cuda=torch.tensor([2, 3], dtype=torch.long),
            helix_total_input_len=torch.tensor([10, 20], dtype=torch.long),
            helix_position_offsets=torch.empty(num_tokens, dtype=torch.int32),
            helix_is_inactive_rank=torch.empty(num_tokens, dtype=torch.bool),
            helix_is_inactive_rank_per_token=False,
        )
        position_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.long)

        owner_counts = worker._helix_draft_owner_mask(attn_metadata,
                                                      position_ids,
                                                      batch_size)

        torch.testing.assert_close(
            attn_metadata.helix_is_inactive_rank,
            torch.tensor([True, True, True, True, False]))
        torch.testing.assert_close(owner_counts,
                                   torch.tensor([0, 1], dtype=torch.int32))
        self.assertTrue(attn_metadata.helix_is_inactive_rank_per_token)

    def test_update_helix_param_records_inactive_layout(self):
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=2, max_num_tokens=8, kv_cache_manager=None)
        attn_metadata.helix_position_offsets = torch.empty(8,
                                                           dtype=torch.int32)
        attn_metadata.helix_position_offsets_cpu = torch.empty_like(
            attn_metadata.helix_position_offsets)
        attn_metadata.helix_is_inactive_rank = torch.empty(8,
                                                           dtype=torch.bool)
        attn_metadata.helix_is_inactive_rank_cpu = torch.empty_like(
            attn_metadata.helix_is_inactive_rank)

        attn_metadata.update_helix_param(
            helix_position_offsets=[10, 11],
            helix_is_inactive_rank=[False, True],
            helix_is_inactive_rank_per_token=False,
        )
        self.assertFalse(attn_metadata.helix_is_inactive_rank_per_token)

        attn_metadata.update_helix_param(
            helix_position_offsets=[12, 13, 14],
            helix_is_inactive_rank=[False, True, False],
            helix_is_inactive_rank_per_token=True,
        )
        self.assertTrue(attn_metadata.helix_is_inactive_rank_per_token)

    def test_helix_cp_selective_reduce_rows_matches_full_rows(self):
        num_tokens = 5
        hidden_size = 3
        cp_size = 2
        gather_ids = torch.tensor([1, 4], dtype=torch.long)
        full_hidden_states = torch.arange(num_tokens * hidden_size,
                                          dtype=torch.float32).reshape(
                                              num_tokens, hidden_size)
        chunk_size = (num_tokens + cp_size - 1) // cp_size
        padded = torch.nn.functional.pad(
            full_hidden_states, (0, 0, 0, chunk_size * cp_size - num_tokens))
        rank_major_gathered = torch.stack([
            padded[rank * chunk_size:(rank + 1) * chunk_size]
            for rank in range(cp_size)
        ]).reshape(cp_size * chunk_size, hidden_size)

        def selected_contributions_by_rank():
            selected_by_rank = []
            for rank in range(cp_size):
                local_chunk = rank_major_gathered[rank * chunk_size:(rank + 1)
                                                  * chunk_size]
                local_start = rank * chunk_size
                local_offsets = gather_ids - local_start
                is_local = (local_offsets >= 0) & (local_offsets < chunk_size)
                selected = local_chunk[local_offsets.clamp(0, chunk_size - 1)]
                selected_by_rank.append(selected * is_local.unsqueeze(-1))
            return selected_by_rank

        all_contributions = selected_contributions_by_rank()
        expected = full_hidden_states[gather_ids]
        for cp_rank in range(cp_size):
            reduce_inputs = []

            def fake_reduce(tensor, mapping):
                reduce_inputs.append(tensor.clone())
                return torch.stack(all_contributions).sum(dim=0)

            mapping = SimpleNamespace(cp_size=cp_size,
                                      cp_rank=cp_rank,
                                      enable_attention_dp=True,
                                      has_cp_helix=lambda: True)
            local_chunk = padded[cp_rank * chunk_size:(cp_rank + 1) *
                                 chunk_size]
            attn_metadata = SimpleNamespace(num_tokens=num_tokens)
            with self.subTest(cp_rank=cp_rank):
                selected = helix_cp_selective_reduce_rows(
                    local_chunk,
                    gather_ids,
                    attn_metadata,
                    mapping,
                    reduce_fn=fake_reduce,
                )
                torch.testing.assert_close(reduce_inputs[0],
                                           all_contributions[cp_rank])
                torch.testing.assert_close(selected, expected)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_helix_draft_owner_mask_cuda_graph_safe(self):
        mapping = SimpleNamespace(cp_size=2,
                                  cp_rank=1,
                                  has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)

        batch_size = 2
        num_tokens = 5
        attn_metadata = SimpleNamespace(
            num_tokens=num_tokens,
            num_contexts=0,
            tokens_per_block=2,
            seq_lens_cuda=torch.tensor([2, 3],
                                       dtype=torch.long,
                                       device="cuda"),
            helix_total_input_len=torch.tensor([10, 20],
                                               dtype=torch.long,
                                               device="cuda"),
            helix_position_offsets=torch.empty(num_tokens,
                                               dtype=torch.int32,
                                               device="cuda"),
            helix_is_inactive_rank=torch.empty(num_tokens,
                                               dtype=torch.bool,
                                               device="cuda"),
        )
        position_ids = torch.tensor([10, 11, 20, 21, 22],
                                    dtype=torch.long,
                                    device="cuda")

        for _ in range(3):
            worker._helix_draft_owner_mask(attn_metadata, position_ids,
                                           batch_size)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            worker._helix_draft_owner_mask(attn_metadata, position_ids,
                                           batch_size)

        attn_metadata.helix_is_inactive_rank.fill_(False)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(
            attn_metadata.helix_is_inactive_rank.cpu(),
            torch.tensor([True, True, True, True, False]))

    def test_helix_draft_logits_use_explicit_gather_ids(self):
        mapping = SimpleNamespace(has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)
        attn_metadata = SimpleNamespace(
            num_seqs=3,
            seq_lens_cuda=torch.tensor([2, 3, 1], dtype=torch.long),
        )
        hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)

        calls = []

        class FakeMTPLayer:

            def shared_head(self,
                            hidden_states_arg,
                            lm_head,
                            attn_metadata_arg,
                            return_context_logits=False):
                calls.append((hidden_states_arg.clone(),
                              return_context_logits))
                return hidden_states_arg

        logits, gather_ids, sampler_mapping = worker._draft_logits_for_sampling(
            FakeMTPLayer(), hidden_states, SimpleNamespace(), attn_metadata)

        expected_gather_ids = torch.tensor([1, 4, 5], dtype=torch.long)
        torch.testing.assert_close(gather_ids, expected_gather_ids)
        torch.testing.assert_close(logits, hidden_states[expected_gather_ids])
        torch.testing.assert_close(calls[0][0], hidden_states[expected_gather_ids])
        self.assertTrue(calls[0][1])
        self.assertIsNone(sampler_mapping)

    def test_helix_draft_logits_return_shared_head_sampler_mapping(self):
        mapping = SimpleNamespace(has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)
        attn_metadata = SimpleNamespace(
            num_seqs=1,
            seq_lens_cuda=torch.tensor([2], dtype=torch.long),
        )
        hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        sampler_mapping = SimpleNamespace(tp_size=2)

        class FakeSharedHead:
            mapping_lm_head_tp = sampler_mapping

            def __call__(self,
                         hidden_states_arg,
                         lm_head,
                         attn_metadata_arg,
                         return_context_logits=False):
                return hidden_states_arg

        fake_mtp_layer = SimpleNamespace(shared_head=FakeSharedHead())

        logits, gather_ids, returned_mapping = worker._draft_logits_for_sampling(
            fake_mtp_layer, hidden_states, SimpleNamespace(), attn_metadata)

        torch.testing.assert_close(logits, hidden_states[[1]])
        torch.testing.assert_close(gather_ids, torch.tensor([1]))
        self.assertIs(returned_mapping, sampler_mapping)

    def test_helix_draft_logits_preserve_selected_hidden_states(self):
        mapping = SimpleNamespace(has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)
        attn_metadata = SimpleNamespace(
            num_seqs=2,
            seq_lens_cuda=torch.tensor([2, 3], dtype=torch.long),
        )
        selected_hidden_states = torch.arange(8, dtype=torch.float32).reshape(
            2, 4)
        gather_ids = torch.tensor([1, 4], dtype=torch.long)

        calls = []

        class FakeMTPLayer:

            def shared_head(self,
                            hidden_states_arg,
                            lm_head,
                            attn_metadata_arg,
                            return_context_logits=False):
                calls.append((hidden_states_arg.clone(),
                              return_context_logits))
                return hidden_states_arg

        logits, returned_gather_ids, sampler_mapping = worker._draft_logits_for_sampling(
            FakeMTPLayer(),
            selected_hidden_states,
            SimpleNamespace(),
            attn_metadata,
            gather_ids,
            hidden_states_are_selected=True)

        torch.testing.assert_close(logits, selected_hidden_states)
        torch.testing.assert_close(calls[0][0], selected_hidden_states)
        torch.testing.assert_close(returned_gather_ids, gather_ids)
        self.assertTrue(calls[0][1])
        self.assertIsNone(sampler_mapping)

    def test_mtp_eagle_forwards_helix_selected_rows_to_mtp_layer(self):
        mapping = SimpleNamespace(has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPEagleWorker(MTPDecodingConfig(max_draft_len=2),
                                model_config=model_config)
        self.assertTrue(worker.is_mtp_eagle)

        class FakeMTPLayer:

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return kwargs["hidden_states"]

        fake_mtp_layer = FakeMTPLayer()
        draft_model = SimpleNamespace(mtp_layers=[fake_mtp_layer],
                                      embed_tokens=object())
        inputs = {
            "input_ids": torch.tensor([1, 2], dtype=torch.int32),
            "position_ids": torch.tensor([10, 11], dtype=torch.long),
            "hidden_states": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "attn_metadata": SimpleNamespace(),
        }
        spec_metadata = SimpleNamespace(all_rank_num_tokens=[2],
                                        subseq_all_rank_num_tokens=[1])
        gather_ids = torch.tensor([1], dtype=torch.long)

        hidden_states, hidden_states_to_save = worker._run_draft_forward(
            draft_model,
            inputs,
            spec_metadata,
            step_idx=0,
            helix_cp_gather_ids=gather_ids,
            helix_cp_reduce_selected_rows=True)

        self.assertIsNone(hidden_states_to_save)
        torch.testing.assert_close(hidden_states, inputs["hidden_states"])
        self.assertEqual(len(fake_mtp_layer.calls), 1)
        self.assertIs(fake_mtp_layer.calls[0]["helix_cp_gather_ids"],
                      gather_ids)
        self.assertTrue(
            fake_mtp_layer.calls[0]["helix_cp_reduce_selected_rows"])

    def test_helix_select_cp_owner_values_uses_owner_rank_rows(self):
        attn_metadata = SimpleNamespace(num_tokens=5)
        gather_ids = torch.tensor([1, 4], dtype=torch.long)
        per_rank_values = [
            torch.tensor([11, -1], dtype=torch.int32),
            torch.tensor([-2, 44], dtype=torch.int32),
        ]
        per_rank_expected_inputs = [
            torch.tensor([11, 0], dtype=torch.int32),
            torch.tensor([0, 44], dtype=torch.int32),
        ]
        expected = torch.tensor([11, 44], dtype=torch.int32)

        for cp_rank in range(2):
            mapping = SimpleNamespace(cp_size=2,
                                      cp_rank=cp_rank,
                                      enable_attention_dp=True,
                                      has_cp_helix=lambda: True)
            model_config = SimpleNamespace(mapping=mapping)
            worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                               model_config=model_config)
            reduce_inputs = []

            def fake_reduce(values, mapping):
                reduce_inputs.append(values.clone())
                return expected

            with self.subTest(cp_rank=cp_rank):
                selected = worker._helix_select_cp_owner_values(
                    per_rank_values[cp_rank],
                    gather_ids,
                    attn_metadata,
                    reduce_fn=fake_reduce)
                torch.testing.assert_close(reduce_inputs[0],
                                           per_rank_expected_inputs[cp_rank])
                torch.testing.assert_close(selected, expected)

    def test_helix_select_cp_owner_values_skips_non_adp(self):
        mapping = SimpleNamespace(cp_size=2,
                                  cp_rank=0,
                                  enable_attention_dp=False,
                                  has_cp_helix=lambda: True)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)
        values = torch.tensor([11, 22], dtype=torch.int32)

        def fail_reduce(values, mapping):
            raise AssertionError("non-ADP Helix should not reduce token IDs")

        selected = worker._helix_select_cp_owner_values(
            values,
            torch.tensor([1, 4], dtype=torch.long),
            SimpleNamespace(num_tokens=5),
            reduce_fn=fail_reduce)

        torch.testing.assert_close(selected, values)

    def test_non_helix_draft_logits_preserve_default_row_selection(self):
        mapping = SimpleNamespace(has_cp_helix=lambda: False)
        model_config = SimpleNamespace(mapping=mapping)
        worker = MTPWorker(MTPDecodingConfig(max_draft_len=2),
                           model_config=model_config)
        attn_metadata = SimpleNamespace(
            num_seqs=3,
            seq_lens_cuda=torch.tensor([2, 3, 1], dtype=torch.long),
        )
        hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)

        calls = []

        class FakeMTPLayer:

            def shared_head(self,
                            hidden_states_arg,
                            lm_head,
                            attn_metadata_arg,
                            return_context_logits=False):
                calls.append((hidden_states_arg.clone(),
                              return_context_logits))
                return hidden_states_arg

        logits, gather_ids, sampler_mapping = worker._draft_logits_for_sampling(
            FakeMTPLayer(), hidden_states, SimpleNamespace(), attn_metadata)

        self.assertIsNone(gather_ids)
        self.assertIsNone(sampler_mapping)
        torch.testing.assert_close(logits, hidden_states)
        torch.testing.assert_close(calls[0][0], hidden_states)
        self.assertFalse(calls[0][1])

    def load_sample_and_accept_draft_tokens_test_cases():
        test_cases = []

        # '''
        ################# CASE 0 ##########################
        # BS=1, 1 context request
        mtp_num_modules = 1
        num_context_requests = 1
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([0], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")  # [batch_size]

        test_cases += [
            [
                "case0",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 1 ##########################
        # BS=4, 4 context request
        mtp_num_modules = 1
        num_context_requests = 4
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([0, 0, 0, 0], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0], [3, 0], [3, 0], [6, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor(
            [1, 1, 1, 1], dtype=torch.int, device="cuda"
        )  # [batch_size]

        test_cases += [
            [
                "case1",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 2 ##########################
        # BS=1, 1 generation request
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one from golden token + three draft tokens
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 4], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([3], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor([3], dtype=torch.int, device="cuda")  # [batch_size]

        test_cases += [
            [
                "case2",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 3 ##########################
        # BS=2, 2 generation request
        # Assume there are 1 MTP layers
        # For each generation request, there are two logits here: one golden token + one draft token
        mtp_num_modules = 1
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 5], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([1, 1], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3], [4, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor(
            [2, 1], dtype=torch.int, device="cuda"
        )  # [batch_size]

        test_cases += [
            [
                "case3",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 4 ##########################
        # BS=2, 2 generation request
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one golden token + three draft token
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 4, 4, 7, 3], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([3, 3], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0], [4, 6, 0, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor(
            [3, 2], dtype=torch.int, device="cuda"
        )  # [batch_size]

        test_cases += [
            [
                "case4",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 5 ##########################
        # BS=3, 3 generation request, 2 accept partial, 1 accept all, 1 accept none
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one golden token + three draft token
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, -100, -100, -100, -100, -100, 0],  # Top1 id = 7
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 5, 4, 6, 5, 5, 7, 4], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([3, 3, 3], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0], [4, 6, 5, 2], [4, 0, 0, 0]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor(
            [3, 4, 1], dtype=torch.int, device="cuda"
        )  # [batch_size]

        test_cases += [
            [
                "case5",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        ################# CASE 6 ##########################
        # BS=2, 1 context request, 1 generation request
        # request0 is context request, and request1 is generation request
        # Assume there are 1 MTP layers
        # For each generation request, there are two logits here: one golden token + one draft token
        mtp_num_modules = 1
        num_context_requests = 1
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [4], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        draft_len = torch.tensor([0, 1], dtype=torch.int, device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0], [4, 6]], dtype=torch.int, device="cuda"
        )  # [batch_size * max_draft_len]

        ref_num_accepted_tokens = torch.tensor(
            [1, 2], dtype=torch.int, device="cuda"
        )  # [batch_size]

        test_cases += [
            [
                "case6",
                mtp_num_modules,
                logits,
                draft_tokens,
                draft_len,
                num_context_requests,
                ref_accepted_tokens,
                ref_num_accepted_tokens,
            ]
        ]

        return test_cases

    @parameterized.expand(
        load_sample_and_accept_draft_tokens_test_cases, name_func=unittest_name_func
    )
    def test_sample_and_accept_draft_tokens(
        self,
        test_case_name,
        mtp_num_modules,
        logits,
        draft_tokens,
        draft_len,
        num_context_requests,
        ref_accepted_tokens,
        ref_num_accepted_tokens,
    ):
        batch_size = len(draft_len)
        spec_config = MTPDecodingConfig(max_draft_len=mtp_num_modules)
        # Simulate update_spec_config_from_model_config for vanilla MTP tests
        spec_config.num_nextn_predict_layers = mtp_num_modules

        # attention metedata
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size, max_num_tokens=1024, kv_cache_manager=None
        )
        attn_metadata.seq_lens = torch.tensor(
            [1] * batch_size, dtype=torch.int
        )  # dummy sequence length
        attn_metadata.num_contexts = num_context_requests

        # speculative decoding metadata
        spec_metadata = MTPSpecMetadata(
            max_num_requests=32,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_draft_len=mtp_num_modules,
            max_total_draft_tokens=mtp_num_modules,
            mtp_num_modules=mtp_num_modules,
        )
        spec_metadata.draft_tokens = draft_tokens

        # mtp worker
        mtpworker = MTPWorker(spec_config)

        # Test thop kernel
        # Test native torch op
        for is_thop in [True, False]:
            mtpworker.is_thop = is_thop
            # TODO: add unit tests for relaxed acceptance
            accepted_tokens, num_accepted_tokens = mtpworker.sample_and_accept_draft_tokens(
                None, logits, spec_metadata, attn_metadata
            )

            torch.testing.assert_close(num_accepted_tokens, ref_num_accepted_tokens)
            for i in range(len(draft_len)):
                torch.testing.assert_close(
                    accepted_tokens[i][0 : ref_num_accepted_tokens[i]],
                    ref_accepted_tokens[i][0 : ref_num_accepted_tokens[i]],
                )


class TestMTPUpdateMTPHiddenStates(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")

    def load_update_mtp_hidden_states_test_cases():
        def gen_data(batch_size, num_nextn_predict_layers, hidden_size):
            mtp_past_hidden_states_ptrs = []
            mtp_past_tokens_ptrs = []
            mtp_hidden_states_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers, hidden_size),
                device="cuda",
                dtype=torch.float32,
            )
            mtp_tokens_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers), device="cuda", dtype=torch.int
            )

            for bix in range(batch_size):
                mtp_hidden_states_tensor_pool[bix] = (
                    mtp_hidden_states_tensor_pool[bix] * bix
                )  # be different
                mtp_past_hidden_states_ptrs.append(mtp_hidden_states_tensor_pool[bix].data_ptr())

                mtp_tokens_tensor_pool[bix] = mtp_tokens_tensor_pool[bix] * bix  # be different
                mtp_past_tokens_ptrs.append(mtp_tokens_tensor_pool[bix].data_ptr())
            return (
                mtp_past_hidden_states_ptrs,
                mtp_past_tokens_ptrs,
                mtp_hidden_states_tensor_pool,
                mtp_tokens_tensor_pool,
            )

        test_cases = []

        ################# CASE 0 ##########################
        # BS=1, 1 context request
        batch_size = 1
        num_context_request = 1
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int, device="cuda")

        seq_lens = torch.tensor([5], dtype=torch.int, device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([9], dtype=torch.int, device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor(
            [
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        test_cases += [
            [
                "case0_is_thop_false",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                False,
            ]
        ]

        test_cases += [
            [
                "case0_is_thop_true",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                True,
            ]
        ]

        ################## CASE 1 ##########################
        # BS=3, 3 context request, num_nextn_predict_layers = 2
        batch_size = 3
        num_context_request = 3
        num_nextn_predict_layers = 2
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor(
            [
                5,
                6,
                7,
                8,
                9,  # request 1
                11,
                12,
                13,
                14,  # request 2
                21,
                22,
                23,
                24,
                25,
                26,  # request 3
            ],
            dtype=torch.int,
            device="cuda",
        )

        seq_lens = torch.tensor([5, 4, 6], dtype=torch.int, device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                # request 1
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                # request 2
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
                # request 3
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor(
            [1, 1, 1], dtype=torch.int, device="cuda"
        )  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([8, 9], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([13, 14], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([25, 26], dtype=torch.int, device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor(
            [
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[1] = torch.tensor(
            [
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[2] = torch.tensor(
            [
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        test_cases += [
            [
                "case1_is_thop_false",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                False,
            ]
        ]

        test_cases += [
            [
                "case1_is_thop_true",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                True,
            ]
        ]

        ################## CASE 2 ##########################
        # BS=1, 1 generation request, num_nextn_predict_layers = 1
        batch_size = 1
        num_context_request = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 42], dtype=torch.int, device="cuda")

        seq_lens = torch.tensor([2], dtype=torch.int, device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([2], dtype=torch.int, device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([42], dtype=torch.int, device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        test_cases += [
            [
                "case2_is_thop_false",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                False,
            ]
        ]

        test_cases += [
            [
                "case2_is_thop_true",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                True,
            ]
        ]

        ################## CASE 3 ##########################
        # BS=4, 4 generation request, num_nextn_predict_layers = 1
        batch_size = 4
        num_context_request = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor(
            [
                5,
                6,  # request 1
                7,
                8,  # request 2
                9,
                10,  # request 3
                11,
                12,  # request 4
            ],
            dtype=torch.int,
            device="cuda",
        )

        seq_lens = torch.tensor([2, 2, 2, 2], dtype=torch.int, device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor(
            [2, 1, 1, 2], dtype=torch.int, device="cuda"
        )  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([6], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([7], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([9], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[3] = torch.tensor([12], dtype=torch.int, device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[1] = torch.tensor(
            [
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[2] = torch.tensor(
            [
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[3] = torch.tensor(
            [
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        test_cases += [
            [
                "case3_is_thop_false",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                False,
            ]
        ]

        test_cases += [
            [
                "case3_is_thop_true",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                True,
            ]
        ]

        ################## CASE 4 ##########################
        # BS=4, 2 context request, 2 generation request, num_nextn_predict_layers = 2
        num_context = 2
        num_generation = 2
        num_context_request = num_context
        batch_size = num_context + num_generation
        num_nextn_predict_layers = 2
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor(
            [
                5,
                6,
                7,
                8,
                9,  # request 1
                10,
                11,
                12,
                13,  # request 2
                26,
                27,
                28,  # request 3
                31,
                32,
                33,  # request 4
            ],
            dtype=torch.int,
            device="cuda",
        )

        seq_lens = torch.tensor([5, 4, 3, 3], dtype=torch.int, device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                # request 1
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                # request 2
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
                # request 3
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                # request 4
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor(
            [1, 1, 3, 1], dtype=torch.int, device="cuda"
        )  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([8, 9], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([12, 13], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([27, 28], dtype=torch.int, device="cuda")
        ref_mtp_tokens_dict[3] = torch.tensor([3, 31], dtype=torch.int, device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor(
            [
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[1] = torch.tensor(
            [
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[2] = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        ref_mtp_hidden_state_dict[3] = torch.tensor(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        test_cases += [
            [
                "case4_is_thop_false",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                False,
            ]
        ]

        test_cases += [
            [
                "case4_is_thop_true",
                num_nextn_predict_layers,
                num_context_request,
                input_ids,
                seq_lens,
                hidden_states,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                request_ids,
                num_accepted_tokens,
                ref_mtp_tokens_dict,
                ref_mtp_hidden_state_dict,
                True,
            ]
        ]

        return test_cases

    @parameterized.expand(load_update_mtp_hidden_states_test_cases, name_func=unittest_name_func)
    def test_mtp_update_mtp_hidden_states(
        self,
        test_case_name,
        num_nextn_predict_layers,
        num_context_request,
        input_ids,
        seq_lens,
        hidden_states,
        mtp_hidden_states_ptrs,
        mtp_past_tokens_ptrs,
        mtp_hidden_states_tensor_pool,
        mtp_tokens_tensor_pool,
        request_ids,
        num_accepted_tokens,
        ref_mtp_tokens_dict,
        ref_mtp_hidden_state_dict,
        is_thop,
    ):
        batch_size = len(request_ids)
        batch_size - num_context_request
        hidden_size = hidden_states.shape[1]
        spec_config = MTPDecodingConfig(max_draft_len=num_nextn_predict_layers)
        # Simulate update_spec_config_from_model_config for vanilla MTP tests
        spec_config.num_nextn_predict_layers = num_nextn_predict_layers

        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size, max_num_tokens=1024, kv_cache_manager=None
        )
        attn_metadata.seq_lens = seq_lens.to("cpu")
        attn_metadata.num_contexts = num_context_request

        spec_manager = MTPHiddenStatesManager(
            config=spec_config,
            dtype=torch.float32,
            hidden_size=hidden_size,
            max_num_requests=batch_size,
        )
        for i in range(batch_size):
            # for the generation requests, we also need to manually add slot
            # because these generation requests are also first use
            spec_manager.slot_manager.add_slot(request_ids[i])

        spec_metadata = MTPSpecMetadata(
            max_num_requests=32,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_draft_len=num_nextn_predict_layers,
            max_total_draft_tokens=num_nextn_predict_layers,
            mtp_num_modules=num_nextn_predict_layers,
            mtp_hidden_states_manager=spec_manager,
        )
        spec_metadata.request_ids = request_ids
        spec_metadata.mtp_hidden_states_ptrs = mtp_hidden_states_ptrs
        spec_metadata.mtp_past_tokens_ptrs = mtp_past_tokens_ptrs

        spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool = (
            mtp_hidden_states_tensor_pool
        )
        spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool = mtp_tokens_tensor_pool
        spec_metadata.prepare()

        mtpworker = MTPWorker(spec_config)
        mtpworker.is_thop = is_thop

        mtpworker.update_mtp_hidden_states(
            input_ids=input_ids,
            hidden_states=hidden_states,
            num_accepted_tokens=num_accepted_tokens,
            spec_metadata=spec_metadata,
            attn_metadata=attn_metadata,
        )

        # Verify
        mtp_past_hidden_states_pool = (
            spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
        )
        mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool

        for bix in range(batch_size):
            torch.testing.assert_close(mtp_past_tokens_pool[bix], ref_mtp_tokens_dict[bix])
            torch.testing.assert_close(
                mtp_past_hidden_states_pool[bix], ref_mtp_hidden_state_dict[bix]
            )


class TestMTPPrepareDrafterInputs(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")

    def load_prepare_drafter_inputs_test_cases():
        def gen_data(batch_size, num_nextn_predict_layers, hidden_size):
            mtp_past_hidden_states_ptrs = []
            mtp_past_tokens_ptrs = []
            mtp_hidden_states_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers, hidden_size),
                device="cuda",
                dtype=torch.float32,
            )
            mtp_tokens_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers), device="cuda", dtype=torch.int
            )

            for bix in range(batch_size):
                mtp_hidden_states_tensor_pool[bix] = (
                    mtp_hidden_states_tensor_pool[bix] * bix
                )  # be different
                mtp_past_hidden_states_ptrs.append(mtp_hidden_states_tensor_pool[bix].data_ptr())

                mtp_tokens_tensor_pool[bix] = mtp_tokens_tensor_pool[bix] * bix  # be different
                mtp_past_tokens_ptrs.append(mtp_tokens_tensor_pool[bix].data_ptr())
            return (
                mtp_past_hidden_states_ptrs,
                mtp_past_tokens_ptrs,
                mtp_hidden_states_tensor_pool,
                mtp_tokens_tensor_pool,
            )

        test_cases = []

        ################# CASE 0 ##########################
        # MTP0, BS=1, 1 context request
        batch_size = 1
        num_nextn_predict_layers = 1
        num_contexts = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two da ta,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.int, device="cuda")
        position_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int, device="cuda")

        seq_lens = torch.tensor([5], dtype=torch.int, device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        )  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[0, -1]], dtype=torch.int, device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor([6, 7, 8, 9, 0], dtype=torch.int, device="cuda")
        ref_previous_hidden_states = previous_layer_hidden_states
        attn_metadata = None

        test_cases += [
            [
                "case0_is_thop_false",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                False,
            ]
        ]

        test_cases += [
            [
                "case0_is_thop_true",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                True,
            ]
        ]

        ################# CASE 1 ##########################
        # MTP0, BS=3, 3 context request
        batch_size = 3
        num_contexts = 3
        num_nextn_predict_layers = 1
        hidden_size = 16
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        (
            mtp_past_hidden_states_ptrs_v1,
            mtp_past_tokens_ptrs_v1,
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)
        (
            mtp_past_hidden_states_ptrs_v2,
            mtp_past_tokens_ptrs_v2,
            mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2,
        ) = gen_data(batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor(
            [
                5,
                6,
                7,
                8,
                9,  # request0
                10,
                11,
                12,
                13,  # request1
                20,
                21,
                22,
                23,
                24,
                25,  # request2
            ],
            dtype=torch.int,
            device="cuda",
        )
        position_ids = torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,  # request0
                    0,
                    1,
                    2,
                    3,  # request1
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,  # request2
                ]
            ],
            dtype=torch.int,
            device="cuda",
        )

        seq_lens = torch.tensor([5, 4, 6], dtype=torch.int, device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (len(input_ids), hidden_size), dtype=torch.float32, device="cuda"
        )  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[0, -1], [1, -1], [2, -1]], dtype=torch.int, device="cuda")
        num_accepted_tokens = torch.tensor([1, 1, 1], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor(
            [6, 7, 8, 9, 0, 11, 12, 13, 1, 21, 22, 23, 24, 25, 2], dtype=torch.int, device="cuda"
        )
        ref_previous_hidden_states = previous_layer_hidden_states

        attn_metadata = None

        test_cases += [
            [
                "case1_is_thop_false",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                False,
            ]
        ]

        test_cases += [
            [
                "case1_is_thop_true",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v2, device="cuda"),
                mtp_hidden_states_tensor_pool_v2,
                mtp_tokens_tensor_pool_v2,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                True,
            ]
        ]

        ################## CASE 2 ##########################
        # BS=1, 1 generation request, num_nextn_predict_layers = 1
        batch_size = 1
        num_contexts = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        mtp_past_hidden_states_ptrs_v1 = []
        mtp_past_tokens_ptrs_v1 = []
        mtp_hidden_states_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers, hidden_size), device="cuda", dtype=torch.float32
        )
        mtp_tokens_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers), device="cuda", dtype=torch.int
        )

        mtp_hidden_states_tensor_pool_v1[0] = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], device="cuda", dtype=torch.float32
        )
        mtp_past_hidden_states_ptrs_v1.append(mtp_hidden_states_tensor_pool_v1[0].data_ptr())
        mtp_tokens_tensor_pool_v1[0] = torch.tensor([42], device="cuda", dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[0].data_ptr())

        input_ids = torch.tensor([6, 42], dtype=torch.int, device="cuda")

        position_ids = torch.tensor([10, 11], dtype=torch.int, device="cuda")

        # already '-1' in 'update_mtp_hidden_states'
        seq_lens = torch.tensor([2], dtype=torch.int, device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (len(input_ids), hidden_size), dtype=torch.float32, device="cuda"
        )  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[43, -1]], dtype=torch.int, device="cuda")

        num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor([43], dtype=torch.int, device="cuda")

        ref_previous_hidden_states = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], device="cuda", dtype=torch.float32
        )

        attn_metadata = None

        test_cases += [
            [
                "case2_is_thop_false",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                False,
            ]
        ]

        test_cases += [
            [
                "case2_is_thop_true",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                True,
            ]
        ]

        ################## CASE 3 ##########################
        # BS=3, 3 generation request, num_nextn_predict_layers = 3
        batch_size = 3
        num_contexts = 0
        num_nextn_predict_layers = 3
        hidden_size = 12
        request_ids = range(batch_size)

        mtp_past_hidden_states_ptrs_v1 = []
        mtp_past_tokens_ptrs_v1 = []
        mtp_hidden_states_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers, hidden_size), device="cuda", dtype=torch.float32
        )
        mtp_tokens_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers), device="cuda", dtype=torch.int
        )

        mtp_hidden_states_tensor_pool_v1[0] = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        mtp_past_hidden_states_ptrs_v1.append(mtp_hidden_states_tensor_pool_v1[0].data_ptr())

        mtp_hidden_states_tensor_pool_v1[1] = torch.tensor(
            [
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        mtp_past_hidden_states_ptrs_v1.append(mtp_hidden_states_tensor_pool_v1[1].data_ptr())

        mtp_hidden_states_tensor_pool_v1[2] = torch.tensor(
            [
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        mtp_past_hidden_states_ptrs_v1.append(mtp_hidden_states_tensor_pool_v1[2].data_ptr())

        mtp_tokens_tensor_pool_v1[0] = torch.tensor([19, 20, 21], device="cuda", dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[0].data_ptr())

        mtp_tokens_tensor_pool_v1[1] = torch.tensor([29, 30, 31], device="cuda", dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[1].data_ptr())

        mtp_tokens_tensor_pool_v1[2] = torch.tensor([39, 40, 41], device="cuda", dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[2].data_ptr())

        input_ids = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int, device="cuda"
        )  # useless
        position_ids = torch.tensor(
            [10, 11, 12, 13, 21, 22, 23, 24, 32, 33, 34, 35], dtype=torch.int, device="cuda"
        )

        seq_lens = torch.tensor([4, 4, 4], dtype=torch.int, device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (12, hidden_size), device="cuda", dtype=torch.float32
        )  # useless

        accepted_tokens = torch.tensor(
            [[22, -1, -1, -1], [31, 32, -1, -1], [0, 40, 41, 42]], dtype=torch.int, device="cuda"
        )

        num_accepted_tokens = torch.tensor([1, 2, 4], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor(
            [20, 21, 22, 30, 31, 32, 40, 41, 42], dtype=torch.int, device="cuda"
        )

        ref_previous_hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        attn_metadata = None

        test_cases += [
            [
                "case3_is_thop_false",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                False,
            ]
        ]

        test_cases += [
            [
                "case3_is_thop_true",
                num_nextn_predict_layers,
                input_ids,
                position_ids,
                seq_lens,
                torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
                torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
                mtp_hidden_states_tensor_pool_v1,
                mtp_tokens_tensor_pool_v1,
                previous_layer_hidden_states,
                request_ids,
                num_contexts,
                accepted_tokens,
                num_accepted_tokens,
                attn_metadata,
                ref_input_ids,
                ref_previous_hidden_states,
                True,
            ]
        ]

        return test_cases

    @parameterized.expand(load_prepare_drafter_inputs_test_cases, name_func=unittest_name_func)
    def test_prepare_drafter_inputs(
        self,
        test_case_name,
        num_nextn_predict_layers,
        input_ids,
        position_ids,
        seq_lens,
        mtp_past_hidden_states_ptrs,
        mtp_past_tokens_ptrs,
        mtp_hidden_states_tensor_pool,
        mtp_tokens_tensor_pool,
        previous_layer_hidden_states,
        request_ids,
        num_contexts,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        ref_input_ids,
        ref_previous_hidden_states,
        is_thop,
    ):
        batch_size = len(request_ids)
        if previous_layer_hidden_states is not None:
            hidden_size = previous_layer_hidden_states.shape[1]
        else:
            hidden_size = 10
        spec_config = MTPDecodingConfig(max_draft_len=num_nextn_predict_layers)
        # Simulate update_spec_config_from_model_config for vanilla MTP tests
        spec_config.num_nextn_predict_layers = num_nextn_predict_layers

        if attn_metadata is None:
            attn_metadata = TrtllmAttentionMetadata(
                max_num_requests=batch_size, max_num_tokens=1024, kv_cache_manager=None
            )
            attn_metadata.seq_lens = seq_lens.to("cpu")
            attn_metadata.num_contexts = num_contexts
            # dummy kv cache param
            attn_metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[10] * batch_size
            )

        spec_manager = MTPHiddenStatesManager(
            config=spec_config,
            dtype=torch.float32,
            hidden_size=hidden_size,
            max_num_requests=batch_size,
        )
        for i in range(batch_size):
            # for the generation requests, we also need to manually add slot
            # because these generation requests are also first use
            spec_manager.slot_manager.add_slot(request_ids[i])

        spec_metadata = MTPSpecMetadata(
            max_num_requests=32,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_draft_len=num_nextn_predict_layers,
            max_total_draft_tokens=num_nextn_predict_layers,
            mtp_num_modules=num_nextn_predict_layers,
            mtp_hidden_states_manager=spec_manager,
        )
        spec_metadata.request_ids = request_ids
        spec_metadata.mtp_hidden_states_ptrs = mtp_past_hidden_states_ptrs
        spec_metadata.mtp_past_tokens_ptrs = mtp_past_tokens_ptrs

        spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool = (
            mtp_hidden_states_tensor_pool
        )
        spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool = mtp_tokens_tensor_pool
        spec_metadata.prepare()

        mtpworker = MTPWorker(spec_config)
        mtpworker.is_thop = is_thop
        draft_inputs = mtpworker.prepare_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=previous_layer_hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            spec_metadata=spec_metadata,
            attn_metadata=attn_metadata,
        )

        torch.testing.assert_close(draft_inputs["input_ids"], ref_input_ids)
        torch.testing.assert_close(draft_inputs["hidden_states"], ref_previous_hidden_states)
