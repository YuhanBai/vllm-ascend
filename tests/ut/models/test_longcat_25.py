# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_ascend.models.longcat_25_config import LongCat25Config
from vllm_ascend.models.longcat_25_ops import over_embedding_ids, route_experts, select_longcat_topk


def test_config_normalizes_checkpoint_fields():
    config = LongCat25Config(
        architectures=["LongcatCausalLM"],
        auto_map={"AutoConfig": "missing.LongcatConfig"},
        num_layers=3,
        ffn_hidden_size=128,
        expert_ffn_hidden_size=64,
        oe_ignore_tokens="3:5,8,4",
        rope_scaling={"rope_type": "deepseek_yarn", "factor": 120},
    )
    restored = LongCat25Config.from_dict(config.to_dict())
    assert restored.num_hidden_layers == 3
    assert restored.intermediate_size == 128
    assert restored.moe_intermediate_size == 64
    assert restored.oe_ignored_token_ids == [3, 4, 8]
    assert restored.rope_parameters["rope_type"] == "deepseek_yarn"
    assert "auto_map" not in restored.to_dict()


def test_over_embedding_stops_at_previous_eos_and_start_of_request():
    current = torch.tensor([3, 2, 3])
    history = torch.tensor([[2, 7], [6, 7], [-1, -1]])
    weights = torch.tensor([[[1, 10, 9]], [[1, 10, 5]]], dtype=torch.int32)
    moduli = torch.tensor([[13], [19]], dtype=torch.int32)
    offsets = torch.tensor([0, 13, 32], dtype=torch.int32)
    # The bigram ignores its third coefficient. An EOS in the current
    # position still hashes with its history, unlike an EOS in the history.
    expected = torch.tensor([[3, 16], [62 % 13, 13 + 97 % 19], [3, 16]], dtype=torch.int32)
    torch.testing.assert_close(over_embedding_ids(current, history, 2, weights, moduli, offsets), expected)


def test_identity_routes_are_removed_from_dispatch_without_renormalizing():
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0]])
    bias = torch.tensor([0.0, 0.0, 0.0, 1.0])
    weights, ids, identity = route_experts(logits, bias, topk=2, num_experts=3, scale=9)
    probs = logits.softmax(dim=-1) * 9
    assert ids.tolist() == [[0, 0]]
    torch.testing.assert_close(weights.sum(dim=-1), probs[:, 0])
    torch.testing.assert_close(identity.sum(dim=-1), probs[:, 3])


def test_bf16_indexer_passes_mandatory_token_counts():
    query = torch.zeros((3, 32, 128), dtype=torch.bfloat16)
    key = torch.zeros((2, 128, 1, 128), dtype=torch.bfloat16)
    indexer = SimpleNamespace(topk_tokens=2048, k_cache=SimpleNamespace(kv_cache=(key,)))
    metadata = SimpleNamespace(
        actual_seq_lengths_query=torch.tensor([2, 3], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([2, 160], dtype=torch.int32),
        block_table=torch.tensor([[0, -1], [0, 1]], dtype=torch.int32),
    )
    selected = torch.zeros((3, 1, 2048), dtype=torch.int32)

    def indexer_op(**kwargs):
        assert kwargs["key"] is key
        assert kwargs["weights"].dtype == torch.bfloat16
        assert kwargs["sparse_count"] == 2048
        assert kwargs["init_tensor"].tolist() == [16, 16]
        assert kwargs["local_tensor"].tolist() == [1024, 1024, 1024]
        return selected, None

    assert (
        select_longcat_topk(
            query, torch.zeros(3, 32), indexer, metadata, None, op=indexer_op, init_tokens=16, local_tokens=1024
        )
        is selected
    )
