# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.longcat950_indexer import lightning_indexer_with_mandatory_tokens
from vllm_ascend.models.longcat_25_config import LongCat25Config
from vllm_ascend.models.longcat_25_ops import over_embedding_ids, route_experts, select_longcat_topk
from vllm_ascend.models.longcat_25_state import LongCat25ModelState
from vllm_ascend.worker.v2.model_states.default import AscendModelState


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


def test_a5_indexer_keeps_mandatory_positions_and_masks_graph_padding(monkeypatch):
    candidates = torch.tensor([[[0, 1, 2, 3]], [[0, 1, 2, 3]], [[-1, -1, -1, -1]]], dtype=torch.int32)

    def native_indexer(**kwargs):
        assert "init_tensor" not in kwargs
        assert "local_tensor" not in kwargs
        return candidates, None

    monkeypatch.setattr(torch.ops.npu, "npu_lightning_indexer", native_indexer, raising=False)
    indices, _ = lightning_indexer_with_mandatory_tokens(
        torch.ones((3, 1, 1), dtype=torch.bfloat16),
        torch.tensor([9, 8, 7, 6, 5, 4], dtype=torch.bfloat16).reshape(1, 6, 1, 1),
        torch.ones((3, 1), dtype=torch.bfloat16),
        actual_seq_lengths_query=torch.tensor([2], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([6], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=4,
        sparse_mode=3,
        init_tensor=torch.tensor([1], dtype=torch.int32),
        local_tensor=torch.ones(3, dtype=torch.int32),
    )
    assert indices[:, 0, :].sort(dim=-1).values.tolist() == [[0, 1, 2, 4], [0, 1, 2, 5], [-1, -1, -1, -1]]


def test_decode_graph_reuses_over_embedding_storage(monkeypatch):
    def init_parent(self, config, model, encoder_cache, device):
        self.model_config = SimpleNamespace(enforce_eager=False)
        self.inputs_embeds_size = 2
        self.dtype = torch.float32
        self.device = device
        self.supports_mm_inputs = False

    monkeypatch.setattr(AscendModelState, "__init__", init_parent)
    monkeypatch.setattr(AscendModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(AscendModelState, "prepare_inputs", lambda *args: {})
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, max_cudagraph_capture_size=4)
    )
    embedding = SimpleNamespace(
        n=2,
        eos_id=2,
        ne_weights=torch.tensor([[[1, 10]]], dtype=torch.int32),
        ne_mods=torch.tensor([[13]], dtype=torch.int32),
        exclusive_sizes=torch.tensor([0, 13], dtype=torch.int32),
        embed_batched=lambda tokens, ids: torch.stack((tokens, ids[:, 0]), dim=1).float(),
    )
    model = SimpleNamespace(model=SimpleNamespace(ngram_embeddings=embedding))
    state = LongCat25ModelState(config, model, None, torch.device("cpu"))
    captured = state.prepare_dummy_inputs(4, 4)["inputs_embeds"]
    captured.fill_(99)
    batch = SimpleNamespace(
        is_dummy=False,
        num_tokens=2,
        num_tokens_after_padding=4,
        num_reqs=2,
        input_ids=torch.tensor([5, 4, 0, 0]),
        query_start_loc=torch.tensor([0, 1, 2, 3, 4]),
        idx_mapping=torch.tensor([1, 0, 0, 0]),
        positions=torch.tensor([1, 2, 0, 0]),
    )
    requests = SimpleNamespace(all_token_ids=SimpleNamespace(gpu=torch.tensor([[7, 2, 4], [3, 5, 0]])))
    replay = state.prepare_inputs(batch, requests)["inputs_embeds"]
    assert replay.data_ptr() == captured.data_ptr()
    torch.testing.assert_close(captured, torch.tensor([[5.0, 9.0], [4.0, 4.0], [0.0, 0.0], [0.0, 0.0]]))

    # A later request refreshes the captured rows and clears previously live rows.
    del embedding.eos_id
    embedding.eos_token_id = 2
    batch.num_tokens = batch.num_reqs = 1
    batch.input_ids[0] = 6
    replay = state.prepare_inputs(batch, requests)["inputs_embeds"]
    assert replay.data_ptr() == captured.data_ptr()
    torch.testing.assert_close(captured, torch.tensor([[6.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    # Longer eager prefill batches use their own storage.
    batch.num_tokens_after_padding = 6
    eager = state.prepare_inputs(batch, requests)["inputs_embeds"]
    assert eager.shape == (6, 2)
    assert eager.data_ptr() != captured.data_ptr()


def test_image_embeddings_replace_only_placeholders_in_over_embedding_buffer(monkeypatch):
    state = object.__new__(LongCat25ModelState)
    state.embedding = object()
    state.supports_mm_inputs = True
    state._graph_embeds = torch.full((4, 2), 99.0)
    calls = []
    monkeypatch.setattr(state, "execute_mm_encoder", calls.append)
    monkeypatch.setattr(
        state, "gather_mm_embeddings", lambda batch: ([torch.tensor([[8.0, 9.0]])], torch.tensor([False, True, False]))
    )
    monkeypatch.setattr(
        state, "_compute_over_embeddings", lambda batch, reqs: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    )
    monkeypatch.setattr(AscendModelState, "prepare_inputs", lambda *args: {})
    batch = SimpleNamespace(num_tokens_after_padding=4)
    scheduled = {"image-request": [0]}
    result = state.prepare_inputs_embeds(scheduled, batch, None)
    assert calls == [scheduled]
    assert result.data_ptr() == state._graph_embeds.data_ptr()
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [8.0, 9.0], [5.0, 6.0], [0.0, 0.0]]))
    assert state.prepare_inputs(batch, None) == {}
