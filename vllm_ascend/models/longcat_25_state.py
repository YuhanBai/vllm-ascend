# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import functional as F
from vllm.config import CUDAGraphMode

from vllm_ascend.models.longcat_25_ops import over_embedding_ids
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class LongCat25ModelState(AscendModelState):
    supports_prompt_embeds = False

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.embedding = model.model.ngram_embeddings
        self._graph_embeds = None
        compilation = vllm_config.compilation_config
        if (
            self.embedding is not None
            and not self.model_config.enforce_eager
            and compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
            and (capacity := compilation.max_cudagraph_capture_size)
        ):
            self._graph_embeds = torch.zeros((capacity, self.inputs_embeds_size), dtype=self.dtype, device=self.device)

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int):
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self._graph_embeds is not None:
            # FULL replay does not call forward with the tensors returned by
            # prepare_inputs. Capture must read the exact storage it updates.
            inputs["inputs_embeds"] = self._graph_embeds[:num_tokens]
        return inputs

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        if self.embedding is None:
            return inputs

        num_tokens = input_batch.num_tokens
        num_padded = input_batch.num_tokens_after_padding
        graph_buffer = self._graph_embeds
        graph_embeds = (
            graph_buffer[:num_padded] if graph_buffer is not None and num_padded <= len(graph_buffer) else None
        )
        if input_batch.is_dummy or num_tokens == 0:
            if graph_embeds is not None:
                inputs["inputs_embeds"] = graph_embeds.zero_()
            return inputs

        embeds = self._compute_over_embeddings(input_batch, req_states)
        if graph_embeds is None:
            inputs["inputs_embeds"] = F.pad(embeds, (0, 0, 0, num_padded - num_tokens))
        else:
            graph_embeds[:num_tokens].copy_(embeds)
            graph_embeds[num_tokens:].zero_()
            inputs["inputs_embeds"] = graph_embeds
        return inputs

    def _compute_over_embeddings(self, input_batch, req_states):
        embedding = self.embedding
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        input_ids = input_batch.input_ids[:num_tokens].long()
        offsets = input_batch.query_start_loc[: num_reqs + 1]
        rows = torch.repeat_interleave(
            input_batch.idx_mapping[:num_reqs].long(), offsets.diff(), output_size=num_tokens
        )
        distances = torch.arange(1, embedding.n, device=input_ids.device)
        previous_positions = input_batch.positions[:num_tokens, None] - distances
        # Read the scheduler's canonical history. Prefix hits, resumed requests
        # and a reused request slot require no separate OE state to be repaired.
        history = req_states.all_token_ids.gpu[rows[:, None], previous_positions.clamp_min(0)]
        history = history.masked_fill(previous_positions < 0, -1)
        oe_ids = over_embedding_ids(
            input_ids, history, embedding.eos_id, embedding.ne_weights, embedding.ne_mods, embedding.exclusive_sizes
        )
        return embedding.embed_batched(input_ids, oe_ids)
