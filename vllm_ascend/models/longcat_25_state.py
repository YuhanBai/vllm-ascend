# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn import functional as F
from vllm.config import CUDAGraphMode
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.models.longcat_25_ops import over_embedding_ids
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class LongCat25ModelState(AscendModelState):
    supports_prompt_embeds = False

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self.embedding = model.model.ngram_embeddings
        self._over_embedding_ids_from_token_table = None
        if (
            HAS_TRITON
            and self.device.type == "npu"
            and get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        ):
            from vllm_ascend.ops.triton.over_embedding_ids import over_embedding_ids_from_token_table

            self._over_embedding_ids_from_token_table = over_embedding_ids_from_token_table
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
        if self.embedding is None or self.supports_mm_inputs:
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
        inputs["inputs_embeds"] = self._pad_embeddings(embeds, num_padded, graph_embeds)
        return inputs

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        if not self.supports_mm_inputs or self.embedding is None:
            return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)

        self.execute_mm_encoder(scheduled_encoder_inputs)
        multimodal_embeddings, is_multimodal = self.gather_mm_embeddings(input_batch)
        self._mm_skip_mask = is_multimodal if multimodal_embeddings else None
        try:
            embeds = self._compute_over_embeddings(input_batch, req_states)
        finally:
            self._mm_skip_mask = None
        if multimodal_embeddings:
            embeds = _merge_multimodal_embeddings(embeds, multimodal_embeddings, is_multimodal)

        num_padded = input_batch.num_tokens_after_padding
        graph_buffer = self._graph_embeds
        graph_embeds = (
            graph_buffer[:num_padded] if graph_buffer is not None and num_padded <= len(graph_buffer) else None
        )
        return self._pad_embeddings(embeds, num_padded, graph_embeds)

    @staticmethod
    def _pad_embeddings(embeds, num_padded, graph_embeds):
        num_tokens = embeds.shape[0]
        if graph_embeds is None:
            return F.pad(embeds, (0, 0, 0, num_padded - num_tokens))
        graph_embeds[:num_tokens].copy_(embeds)
        graph_embeds[num_tokens:].zero_()
        return graph_embeds

    def _compute_over_embeddings(self, input_batch, req_states):
        embedding = self.embedding
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        input_ids = input_batch.input_ids[:num_tokens].long()
        positions = input_batch.positions[:num_tokens]
        rows = input_batch.idx_mapping[:num_reqs]
        compute_from_token_table = self._over_embedding_ids_from_token_table
        skip_mask = getattr(self, "_mm_skip_mask", None)
        # Equal batch counts alone can also occur with empty or multi-token requests.
        offsets_cpu = getattr(input_batch, "query_start_loc_np", None)
        if (
            skip_mask is not None
            or compute_from_token_table is None
            or num_tokens != num_reqs
            or offsets_cpu is None
            or not ((offsets_cpu[1 : num_reqs + 1] - offsets_cpu[:num_reqs]) == 1).all()
        ):
            offsets = input_batch.query_start_loc[: num_reqs + 1]
            rows = torch.repeat_interleave(rows.long(), offsets.diff(), output_size=num_tokens)

        text_indices = None
        if skip_mask is not None:
            text_indices = (~skip_mask[:num_tokens]).nonzero(as_tuple=False).squeeze(-1)
            if text_indices.numel() == 0:
                return torch.empty((num_tokens, self.inputs_embeds_size), dtype=self.dtype, device=self.device)
            if text_indices.numel() < num_tokens:
                text_indices = text_indices.to(device=input_ids.device, non_blocking=True)
                input_ids = input_ids.index_select(0, text_indices)
                positions = positions.index_select(0, text_indices)
                rows = rows.index_select(0, text_indices)
            else:
                text_indices = None

        # Read the scheduler's canonical history. Prefix hits, resumed requests
        # and a reused request slot require no separate OE state to be repaired.
        token_table = req_states.all_token_ids.gpu
        eos_token_id = getattr(embedding, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = embedding.eos_id
        if compute_from_token_table is not None:
            table_ids = compute_from_token_table(
                input_ids,
                token_table,
                positions,
                rows,
                eos_token_id,
                embedding.ne_weights,
                embedding.ne_mods,
                embedding.exclusive_sizes,
            )
        else:
            distances = torch.arange(1, embedding.n, device=input_ids.device)
            previous_positions = positions[:, None] - distances
            history = token_table[rows[:, None], previous_positions.clamp_min(0)]
            history = history.masked_fill(previous_positions < 0, -1)
            token_ids = over_embedding_ids(
                input_ids, history, eos_token_id, embedding.ne_weights, embedding.ne_mods, embedding.exclusive_sizes
            )
            table_ids = token_ids.T.contiguous()
        text_embeds = embedding.embed_batched(input_ids, table_ids)
        if text_indices is None:
            return text_embeds
        embeds = torch.empty((num_tokens, text_embeds.shape[-1]), dtype=text_embeds.dtype, device=text_embeds.device)
        embeds.index_copy_(0, text_indices, text_embeds)
        return embeds
