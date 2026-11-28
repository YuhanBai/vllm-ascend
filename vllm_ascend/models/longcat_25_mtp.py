# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight replicated 3-layer MTP predictor for LongCat-2.5 on Ascend."""

from collections.abc import Iterable
from functools import partial

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.models.longcat_flash import FlashMLP, scale_mla_lora_norms_on_load
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.longcat_25 import LongCat25IndexerNorm, LongCat25KeyProjection
from vllm_ascend.models.longcat_25_ops import get_longcat_lightning_indexer, select_longcat_topk


class LongCat25MTPLayer(nn.Module):
    """Single dense MLA + MLP predictor layer replicated across MTP steps."""

    def __init__(self, vllm_config: VllmConfig, prefix: str, topk_buffer: torch.Tensor, indexer_op):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_buffer,
        )
        scale_mla_lora_norms_on_load(self.self_attn, config)
        wrapper = self.self_attn.mla_attn.mla_attn.impl.indexer
        key_projection = LongCat25KeyProjection(config)
        key_norm = LongCat25IndexerNorm(config.index_head_dim)
        for indexer in (self.self_attn.indexer, wrapper, wrapper.impl):
            indexer.wk_weights_proj = key_projection
            indexer.k_norm = key_norm
        wrapper.impl.topk_selector = partial(
            select_longcat_topk,
            op=indexer_op,
            init_tokens=config.index_init_tokens,
            local_tokens=config.index_local_tokens,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FlashMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        enormed = self.enorm(inputs_embeds)
        hnormed = self.hnorm(previous_hidden_states)
        hidden_states, _ = self.eh_proj(torch.cat((enormed, hnormed), dim=-1))

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


def normalize_mtp_weight_name(name: str) -> str | None:
    """Map official LongCat-2.5 ``model.mtp.*`` keys to ``AscendLongCat25MTP`` parameters."""
    if not name.startswith("model.mtp."):
        return None
    suffix = name.removeprefix("model.mtp.")
    # MTP disables the over-tokenizer (mtp_disable_over_tokenizer=true).
    if suffix.startswith("oe_embed_"):
        return None
    if suffix == "embed_tokens.weight":
        return "model.embed_tokens.weight"
    if suffix == "norm.weight":
        return "model.norm.weight"
    if not suffix.startswith("layers.0."):
        return None
    layer_suffix = suffix.removeprefix("layers.0.")
    layer_suffix = layer_suffix.replace("enorm.m.", "enorm.")
    layer_suffix = layer_suffix.replace("hnorm.m.", "hnorm.")
    layer_suffix = layer_suffix.replace("transformer_layer.mlp.", "mlp.")
    layer_suffix = layer_suffix.replace(".indexer.wk.", ".indexer.wk_weights_proj.wk.")
    layer_suffix = layer_suffix.replace(".indexer.weights_proj.", ".indexer.wk_weights_proj.weights_proj.")
    return f"model.mtp_layer.{layer_suffix}"


class LongCat25MTPModel(nn.Module):
    """MTP backbone reusing a single dense layer across ``mtp_num_layers`` draft steps."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.do_not_compile = True
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.num_mtp_layers = getattr(config, "mtp_num_layers", 3)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        topk_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens, config.index_topk),
            dtype=torch.int32,
            device=vllm_config.device_config.device,
        )
        indexer_op = get_longcat_lightning_indexer(
            use_triton=(vllm_config.additional_config or {}).get("longcat_indexer_triton", False)
        )
        self.mtp_layer = LongCat25MTPLayer(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "layers.0"),
            topk_buffer=topk_buffer,
            indexer_op=indexer_op,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.mtp_layer(inputs_embeds, hidden_states, positions)
        return self.norm(hidden_states)


class AscendLongCat25MTP(nn.Module):
    """LongCat-2.5 lightweight 3-step MTP drafter sharing target lm_head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.do_not_compile = True
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.has_own_lm_head = False
        self.model = LongCat25MTPModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        loaded: set[str] = set()
        for orig_name, loaded_weight in weights:
            if orig_name == "lm_head.weight" and "lm_head.weight" in params_dict:
                param = params_dict["lm_head.weight"]
                getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
                loaded.add("lm_head.weight")
                continue
            mapped = normalize_mtp_weight_name(orig_name)
            if mapped is None:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in mapped:
                    target = mapped.replace(weight_name, param_name)
                    if target in params_dict:
                        param = params_dict[target]
                        param.weight_loader(param, loaded_weight, shard_id)
                        loaded.add(target)
                    break
            else:
                if mapped in params_dict:
                    param = params_dict[mapped]
                    getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
                    loaded.add(mapped)
        return loaded
