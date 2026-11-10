# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Iterable
from copy import copy
from functools import partial

import torch
import torch_npu
from torch import nn
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
from vllm.model_executor.models.longcat_flash import (
    FlashDecoderLayer,
    FlashMLP,
    FlashModel,
    LongcatFlashForCausalLM,
    LongcatRouter,
    scale_mla_lora_norms_on_load,
)
from vllm.model_executor.models.longcat_flash_ngram import NgramEmbedding
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

from vllm_ascend.models.longcat_25_ops import (
    get_longcat_lightning_indexer,
    route_experts,
    select_longcat_topk,
)
from vllm_ascend.models.longcat_25_state import LongCat25ModelState
from vllm_ascend.utils import enable_sp

_OE_WEIGHT = re.compile(r"oe_embed_(tokens|proj)([0-9]+)\.weight")


class LongCat25Embedding(NgramEmbedding):
    def __init__(self, config, word_embeddings):
        super().__init__(config, word_embeddings)
        ignore = torch.zeros(config.vocab_size, dtype=torch.bool)
        ignore[config.oe_ignored_token_ids] = True
        self.register_buffer("ignore_tokens", ignore, persistent=False)

    def embed_batched(self, input_ids, oe_ids):
        word = self.word_embeddings(input_ids)
        oe = self.oe_embedder(oe_ids.T.contiguous())
        # The BF16 projection weights have already been scaled at load time.
        combined = word / (1 + self.num_embedders) + torch.bmm(oe, self.oe_projection).sum(dim=0)
        return torch.where(self.ignore_tokens[input_ids, None], word, combined)


class LongCat25KeyProjection(nn.Module):
    """Keep the checkpoint's BF16 key and FP32 scoring projections separate."""

    def __init__(self, config):
        super().__init__()
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        key = self.wk(hidden_states)
        weights = self.weights_proj(hidden_states.float()).to(key.dtype)
        return torch.cat((key, weights), dim=-1), None


class LongCat25IndexerNorm(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x):
        normalized, _ = torch_npu.npu_rms_norm(x.float(), self.weight.float(), 1e-6)
        return normalized.to(x.dtype)


class LongCat25MoE(nn.Module):
    def __init__(self, config, prefix, enable_eplb):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.top_k = config.moe_topk
        self.scale = config.routed_scaling_factor
        self.router = LongcatRouter(config, config.zero_expert_num, torch.float32, prefix=f"{prefix}.router")
        self.experts = FusedMoEFactory(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=False,
            custom_routing_function=self.route_real_experts,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
            enable_eplb=enable_eplb,
        )

    def route_real_experts(self, hidden_states, gating_output, topk, renormalize):
        weights, ids, _ = self._route(gating_output, topk)
        return weights, ids

    def _route(self, logits, topk):
        return route_experts(logits, self.router.e_score_correction_bias, topk, self.num_experts, self.scale)

    def forward(self, hidden_states):
        logits = self.router(hidden_states)
        _, _, identity_weights = self._route(logits, self.top_k)
        # Add identity slots after FusedMoE's TP/EP reductions, once per token.
        identity = hidden_states * identity_weights.to(hidden_states.dtype).sum(dim=-1, keepdim=True)
        return self.experts(hidden_states=hidden_states, router_logits=logits) + identity


class LongCat25DecoderLayer(FlashDecoderLayer):
    """LongCat shortcut block with one indexer shared by two MLA sublayers."""

    def __init__(self, vllm_config, prefix, topk_buffer, indexer_op):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.rsplit(".", 1)[-1])
        consumer_config = copy(config)
        # DeepSeek's skip pattern is indexed by decoder block, not by attention
        # sublayer. Give only the second MLA instance the skip pattern.
        consumer_config.index_topk_pattern = "S" * config.num_layers
        self.self_attn = nn.ModuleList(
            DeepseekV2MLAAttention(
                vllm_config=vllm_config,
                config=attn_config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=vllm_config.cache_config,
                # vLLM's sparse MLA extracts one integer from its runtime
                # name. ModuleList keeps the checkpoint's self_attn.{i} path.
                prefix=f"{prefix}.self_attn_{i}",
                topk_indices_buffer=topk_buffer,
            )
            for i, attn_config in enumerate((config, consumer_config))
        )
        for attention in self.self_attn:
            scale_mla_lora_norms_on_load(attention, config)

        producer = self.self_attn[0]
        wrapper = producer.mla_attn.mla_attn.impl.indexer
        key_projection = LongCat25KeyProjection(config)
        key_norm = LongCat25IndexerNorm(config.index_head_dim)
        for indexer in (producer.indexer, wrapper, wrapper.impl):
            indexer.wk_weights_proj = key_projection
            indexer.k_norm = key_norm
        wrapper.impl.topk_selector = partial(
            select_longcat_topk,
            op=indexer_op,
            init_tokens=config.index_init_tokens,
            local_tokens=config.index_local_tokens,
        )

        self.input_layernorm = nn.ModuleList(RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2))
        self.post_attention_layernorm = nn.ModuleList(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        )
        self.mlps = nn.ModuleList(
            FlashMLP(config.hidden_size, config.intermediate_size, config.hidden_act, prefix=f"{prefix}.mlps.{i}")
            for i in range(2)
        )
        self.mlp = LongCat25MoE(config, f"{prefix}.mlp", vllm_config.parallel_config.enable_eplb)


class LongCat25Model(FlashModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = None
        self.vocab_size = config.vocab_size
        self.ngram_embeddings = None
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
            )
            self.ngram_embeddings = LongCat25Embedding(config, self.embed_tokens)
        else:
            self.embed_tokens = PPMissingLayer()

        topk_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens, config.index_topk),
            dtype=torch.int32,
            device=vllm_config.device_config.device,
        )
        indexer_op = get_longcat_lightning_indexer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_layers,
            lambda prefix: LongCat25DecoderLayer(vllm_config, prefix, topk_buffer, indexer_op),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if get_pp_group().is_last_rank else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()

        def language_weights():
            for name, weight in weights:
                match = _OE_WEIGHT.fullmatch(name)
                if match is not None:
                    if self.ngram_embeddings is not None:
                        kind, index = match.groups()
                        part = "embedders" if kind == "tokens" else "post_projs"
                        if kind == "proj":
                            weight = weight / (1 + self.ngram_embeddings.num_embedders)
                        loaded.add(self.ngram_embeddings.load_weight(f"ngram_embeddings.{part}.{index}.weight", weight))
                    continue
                name = name.replace(".indexer.wk.", ".indexer.wk_weights_proj.wk.")
                name = name.replace(".indexer.weights_proj.", ".indexer.wk_weights_proj.weights_proj.")
                yield name, weight

        loaded.update(super().load_weights(language_weights()))
        return loaded


class AscendLongCat25ForCausalLM(LongcatFlashForCausalLM):
    """BF16 text generation from the official LongCat-2.5 checkpoint."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.quant_config = None
        self._validate_config(vllm_config)
        self.model = LongCat25Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    @staticmethod
    def _validate_config(vllm_config):
        model = vllm_config.model_config
        config = model.hf_config
        parallel = vllm_config.parallel_config
        if (
            model.dtype != torch.bfloat16
            or vllm_config.quant_config is not None
            or getattr(config, "compression_config", None)
        ):
            raise ValueError("LongCat-2.5 on Ascend currently requires the original BF16 checkpoint.")
        if (config.cli_factor, config.index_k_norm_type, config.zero_expert_type) != (2, "rms", "identity"):
            raise ValueError("Expected LongCat-2.5 paired MLA, RMS index normalization and identity experts.")
        compilation = vllm_config.compilation_config
        if not model.enforce_eager and (
            compilation.mode != CompilationMode.NONE
            or compilation.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY)
        ):
            raise ValueError(
                "LongCat-2.5 requires --enforce-eager or "
                '--compilation-config \'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}\'.'
            )
        if (
            vllm_config.speculative_config is not None
            or vllm_config.kv_transfer_config is not None
            or parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
            or enable_sp(vllm_config)
        ):
            raise ValueError(
                "LongCat-2.5 text inference requires local KV and disables speculation, "
                "context parallelism and sequence parallelism."
            )
        if vllm_config.cache_config.cache_dtype not in (
            "auto",
            "bfloat16",
            "bf16",
        ) or vllm_config.attention_config.indexer_kv_dtype not in ("auto", "bf16"):
            raise ValueError("LongCat-2.5 requires BF16 MLA and indexer KV caches.")

    @staticmethod
    def get_model_state_cls() -> type[LongCat25ModelState]:
        return LongCat25ModelState

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        oe_names = set()

        def track_oe_weights():
            for name, weight in weights:
                if _OE_WEIGHT.fullmatch(name.removeprefix("model.")):
                    oe_names.add(name)
                yield name, weight

        loader = AutoWeightsLoader(
            self, skip_prefixes=["model.mtp.", "model.visual_model.", "model.visual_bridge_model."]
        )
        loaded = loader.load_weights(track_oe_weights())
        if self.model.ngram_embeddings is not None:
            expected = {
                f"model.oe_embed_{kind}{i}.weight"
                for kind in ("tokens", "proj")
                for i in range(self.model.ngram_embeddings.num_embedders)
            }
            if missing := expected - oe_names:
                raise ValueError(f"LongCat-2.5 checkpoint is missing OE weights: {sorted(missing)}")
        return loaded
