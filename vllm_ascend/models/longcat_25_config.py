# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig


class LongCat25Config(PretrainedConfig):
    """Native configuration for the LongCat-2.5 BF16 language model."""

    model_type = "pro_v"

    def __init__(
        self,
        num_layers=38,
        ffn_hidden_size=12288,
        expert_ffn_hidden_size=2048,
        hidden_act="silu",
        oe_vocab_size_ratio=100.567,
        oe_split_num=4,
        oe_neighbor_num=5,
        oe_ignore_tokens="199:224",
        rope_theta=1000000.0,
        rope_scaling=None,
        rope_parameters=None,
        **kwargs,
    ):
        # The published auto_map names Python files absent from the checkpoint.
        kwargs.pop("auto_map", None)
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(**kwargs)
        self.num_layers = self.num_hidden_layers = num_layers
        self.ffn_hidden_size = self.intermediate_size = ffn_hidden_size
        self.expert_ffn_hidden_size = self.moe_intermediate_size = expert_ffn_hidden_size
        self.hidden_act = hidden_act
        # Aliases for vLLM's existing LongCat n-gram embedding and weight loader.
        self.oe_vocab_size_ratio = self.ngram_vocab_size_ratio = oe_vocab_size_ratio
        self.oe_split_num = self.emb_split_num = oe_split_num
        self.oe_neighbor_num = self.emb_neighbor_num = oe_neighbor_num
        self.oe_ignore_tokens = oe_ignore_tokens
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            **(rope_parameters or rope_scaling or {}),
        }
        self.use_index_cache = True

    @property
    def oe_ignored_token_ids(self) -> list[int]:
        ids = set()
        for item in self.oe_ignore_tokens.split(","):
            start, separator, end = item.strip().partition(":")
            if start:
                ids.update(range(int(start), int(end)) if separator else (int(start),))
        return sorted(ids)
