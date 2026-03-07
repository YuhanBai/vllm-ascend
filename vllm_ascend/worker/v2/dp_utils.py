# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.v1.worker.gpu.dp_utils import get_cudagraph_and_dp_padding as _get_cudagraph_and_dp_padding

from vllm_ascend.ascend_forward_context import MoECommType, select_moe_comm_method
from vllm_ascend.utils import is_moe_model

if TYPE_CHECKING:
    pass

_ascend_dp_config: threading.local = threading.local()


def set_ascend_dp_config(vllm_config: VllmConfig, is_kv_consumer: bool = False) -> None:
    """Set the Ascend DP configuration for the current thread.

    This function should be called during NPUModelRunner initialization to
    enable Ascend-specific DP optimizations.
    """
    _ascend_dp_config.vllm_config = vllm_config
    _ascend_dp_config.is_kv_consumer = is_kv_consumer


def get_ascend_dp_config() -> tuple[VllmConfig | None, bool]:
    """Get the Ascend DP configuration for the current thread.

    Returns:
        tuple[VllmConfig | None, bool]: (vllm_config, is_kv_consumer)
    """
    return (
        getattr(_ascend_dp_config, 'vllm_config', None),
        getattr(_ascend_dp_config, 'is_kv_consumer', False),
    )


@contextlib.contextmanager
def ascend_dp_config_context(vllm_config: VllmConfig, is_kv_consumer: bool = False):
    """Context manager for Ascend DP configuration.

    This is useful for scenarios where you need to temporarily set the config.
    """
    old_config = get_ascend_dp_config()
    set_ascend_dp_config(vllm_config, is_kv_consumer)
    try:
        yield
    finally:
        if old_config[0] is not None:
            set_ascend_dp_config(old_config[0], old_config[1])
        else:
            _ascend_dp_config.__dict__.clear()


def skip_all_reduce_across_dp_group(vllm_config: VllmConfig, is_kv_consumer: bool = False) -> bool:
    """
    Decide whether to skip the all-reduce across the data-parallel (DP) group.

    Skipping is applicable for all dense models and for moe models only on ranks
    that act as KV consumers.
    """
    if vllm_config.parallel_config.data_parallel_size == 1:
        return True

    if not is_moe_model(vllm_config):
        return True

    if not is_kv_consumer:
        return False

    def needs_mc2(num_tokens: int) -> bool:
        return select_moe_comm_method(num_tokens, vllm_config) in {
            MoECommType.MC2,
            MoECommType.FUSED_MC2,
        }

    compilation_config = vllm_config.compilation_config
    scheduler_config = vllm_config.scheduler_config
    ascend_config = vllm_config.quant_config

    if compilation_config.cudagraph_capture_sizes:
        potential_max_tokens = compilation_config.max_cudagraph_capture_size
    else:
        uniform_decode_query_len = 1
        if vllm_config.speculative_config is not None:
            uniform_decode_query_len += vllm_config.speculative_config.num_speculative_tokens
        potential_max_tokens = scheduler_config.max_num_seqs * uniform_decode_query_len

    decode_must_use_mc2 = needs_mc2(potential_max_tokens)
    prefill_must_use_mc2 = needs_mc2(scheduler_config.max_num_batched_tokens)

    recompute_scheduler_enable = False
    if ascend_config is not None and hasattr(ascend_config, 'recompute_scheduler_enable'):
        recompute_scheduler_enable = ascend_config.recompute_scheduler_enable

    return decode_must_use_mc2 and (prefill_must_use_mc2 or recompute_scheduler_enable)


def get_cudagraph_and_dp_padding(
    num_tokens: int,
    cudagraph_size: int | None,
    cudagraph_runtime_mode: int,
    dp_size: int,
    dp_rank: int,
    skip_all_reduce: bool = False,
) -> tuple[int, torch.Tensor | None, int]:
    """
    Ascend-specific version of get_cudagraph_and_dp_padding.

    This function extends the vLLM implementation with an option to skip
    the all-reduce operation for dense models or specific MoE scenarios.

    Args:
        num_tokens: Number of tokens in the current batch.
        cudagraph_size: The cudagraph size for this batch, or None if not applicable.
        cudagraph_runtime_mode: The cudagraph runtime mode (0=NONE, 1=PIECEWISE, 2=FULL).
        dp_size: Data parallel size.
        dp_rank: Data parallel rank.
        skip_all_reduce: Whether to skip the all-reduce operation.

    Returns:
        tuple[int, torch.Tensor | None, int]:
            - num_tokens_after_padding: Number of tokens after padding.
            - num_tokens_across_dp: Tensor of token counts across DP ranks, or None.
            - synced_cudagraph_mode: Synchronized cudagraph mode across DP ranks.
    """
    if dp_size == 1:
        if cudagraph_size is not None:
            return cudagraph_size, None, cudagraph_runtime_mode
        else:
            return num_tokens, None, cudagraph_runtime_mode

    if skip_all_reduce:
        num_tokens_after_padding = cudagraph_size if cudagraph_size is not None else num_tokens
        num_tokens_across_dp = torch.full(
            (dp_size,), num_tokens_after_padding, dtype=torch.int32, device="cpu"
        )
        return num_tokens_after_padding, num_tokens_across_dp, cudagraph_runtime_mode

    return _get_cudagraph_and_dp_padding(
        num_tokens,
        cudagraph_size,
        cudagraph_runtime_mode,
        dp_size,
        dp_rank,
    )


def get_cudagraph_and_dp_padding_for_ascend(
    num_tokens: int,
    cudagraph_size: int | None,
    cudagraph_runtime_mode: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[int, torch.Tensor | None, int]:
    """
    Ascend-optimized version that automatically determines skip_all_reduce.

    This function is designed to be used as a drop-in replacement for
    vllm's get_cudagraph_and_dp_padding. It reads the configuration from
    thread-local storage set by set_ascend_dp_config.

    This allows the parent GPUModelRunner.execute_model to use Ascend-specific
    DP optimizations without needing to override the entire method.
    """
    vllm_config, is_kv_consumer = get_ascend_dp_config()
    if vllm_config is not None:
        skip_all_reduce = skip_all_reduce_across_dp_group(vllm_config, is_kv_consumer)
    else:
        skip_all_reduce = False

    return get_cudagraph_and_dp_padding(
        num_tokens,
        cudagraph_size,
        cudagraph_runtime_mode,
        dp_size,
        dp_rank,
        skip_all_reduce=skip_all_reduce,
    )


def make_num_tokens_across_dp(dp_size: int, num_tokens: int) -> torch.Tensor | None:
    """Create a tensor of token counts across DP ranks."""
    if dp_size == 1:
        return None
    return torch.full((dp_size,), num_tokens, dtype=torch.int32, device="cpu")


def get_batch_metadata_across_dp(
    num_tokens: int,
    cudagraph_size: int,
    cudagraph_runtime_mode: int,
    dp_size: int,
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get batch metadata across DP ranks via all_reduce.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - num_tokens_across_dp: Token counts across DP ranks.
            - cudagraph_size_across_dp: Cudagraph sizes across DP ranks.
            - cudagraph_mode_across_dp: Cudagraph modes across DP ranks.
    """
    assert dp_size > 1
    group = get_dp_group().cpu_group
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = cudagraph_size
    tensor[2][dp_rank] = cudagraph_runtime_mode
    dist.all_reduce(tensor, group=group)
    return tensor[0], tensor[1], tensor[2]
