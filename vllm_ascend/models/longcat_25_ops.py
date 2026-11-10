# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from contextlib import suppress
from functools import cache
from importlib.util import find_spec
from pathlib import Path

import torch


def route_experts(logits, bias, topk, num_experts, scale):
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    expert_ids = (probs + bias).topk(topk, dim=-1, sorted=False).indices
    weights = probs.gather(-1, expert_ids) * scale
    identity = expert_ids >= num_experts
    return weights.masked_fill(identity, 0), expert_ids.masked_fill(identity, 0), weights.masked_fill(~identity, 0)


def over_embedding_ids(input_ids, history, eos_id, weights, moduli, offsets):
    """Hash current tokens and newest-first history without crossing an EOS or request boundary."""
    n = weights.shape[-1]
    valid = (history >= 0) & (history != eos_id)
    valid = valid.to(torch.int32).cumprod(dim=-1).bool()
    tokens = torch.cat((input_ids[:, None], history.masked_fill(~valid, 0)), dim=-1)
    # Each table includes only its own order of n-gram. Int64 avoids overflow
    # when multiplying a token ID by the checkpoint's large hash moduli.
    orders = torch.arange(2, weights.shape[0] + 2, device=input_ids.device)
    mask = torch.arange(n, device=input_ids.device) < orders[:, None]
    multipliers = (weights * mask[:, None, :]).reshape(-1, n)
    hashed = (tokens.to(torch.int64)[:, None, :] * multipliers).sum(dim=-1)
    return (hashed.remainder(moduli.flatten()) + offsets[:-1]).to(torch.int32)


@cache
def get_longcat_lightning_indexer():
    """Find the BF16 Lightning Indexer with LongCat's mandatory-token inputs."""
    for namespace in (torch.ops.npu, torch.ops.custom):
        op = getattr(namespace, "npu_lightning_indexer", None)
        if op is not None and _supports_init_and_local(op):
            return op

    # Load the dispatcher registrations directly. Importing the Python package
    # also installs its other operators into torch_npu, affecting unrelated models.
    spec = find_spec("flash_ops")
    if spec is not None and spec.origin is not None:
        library = next(Path(spec.origin).parent.glob("custom_ops_lib*.so"), None)
        if library is not None:
            # Some flash_ops builds leave CANN's transformer symbols out of
            # DT_NEEDED. Match the extension's own loader for those builds.
            with suppress(OSError):
                ctypes.CDLL("libopapi_transformer.so", mode=ctypes.RTLD_GLOBAL)
            try:
                torch.ops.load_library(str(library))
            except OSError as exc:
                raise RuntimeError(
                    "Cannot load the LongCat-2.5 flash_ops extension. Check its PyTorch and CANN compatibility."
                ) from exc
            op = getattr(torch.ops.custom, "npu_lightning_indexer", None)
            if op is not None and _supports_init_and_local(op):
                return op
    raise RuntimeError(
        "LongCat-2.5 requires npu_lightning_indexer with init_tensor and local_tensor. "
        "Use a compatible torch_npu build or install and activate the SGLang-FluentLLM flash_ops wheel and CANN OPP."
    )


def _supports_init_and_local(op):
    arguments = {arg.name for arg in op.default._schema.arguments}
    return {"init_tensor", "local_tensor"} <= arguments


def select_longcat_topk(query, weights, indexer, metadata, query_scale, *, op, init_tokens, local_tokens):
    if query_scale is not None:
        raise ValueError("LongCat-2.5 requires an unquantized BF16 indexer.")
    seq_lens = metadata.actual_seq_lengths_key
    init_tensor = torch.full_like(seq_lens, init_tokens, dtype=torch.int32)
    local_tensor = query.new_full((query.shape[0],), local_tokens, dtype=torch.int32)
    indices, _ = op(
        query=query,
        key=indexer.k_cache.kv_cache[0],
        weights=weights.to(torch.bfloat16),
        actual_seq_lengths_query=metadata.actual_seq_lengths_query,
        actual_seq_lengths_key=seq_lens,
        block_table=metadata.block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=indexer.topk_tokens,
        sparse_mode=3,
        init_tensor=init_tensor,
        local_tensor=local_tensor,
    )
    return indices
