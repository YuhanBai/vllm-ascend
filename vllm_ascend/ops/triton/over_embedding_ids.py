# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton over-embedding ID computation for LongCat 2.5."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _over_embedding_ids(
    current,
    history,
    weights,
    moduli,
    offsets,
    output,
    rows,
    eos,
    HISTORY_ROW: tl.constexpr,
    HISTORY_COL: tl.constexpr,
    CURRENT_STRIDE: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    TABLES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    for tile in range(tl.program_id(0), tl.cdiv(rows * TABLES, BLOCK), tl.num_programs(0)):
        index = tile * BLOCK + tl.arange(0, BLOCK)
        row = index // TABLES
        table = index % TABLES
        token = tl.load(current + row * CURRENT_STRIDE, row < rows, other=0).to(tl.int64)
        factor = tl.load(weights + table * N).to(tl.int64)
        hashed = token * factor
        valid = row < rows
        for delta in tl.static_range(1, N):
            token = tl.load(history + row * HISTORY_ROW + (delta - 1) * HISTORY_COL, row < rows, other=-1)
            valid = valid & (token >= 0) & (token != eos)
            factor = tl.load(weights + table * N + delta).to(tl.int64)
            hashed += tl.where(valid & (delta < table // SPLITS + 2), token.to(tl.int64) * factor, 0)
        modulus = tl.load(moduli + table).to(tl.int64)
        offset = tl.load(offsets + table)
        hashed = hashed % modulus
        tl.store(output + index, tl.where(hashed < 0, hashed + modulus, hashed) + offset, row < rows)


def over_embedding_ids_triton(input_ids, history, eos_id, weights, moduli, offsets):
    rows = input_ids.numel()
    tables = moduli.numel()
    output = torch.empty((rows, tables), dtype=torch.int32, device=input_ids.device)
    if rows:
        block = 128
        _over_embedding_ids[(min(256, triton.cdiv(rows * tables, block)),)](
            input_ids,
            history,
            weights,
            moduli,
            offsets,
            output,
            rows,
            eos_id,
            history.stride(0),
            history.stride(1),
            input_ids.stride(0),
            weights.shape[-1],
            weights.shape[1],
            tables,
            block,
        )
    return output
