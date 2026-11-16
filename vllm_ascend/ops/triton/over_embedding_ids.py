# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton over-embedding ID computation for LongCat 2.5."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _over_embedding_ids(
    current,
    history,
    positions,
    request_rows,
    weights,
    moduli,
    offsets,
    output,
    rows,
    eos,
    HISTORY_ROW: tl.constexpr,
    HISTORY_COL: tl.constexpr,
    CURRENT_STRIDE: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    TABLES: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKENS_PER_TILE: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    table = lane // TOKENS_PER_TILE
    table_valid = table < TABLES
    for tile in range(tl.program_id(0), tl.cdiv(rows, TOKENS_PER_TILE), tl.num_programs(0)):
        row = tile * TOKENS_PER_TILE + lane % TOKENS_PER_TILE
        active = (row < rows) & table_valid
        token = tl.load(current + row * CURRENT_STRIDE, active, other=0).to(tl.int64)
        factor = tl.load(weights + table * N, table_valid, other=0).to(tl.int64)
        hashed = token * factor
        valid = active
        if positions is not None:
            position = tl.load(positions + row * POSITION_STRIDE, active, other=0)
            req = tl.load(request_rows + row * REQUEST_STRIDE, active, other=0).to(tl.int64)
        for delta in tl.static_range(1, N):
            if positions is not None:
                lookback = position - delta
                token = tl.load(history + req * HISTORY_ROW + lookback * HISTORY_COL, valid & (lookback >= 0), other=-1)
            else:
                token = tl.load(history + row * HISTORY_ROW + (delta - 1) * HISTORY_COL, valid, other=-1)
            valid = valid & (token >= 0) & (token != eos)
            factor = tl.load(weights + table * N + delta, table_valid, other=0).to(tl.int64)
            hashed += tl.where(valid & (delta < table // SPLITS + 2), token.to(tl.int64) * factor, 0)
        modulus = tl.load(moduli + table, table_valid, other=1).to(tl.int64)
        offset = tl.load(offsets + table, table_valid, other=0)
        hashed = hashed % modulus
        output_index = table * rows + row if positions is not None else row * TABLES + table
        tl.store(output + output_index, tl.where(hashed < 0, hashed + modulus, hashed) + offset, active)


def over_embedding_ids_triton(input_ids, history, eos_id, weights, moduli, offsets):
    """Return token-major IDs for pre-gathered history."""
    return _launch_over_embedding_ids(input_ids, history, eos_id, weights, moduli, offsets)


def over_embedding_ids_from_token_table(input_ids, token_table, positions, request_rows, eos_id, weights, moduli, offsets):
    """Read scheduler history directly and return contiguous table-major IDs."""
    return _launch_over_embedding_ids(
        input_ids, token_table, eos_id, weights, moduli, offsets, positions=positions, request_rows=request_rows
    )


def _launch_over_embedding_ids(input_ids, history, eos_id, weights, moduli, offsets, *, positions=None, request_rows=None):
    rows = input_ids.numel()
    tables = moduli.numel()
    output_shape = (rows, tables) if positions is None else (tables, rows)
    output = torch.empty(output_shape, dtype=torch.int32, device=input_ids.device)
    if rows and tables:
        padded_tables = triton.next_power_of_2(tables)
        block = max(128, padded_tables)
        tokens_per_tile = block // padded_tables
        _over_embedding_ids[(min(256, triton.cdiv(rows, tokens_per_tile)),)](
            input_ids,
            history,
            positions,
            request_rows,
            weights,
            moduli,
            offsets,
            output,
            rows,
            eos_id,
            history.stride(0),
            history.stride(1),
            input_ids.stride(0),
            positions.stride(0) if positions is not None else 0,
            request_rows.stride(0) if request_rows is not None else 0,
            weights.shape[-1],
            weights.shape[1],
            tables,
            block,
            tokens_per_tile,
        )
    return output
