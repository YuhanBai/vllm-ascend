# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton rescoring of paged Lightning Indexer candidates for LongCat 2.5 on Ascend 950."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _rescore_candidates(
    query,
    cache,
    weights,
    block_table,
    sequence,
    chosen,
    prefix,
    end,
    output,
    rows,
    Q_ROW: tl.constexpr,
    Q_HEAD: tl.constexpr,
    Q_DIM: tl.constexpr,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_DIM: tl.constexpr,
    W_ROW: tl.constexpr,
    W_HEAD: tl.constexpr,
    BT_ROW: tl.constexpr,
    BT_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE_COLUMNS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
):
    chunks = tl.cdiv(COUNT, TILE)
    for tile in range(tl.program_id(0), rows * chunks, tl.num_programs(0)):
        row = tile // chunks
        column = (tile % chunks) * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, DIM)
        index = tl.load(chosen + row * C_ROW + column * C_COL, column < COUNT, other=-1)
        req = tl.load(sequence + row)
        lo = tl.load(prefix + row)
        hi = tl.load(end + row)
        logical_page = index // PAGE_SIZE
        valid = (column < COUNT) & (index >= lo) & (index < hi) & (logical_page < PAGE_COLUMNS)
        page = tl.load(block_table + req * BT_ROW + logical_page * BT_COL, valid, other=0)
        page = tl.minimum(tl.maximum(page, 0), PAGES - 1).to(tl.int64)
        key = tl.load(
            cache + page[:, None] * K_PAGE + (index % PAGE_SIZE)[:, None] * K_TOKEN + dims[None, :] * K_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        score = tl.full((TILE,), 0, tl.float32)
        for head in range(HEADS):
            q = tl.load(query + row * Q_ROW + head * Q_HEAD + dims * Q_DIM).to(tl.float32)
            weight = tl.load(weights + row * W_ROW + head * W_HEAD).to(tl.float32)
            score += tl.maximum(tl.sum(key * q[None, :], axis=1), 0.0) * weight
        tl.store(output + row * COUNT + column, tl.where(valid, score, -float("inf")), column < COUNT)


def rescore_candidates_triton(query, key, weights, block_table, sequence, chosen, prefix, end):
    rows, count = chosen.shape
    output = torch.empty((rows, count), dtype=torch.float32, device=query.device)
    if rows and count:
        tile = 16
        _rescore_candidates[(min(256, rows * triton.cdiv(count, tile)),)](
            query,
            key,
            weights,
            block_table,
            sequence,
            chosen,
            prefix,
            end,
            output,
            rows,
            *query.stride(),
            key.stride(0),
            key.stride(1),
            key.stride(3),
            *weights.stride(),
            *block_table.stride(),
            *chosen.stride(),
            key.shape[1],
            key.shape[0],
            block_table.shape[1],
            query.shape[1],
            query.shape[2],
            count,
            tile,
        )
    return output
