# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gather paged keys for LongCat 2.5 Lightning Indexer rescoring on Ascend 950."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows"])
def _gather_candidate_keys(
    cache,
    block_table,
    sequence,
    chosen,
    output,
    rows,
    K_PAGE: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_DIM: tl.constexpr,
    BT_ROW: tl.constexpr,
    BT_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_COLUMNS: tl.constexpr,
    PAGES: tl.constexpr,
    DIM: tl.constexpr,
    COUNT: tl.constexpr,
    TILE: tl.constexpr,
):
    chunks = tl.cdiv(COUNT, TILE)
    for tile in range(tl.program_id(0), rows * chunks, tl.num_programs(0)):
        row = tile // chunks
        column = (tile % chunks) * TILE + tl.arange(0, TILE)
        dims = tl.arange(0, DIM)
        active = column < COUNT
        index = tl.maximum(tl.load(chosen + row * C_ROW + column * C_COL, active, other=0), 0)
        req = tl.load(sequence + row)
        logical_page = index // PAGE_SIZE
        valid = active & (logical_page < PAGE_COLUMNS)
        page = tl.load(block_table + req * BT_ROW + logical_page * BT_COL, valid, other=0)
        page = tl.maximum(page, 0).to(tl.int64)
        valid = valid & (page < PAGES)
        key = tl.load(
            cache + page[:, None] * K_PAGE + (index % PAGE_SIZE)[:, None] * K_TOKEN + dims[None, :] * K_DIM,
            valid[:, None],
            other=0,
        ).to(tl.float32)
        tl.store(output + (row * COUNT + column[:, None]) * DIM + dims[None, :], key, active[:, None])


def gather_candidate_keys_triton(key, block_table, sequence, chosen):
    rows, count = chosen.shape
    dim = key.shape[-1]
    output = torch.empty((rows, count, dim), dtype=torch.float32, device=key.device)
    if rows and count:
        tile = 16
        _gather_candidate_keys[(min(256, rows * triton.cdiv(count, tile)),)](
            key,
            block_table,
            sequence,
            chosen,
            output,
            rows,
            key.stride(0),
            key.stride(1),
            key.stride(3),
            *block_table.stride(),
            *chosen.stride(),
            key.shape[1],
            block_table.shape[1],
            key.shape[0],
            dim,
            count,
            tile,
        )
    return output
