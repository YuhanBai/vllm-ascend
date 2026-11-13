"""Ascend 950 BF16 Lightning Indexer reference-compatible composition."""

import torch
from vllm.triton_utils import HAS_TRITON


def lightning_indexer_with_mandatory_tokens(
    query,
    key,
    weights,
    *,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    layout_query,
    layout_key,
    sparse_count,
    sparse_mode,
    init_tensor,
    local_tensor,
    use_triton=False,
):
    if layout_query != "TND" or layout_key != "PA_BSND" or sparse_mode != 3:
        raise ValueError("LongCat Ascend950 expects TND/PA_BSND causal Lightning Indexer")
    if key.shape[2] != 1:
        raise ValueError("LongCat Ascend950 expects a single key head")
    candidates, _ = torch.ops.npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )
    # Padded graph rows after the last query have a nonpositive valid length.
    rows = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    sequence = (rows[:, None] >= actual_seq_lengths_query[None, :]).sum(-1)
    in_batch = sequence < actual_seq_lengths_query.numel()
    sequence = sequence.clamp(max=actual_seq_lengths_query.numel() - 1)
    valid_len = actual_seq_lengths_key[sequence] - actual_seq_lengths_query[sequence] + rows + 1
    valid_len = torch.where(in_batch, valid_len, 0)
    prefix = init_tensor[sequence].to(torch.int32)
    local = local_tensor.to(torch.int32)
    force = prefix + local < valid_len
    prefix = torch.where(force, prefix, 0)[:, None]
    local = torch.where(force, local, 0)[:, None]
    valid_len = valid_len[:, None]

    order = torch.arange(sparse_count, dtype=torch.int32, device=query.device)[None, :]
    forced_indices = torch.where(order < prefix, order, valid_len - local + order - prefix)
    forced_valid = order < prefix + local
    scores_forced = torch.where(forced_valid, torch.finfo(torch.float32).max / 2, -torch.inf)
    chosen = candidates[:, 0, :]
    middle = (chosen >= prefix) & (chosen < valid_len - local)
    # Ascend 950 returns paged candidates in index order. Rescore only these
    # candidates with the LongCat FP32 ReLU-dot and head-weight reduction.
    if use_triton and HAS_TRITON and query.device.type == "npu" and query.shape[-1] == 128:
        from vllm_ascend.ops.triton.lightning_indexer_rescore import rescore_candidates_triton

        scores_native = rescore_candidates_triton(
            query, key, weights, block_table, sequence, chosen, prefix, valid_len - local
        )
    else:
        block_size = key.shape[1]
        scores_chunks = []
        for start in range(0, query.shape[0], 32):
            stop = start + 32
            safe_indices = chosen[start:stop].clamp_min(0).long()
            physical = block_table[sequence[start:stop, None], safe_indices // block_size].clamp_min(0)
            candidate_keys = key[physical.long(), safe_indices % block_size, 0].float()
            logits = torch.bmm(query[start:stop].float(), candidate_keys.transpose(1, 2))
            scores_chunks.append((logits.relu() * weights[start:stop, :, None].float()).sum(dim=1))
        scores_native = torch.where(middle, torch.cat(scores_chunks), -torch.inf)

    # The mandatory set removes at most its own size from the native top-k,
    # so the remaining native prefix contains enough highest-scoring entries.
    scores = torch.cat((scores_forced, scores_native), dim=-1).float()
    best_scores, best = scores.topk(sparse_count, dim=-1, sorted=True)
    indices = torch.cat((forced_indices, chosen), dim=-1).gather(-1, best)
    return torch.where(best_scores.isfinite(), indices, -1).to(torch.int32)[:, None, :], query.new_empty((0,))
