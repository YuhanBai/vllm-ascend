# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacements for the sampling-mask packing kernels.

Both supported lanes are implemented here, because upstream renamed and
reshaped the kernel between the v0.29.0 release and the verified main commit:

* release lane (v0.29.0): ``_pack_sampling_mask_kernel`` emits the bit-packed
  mask plus a per-request ``counts`` tensor.
* main lane (>= 84030bbe3): ``_compact_sampling_mask_kernel`` additionally emits
  the first ``max_num_kept`` finite token ids per request, so the scheduler can
  skip unpacking the bitmask for the common case.

``vllm_ascend/patch/worker/patch_v2/patch_triton.py`` dispatches on whichever
symbol the installed vLLM actually exposes.

Why an Ascend copy is needed at all
-----------------------------------
1. ``BLOCK_SIZE=8192`` (hard-coded by both upstream launchers) cannot be
   launched by the NPU backend. The tightest constraint is the non-contiguous
   (strided/gather) load path, whose max block is far smaller than a contiguous
   load's; 2048/4096 still fail there. 1024 keeps both the contiguous and the
   strided case under the limit.
2. Release lane only: upstream sums an ``int1`` tensor (``tl.sum(keep)``) and
   casts the *result* to int32. Native CUDA Triton upcasts the reduction
   automatically, triton-ascend does not, so ``counts`` is truncated to 0/1 and
   the sampling mask is wrong. Casting ``keep`` to int32 *before* the reduction
   fixes it; the main lane already does this upstream.
"""

import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

# NPU backend cannot launch the kernel with the upstream BLOCK_SIZE=8192.
# The tightest constraint is the strided-load path: when logits is
# non-contiguous (logits_col_stride != 1) the masked ``tl.load`` lowers to a
# gather, whose max block on Ascend is far smaller than a contiguous load's.
# 2048/4096 fail on that case even though contiguous inputs work; 1024 keeps
# both contiguous and strided cases under the limit. Temporary reduction
# pending an NPU-side fix.
SAMPLING_MASK_BLOCK_SIZE = 1024

# Fallback for the upstream compact-width cap when the attribute is absent.
# The main lane defines ``MAX_COMPACT_SUPPORT = 2048`` in the output module; we
# read it lazily so a release-lane install (which has no such constant) can
# still import this module.
_DEFAULT_MAX_COMPACT_SUPPORT = 2048


def _contiguous_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return ``logits`` with a unit column stride.

    A masked ``tl.load`` whose column stride is not 1 lowers to a gather on
    Ascend, and that path is not merely slower but pathologically slow:
    measured at ``vocab_size=151936`` with ``BLOCK_SIZE=1024``, one row costs
    ~2.3s (19x the contiguous case) and two rows never finish at all, ending in
    ``Vector core execution timed out`` after the device watchdog fires. The
    sampler reaches this with ordinary top-k/top-p settings, because
    ``processed_logits`` comes back non-contiguous from the Ascend
    ``apply_top_k_top_p`` path.

    Upstream never hits this on CUDA, where strided gathers are cheap.
    Materializing costs one ``O(num_reqs * vocab)`` copy, which is negligible
    next to the kernel itself (measured 0.86s for 100 rows).
    """
    if logits.stride(1) == 1:
        return logits
    logger.warning_once(
        "sampling-mask replay: logits have column stride %d; materializing a "
        "contiguous copy before the Ascend packing kernel.",
        logits.stride(1),
    )
    return logits.contiguous()


# ---------------------------------------------------------------------------
# Release lane: vLLM v0.29.0 -- ``_pack_sampling_mask_kernel``
# ---------------------------------------------------------------------------
@triton.jit
def _pack_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        # Cast to int32 before the reduction: triton-ascend keeps the int1 sum
        # in int1, which truncates counts larger than 1 down to 0/1.
        keep_i32 = keep.to(tl.int32)
        count += tl.sum(keep_i32)
        bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            tl.sum(bits, axis=1).to(tl.uint8),
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


def sampling_mask_from_logits(cls, logits, num_sampled_tokens):
    """NPU replacement for the release-lane ``SamplingMaskTensors.from_logits``.

    Identical to the upstream classmethod except it launches with the reduced
    ``SAMPLING_MASK_BLOCK_SIZE`` instead of the upstream hard-coded 8192, and
    materializes non-contiguous logits (see ``_contiguous_logits``).
    """
    logits = _contiguous_logits(logits)
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8

    packed_mask = torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=logits.device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _pack_sampling_mask_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled_tokens,
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        BLOCK_SIZE=SAMPLING_MASK_BLOCK_SIZE,
    )

    return cls(packed_mask, counts, vocab_size)


# ---------------------------------------------------------------------------
# Main lane: verified commit 84030bbe3 -- ``_compact_sampling_mask_kernel``
# ---------------------------------------------------------------------------
@triton.jit
def _compact_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    token_ids_ptr,
    token_ids_row_stride,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    max_num_kept,
    BLOCK_SIZE: tl.constexpr,
):
    """Per row: first ``max_num_kept`` finite-logit ids, the count, the bitmask.

    Upstream already casts ``keep`` to int32 before the reduction, so the only
    difference here is the launch block size.
    """
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + req_idx * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        bits = tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            tl.sum(bits, axis=1).to(tl.uint8),
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


def compact_sampling_mask_from_logits(cls, logits, num_sampled_tokens, max_num_kept):
    """NPU replacement for the main-lane ``SamplingMaskTensors.from_logits``.

    Identical to the upstream classmethod except it launches with the reduced
    ``SAMPLING_MASK_BLOCK_SIZE`` instead of the upstream hard-coded 8192, and
    materializes non-contiguous logits (see ``_contiguous_logits``). The
    compact-width cap is read from the installed vLLM so the two cannot drift.
    """
    # Imported lazily: this constant only exists on the main lane.
    from vllm.v1.worker.gpu.sample import output as vllm_output

    cap = getattr(vllm_output, "MAX_COMPACT_SUPPORT", _DEFAULT_MAX_COMPACT_SUPPORT)

    logits = _contiguous_logits(logits)
    num_reqs, vocab_size = logits.shape
    max_num_kept = min(max_num_kept, vocab_size, cap)
    device = logits.device

    token_ids = torch.empty((num_reqs, max_num_kept), dtype=torch.int32, device=device)
    packed_mask = torch.empty((num_reqs, (vocab_size + 7) // 8), dtype=torch.uint8, device=device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=device)
    _compact_sampling_mask_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled_tokens,
        token_ids,
        token_ids.stride(0),
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        max_num_kept,
        BLOCK_SIZE=SAMPLING_MASK_BLOCK_SIZE,
    )

    return cls(token_ids, packed_mask, counts, vocab_size)
