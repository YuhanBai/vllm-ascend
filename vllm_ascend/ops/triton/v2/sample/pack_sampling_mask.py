# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacement for the sampling-mask ("mask replay") packing kernel.

This is the NPU port of ``_compact_sampling_mask_kernel`` from
``vllm.v1.worker.gpu.sample.output``. For every request that produced a sampled
token it captures the *finite-logit support* -- the token ids whose logits are
neither ``+inf``/``-inf`` nor ``NaN``, which is exactly the set that survived
top-k / top-p / min-p and can therefore be replayed.

The support is emitted as the bit-packed ``uint8`` mask plus the exact
per-request count. ``SamplingMaskTensors.tolists()`` turns that pair into the
support set, so the reconstructed representation is exact for every row.

Two deliberate differences from the upstream kernel
----------------------------------------------------
1. ``BLOCK_SIZE`` is reduced. vLLM hard-codes 8192, which the Ascend backend
   cannot lower at all (``hivm-plan-memory`` / ``ConvertLinalgRToBinary`` fail
   on the TTIR this kernel produces); 4096/2048 still fail on the
   non-contiguous (strided/gather) load path, whose maximum block is much
   smaller than a contiguous load's. 1024 lowers both paths.

2. The compact ``token_ids`` output is not produced, and the ``tl.cumsum``
   prefix scan used to build it is gone.

   ``tl.cumsum`` makes the kernel impossible to capture in an ACLGraph: the
   scan forces triton-ascend into a launch path that issues a device
   ``rtMemcpy``, and the runtime rejects it with

       rtMemcpy execution failed, reason=operation not permitted when a
       stream is capturing and the specified capture mode is not relaxed

   which surfaces as an AIVEC "vector core execution timed out" and kills the
   engine as soon as the first graphed decode step runs. The scan is the only
   construct responsible -- a trivial Triton kernel and this same kernel
   without the scan both capture and replay cleanly, while adding the scan back
   fails immediately (measured on CANN 9.1.0 / torch_npu 2.10.0.post4).

   Dropping it costs nothing semantically: the compact buffer is only a fast
   path for rows whose support is at most ``max_num_kept`` wide, and it used to
   be sized ``min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)``. Returning a
   zero-width compact buffer instead makes ``tolists()`` take its bitmask path
   for every row, which is the exact representation. The only price is that the
   support is materialized from the bitmask on the host rather than sliced out
   of a small int32 buffer.

The upstream ``keep.to(tl.int32)`` widening is kept: triton-ascend leaves an
``int1`` reduction in ``int1``, unlike native CUDA Triton which upcasts to
``int32``, so summing ``keep`` directly would truncate ``counts`` to 0/1 and
silently replay the wrong candidate set.
"""

import torch
from vllm.triton_utils import tl, triton

# See note 1 in the module docstring.
SAMPLING_MASK_BLOCK_SIZE = 1024

# Reused output buffers, keyed by (device, num_reqs, vocab_size).
#
# Upstream allocates fresh outputs on every step. Under ACLGraph those
# allocations (and the launch reading their addresses) land inside the captured
# region, so keep the addresses stable across capture and replay. This is the
# same class of workaround vLLM-Ascend already applies to ``npu_top_k_top_p``
# (see ``worker/v2/sample/apply_top_k_top_p.py``).
_MASK_BUFFER_CACHE: dict = {}


def _mask_buffers(num_reqs: int, vocab_size: int, device):
    key = (str(device), num_reqs, vocab_size)
    buf = _MASK_BUFFER_CACHE.get(key)
    if buf is None:
        packed_width = (vocab_size + 7) // 8
        buf = (
            torch.empty((num_reqs, 0), dtype=torch.int32, device=device),
            torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=device),
            torch.empty(num_reqs, dtype=torch.int32, device=device),
        )
        _MASK_BUFFER_CACHE[key] = buf
    return buf


@triton.jit
def _compact_sampling_mask_kernel_ascend(
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
    """Per row: the bit-packed finite-logit support and its exact size."""
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
        # triton-ascend does not upcast an int1 reduction to int32; widen first
        # so the per-request support size is exact.
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


def compact_sampling_mask_from_logits(
    cls,
    logits: torch.Tensor,
    num_sampled_tokens: torch.Tensor,
    max_num_kept: int,
):
    """NPU replacement for ``SamplingMaskTensors.from_logits``.

    ``max_num_kept`` is accepted for interface compatibility with upstream but
    is not used: the compact token-id buffer is returned with zero width so that
    ``tolists()`` always reconstructs the support from the exact bitmask (see
    note 2 in the module docstring). Wired in from
    ``vllm_ascend/patch/worker/patch_v2/patch_triton.py``.
    """
    num_reqs, vocab_size = logits.shape
    token_ids, packed_mask, counts = _mask_buffers(num_reqs, vocab_size, logits.device)
    _compact_sampling_mask_kernel_ascend[(num_reqs,)](
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
    return cls(token_ids, packed_mask, counts, vocab_size)
