# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision test for the sampling-mask packing kernels (mask_replay).

Validates both Ascend kernels in
``vllm_ascend.ops.triton.v2.sample.pack_sampling_mask`` against a pure NumPy
reference:

* ``_pack_sampling_mask_kernel`` -- release lane (vLLM v0.29.0).
* ``_compact_sampling_mask_kernel`` -- main lane (vLLM >= 84030bbe3), which
  additionally emits the first ``max_num_kept`` finite token ids per request.

Both are wired in by ``patch/worker/patch_v2/patch_triton.py``. The release
kernel's fix (cast ``keep`` to int32 *before* the ``tl.sum`` reduction) is
exercised by the ``counts`` comparison, which fails on the upstream form as
soon as any request keeps more than one token. Both kernels are tested
regardless of which lane is installed, since each is self-contained.

Contract note: ``token_ids`` is allocated with ``torch.empty`` and only
positions ``< min(counts[row], width)`` are written, matching upstream. The
tail of a short row is therefore *unspecified*, and ``tolists()`` never reads
it (it slices ``token_ids[row, :counts[row]]`` and falls back to the bitmask
when ``counts[row] > width``). The assertions below only constrain the
positions that are actually defined.
"""

import numpy as np
import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
    SAMPLING_MASK_BLOCK_SIZE,
    _compact_sampling_mask_kernel,
    _contiguous_logits,
    _pack_sampling_mask_kernel,
)

DEVICE_TYPE = current_platform.device_type

# Mirror the width used for the compact output; the launcher caps it by
# ``min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)``.
MAX_COMPACT_SUPPORT = 2048


def _sampled_tokens(num_reqs: int) -> torch.Tensor:
    """Deterministic per-row activation, always including an inactive row 0."""
    return (torch.arange(num_reqs, dtype=torch.int32) % 4).to(DEVICE_TYPE)


def _launch_release(logits: torch.Tensor, num_sampled_tokens: torch.Tensor):
    """Launch the release-lane kernel, mirroring its ``from_logits``."""
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
    return packed_mask, counts


def _launch_main(logits: torch.Tensor, num_sampled_tokens: torch.Tensor, max_num_kept: int):
    """Launch the main-lane kernel, mirroring its ``from_logits``."""
    num_reqs, vocab_size = logits.shape
    width = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
    packed_width = (vocab_size + 7) // 8
    token_ids = torch.empty((num_reqs, width), dtype=torch.int32, device=logits.device)
    packed_mask = torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=logits.device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
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
        width,
        BLOCK_SIZE=SAMPLING_MASK_BLOCK_SIZE,
    )
    return token_ids, packed_mask, counts


def _reference(logits: torch.Tensor, num_sampled_tokens: torch.Tensor):
    """Pure NumPy reference.

    Returns ``(packed, counts, supports)`` where ``supports[row]`` is the
    ascending array of finite token ids of that row (empty when inactive).
    Packing is little-endian, matching ``np.unpackbits(..., bitorder="little")``.
    """
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8
    logits_np = logits.detach().float().cpu().numpy()
    num_sampled_np = num_sampled_tokens.detach().cpu().numpy()

    keep = np.isfinite(logits_np) & (num_sampled_np[:, None] > 0)
    counts = keep.sum(axis=1).astype(np.int32)

    padded = np.zeros((num_reqs, packed_width * 8), dtype=np.uint8)
    padded[:, :vocab_size] = keep.astype(np.uint8)
    packed = (padded.reshape(num_reqs, packed_width, 8) * (1 << np.arange(8))).sum(axis=2).astype(np.uint8)

    supports = [np.flatnonzero(keep[row]).astype(np.int32) for row in range(num_reqs)]
    return packed, counts, supports


def _assert_compact_matches(token_ids: torch.Tensor, ref_counts: np.ndarray, ref_supports: list):
    """Compare only the defined prefix of each compact row."""
    actual = token_ids.cpu().numpy()
    width = actual.shape[1]
    for row in range(actual.shape[0]):
        defined = int(min(ref_counts[row], width))
        assert actual[row, :defined].tolist() == ref_supports[row][:defined].tolist(), (
            f"row {row}: compact ids differ over the defined prefix (defined={defined})"
        )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
class TestPackSamplingMask:
    @pytest.mark.parametrize(
        "num_reqs,vocab_size",
        [
            (2, 8),  # exact multiple of 8, smallest useful case
            (4, 100),  # non-multiple of 8, tail byte high bits are zero
            (2, 8193),  # just above 2*BLOCK_SIZE, exercises the multi-block tail
            (3, 32000),  # realistic vocab, multiple blocks
        ],
    )
    def test_release_kernel_matches_reference(self, num_reqs, vocab_size):
        torch.manual_seed(0)
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = _sampled_tokens(num_reqs)

        packed, counts = _launch_release(logits, num_sampled_tokens)
        torch.npu.synchronize()

        ref_packed, ref_counts, _ = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))

    @pytest.mark.parametrize(
        "num_reqs,vocab_size,max_num_kept",
        [
            (2, 8, 8),  # compact row exactly covers the support
            (4, 100, 4),  # compact row narrower than the support -> bitmask path
            (2, 8193, 64),  # multi-block accumulation of the compact positions
            (3, 32000, 2048),  # realistic vocab
            (2, 64, 4096),  # max_num_kept above vocab_size, must clamp
        ],
    )
    def test_main_kernel_matches_reference(self, num_reqs, vocab_size, max_num_kept):
        torch.manual_seed(0)
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = _sampled_tokens(num_reqs)

        token_ids, packed, counts = _launch_main(logits, num_sampled_tokens, max_num_kept)
        torch.npu.synchronize()

        expected_width = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
        assert token_ids.shape[1] == expected_width

        ref_packed, ref_counts, ref_supports = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))
        _assert_compact_matches(token_ids, ref_counts, ref_supports)

    def test_main_kernel_overflow_writes_full_row(self):
        """When counts[row] >= width the whole compact row is defined."""
        vocab_size = 4096
        logits = torch.randn(1, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        # Every token is finite, so the support is the whole vocabulary.
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        token_ids, packed, counts = _launch_main(logits, num_sampled_tokens, 16)
        torch.npu.synchronize()

        assert counts.cpu().item() == vocab_size
        assert token_ids.cpu()[0].tolist() == list(range(16))
        # Bitmask is the authoritative fallback once the row overflows.
        ref_packed, _, _ = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))

    def test_non_finite_logits_filtered(self):
        vocab_size = 16
        logits = torch.full((1, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE_TYPE)
        logits[0, 2] = 1.0
        logits[0, 5] = float("inf")
        logits[0, 7] = float("nan")
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch_release(logits, num_sampled_tokens)
        token_ids, main_packed, main_counts = _launch_main(logits, num_sampled_tokens, 8)
        torch.npu.synchronize()

        # Only token 2 is finite (excludes -inf, +inf and NaN) and active.
        assert counts.cpu().item() == 1
        assert packed.cpu()[0, 0].item() == (1 << 2)
        assert main_counts.cpu().item() == 1
        assert main_packed.cpu()[0, 0].item() == (1 << 2)
        assert token_ids.cpu()[0, 0].item() == 2

    def test_num_sampled_zero_clears_row(self):
        vocab_size = 32
        logits = torch.randn(2, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch_release(logits, num_sampled_tokens)
        token_ids, main_packed, main_counts = _launch_main(logits, num_sampled_tokens, vocab_size)
        torch.npu.synchronize()

        # Inactive request: count 0 and fully-zero mask.
        assert counts.cpu()[0].item() == 0
        assert packed.cpu()[0].eq(0).all().item()
        assert main_counts.cpu()[0].item() == 0
        assert main_packed.cpu()[0].eq(0).all().item()
        # Active request with all-finite logits: every token is in the support.
        assert counts.cpu()[1].item() == vocab_size
        assert packed.cpu()[1].eq(0xFF).all().item()
        assert main_counts.cpu()[1].item() == vocab_size
        assert main_packed.cpu()[1].eq(0xFF).all().item()
        assert token_ids.cpu()[1].tolist() == list(range(vocab_size))

    def test_non_contiguous_logits(self):
        vocab_size = 32
        base = torch.randn(1, vocab_size * 2, dtype=torch.float32, device=DEVICE_TYPE)
        logits = base[:, ::2]  # stride-2 view exercises logits_col_stride
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch_release(logits, num_sampled_tokens)
        token_ids, main_packed, main_counts = _launch_main(logits, num_sampled_tokens, 8)
        torch.npu.synchronize()

        ref_packed, ref_counts, ref_supports = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))
        assert torch.equal(main_packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(main_counts.cpu(), torch.from_numpy(ref_counts))
        _assert_compact_matches(token_ids, ref_counts, ref_supports)

    def test_counts_exceed_one(self):
        """Regression for the int1 reduction truncation on the release lane.

        A row keeping N > 1 tokens must report N, not 1. This is the failure
        mode the release-lane int32 cast fixes; the main lane already casts
        upstream.
        """
        vocab_size = 4096
        logits = torch.randn(1, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        _, counts = _launch_release(logits, num_sampled_tokens)
        _, _, main_counts = _launch_main(logits, num_sampled_tokens, 8)
        torch.npu.synchronize()

        assert counts.cpu().item() == vocab_size
        assert main_counts.cpu().item() == vocab_size

    def test_contiguous_logits_helper_is_a_noop_when_contiguous(self):
        logits = torch.randn(3, 128, dtype=torch.float32, device=DEVICE_TYPE)
        assert _contiguous_logits(logits) is logits

    def test_launcher_materializes_strided_logits(self):
        """Regression: strided logits hang the Ascend packing kernel.

        A masked ``tl.load`` with a column stride != 1 lowers to a gather. At
        the real Qwen3-30B-A3B shape (``vocab_size=151936``) with
        ``BLOCK_SIZE=1024``, one row costs ~2.3s and two rows never finish,
        ending in "Vector core execution timed out". The sampler does reach
        this state because ``processed_logits`` comes back non-contiguous from
        the Ascend ``apply_top_k_top_p`` path, so the launcher must copy first.
        Without that copy this test hangs instead of failing.

        Driven through the public launchers rather than the raw kernels, since
        the fix lives in the launcher.
        """
        from vllm.v1.worker.gpu.sample import output as vllm_output

        from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
            compact_sampling_mask_from_logits,
            sampling_mask_from_logits,
        )

        vocab_size = 151936
        base = torch.randn(2, vocab_size * 2, dtype=torch.float32, device=DEVICE_TYPE)
        logits = base[:, ::2]
        assert logits.stride(1) == 2, "test precondition: input must be strided"

        num_sampled_tokens = torch.ones(2, dtype=torch.int32, device=DEVICE_TYPE)
        cls = vllm_output.SamplingMaskTensors
        if hasattr(vllm_output, "_compact_sampling_mask_kernel"):
            tensors = compact_sampling_mask_from_logits(cls, logits, num_sampled_tokens, 50)
        else:
            tensors = sampling_mask_from_logits(cls, logits, num_sampled_tokens)
        torch.npu.synchronize()

        # Every logit is finite, so each row keeps the whole vocabulary.
        assert tensors.counts.cpu().tolist() == [vocab_size, vocab_size]

        ref_packed, ref_counts, _ = _reference(logits, num_sampled_tokens)
        assert torch.equal(tensors.packed_mask.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(tensors.counts.cpu(), torch.from_numpy(ref_counts))
