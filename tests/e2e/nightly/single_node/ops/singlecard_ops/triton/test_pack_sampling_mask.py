# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for the sampling-mask packing kernel used by mask replay.

Validates ``_compact_sampling_mask_kernel_ascend`` from
``vllm_ascend.ops.triton.v2.sample.pack_sampling_mask`` against a pure NumPy
reference. This is the NPU replacement for the upstream
``_compact_sampling_mask_kernel``, wired in from
``patch/worker/patch_v2/patch_triton.py``.

Three Ascend-specific hazards are covered explicitly, because each of them
makes mask replay silently return the wrong candidate set or kills the engine
rather than failing loudly:

* triton-ascend does not upcast an ``int1`` reduction, so a kernel that sums
  ``keep`` before widening truncates ``counts`` to 0/1. The ``counts``
  comparison catches that as soon as any request keeps more than one token.
* the upstream launcher hard-codes ``BLOCK_SIZE=8192``, which the Ascend
  backend cannot lower (``hivm-plan-memory`` fails), so the kernel is launched
  through ``SAMPLING_MASK_BLOCK_SIZE``; importing it here keeps that honest.
* any construct that makes the kernel un-capturable in an ACLGraph (a prefix
  scan with a data-dependent scatter store was the culprit) is caught by
  ``test_capture_replay`` -- without it, mask replay works in eager mode and
  hard-crashes the engine the moment a graphed decode step first replays.
"""

import numpy as np
import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
    SAMPLING_MASK_BLOCK_SIZE,
    _compact_sampling_mask_kernel_ascend,
)

DEVICE_TYPE = current_platform.device_type


def _launch(logits: torch.Tensor, num_sampled_tokens: torch.Tensor, packed_width: int):
    """Launch the kernel, mirroring ``SamplingMaskTensors.from_logits``."""
    num_reqs, vocab_size = logits.shape
    packed_mask = torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=logits.device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
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
    return packed_mask, counts


def _reference(logits: torch.Tensor, num_sampled_tokens: torch.Tensor, packed_width: int):
    """Pure NumPy reference for ``packed_mask``/``counts`` (little-endian)."""
    num_reqs, vocab_size = logits.shape
    logits_np = logits.detach().float().cpu().numpy()
    num_sampled_np = num_sampled_tokens.detach().cpu().numpy()

    keep = np.isfinite(logits_np) & (num_sampled_np[:, None] > 0)
    counts = keep.sum(axis=1).astype(np.int32)

    padded = np.zeros((num_reqs, packed_width * 8), dtype=np.uint8)
    padded[:, :vocab_size] = keep.astype(np.uint8)
    packed = (padded.reshape(num_reqs, packed_width, 8) * (1 << np.arange(8))).sum(axis=2).astype(np.uint8)
    return packed, counts


def _assert_matches(logits, num_sampled, packed_width):
    packed_mask, counts = _launch(logits, num_sampled, packed_width)
    torch.npu.synchronize()
    ref_packed, ref_counts = _reference(logits, num_sampled, packed_width)
    assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts)), "counts mismatch"
    assert torch.equal(packed_mask.cpu(), torch.from_numpy(ref_packed)), "packed_mask mismatch"


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
class TestCompactSamplingMask:
    @pytest.mark.parametrize(
        "num_reqs,vocab_size",
        [
            (2, 8),  # exact multiple of 8, smallest useful case
            (4, 100),  # non-multiple of 8, tail byte high bits are zero
            (2, 8193),  # just above 2*BLOCK_SIZE, exercises the multi-block tail
            (3, 32000),  # realistic vocab, multiple blocks
            (2, 151936),  # Qwen3 vocab, the shape mask replay actually sees
        ],
    )
    def test_matches_reference(self, num_reqs, vocab_size):
        torch.manual_seed(0)
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.bfloat16, device=DEVICE_TYPE)
        num_sampled_tokens = torch.randint(0, 4, (num_reqs,), dtype=torch.int32, device=DEVICE_TYPE)
        # Always include at least one inactive request (num_sampled == 0).
        num_sampled_tokens[0] = 0
        _assert_matches(logits, num_sampled_tokens, (vocab_size + 7) // 8)

    @pytest.mark.parametrize("max_num_kept", [1, 2, 20])
    def test_top_k_like_support(self, max_num_kept):
        """A top-k style row keeps exactly ``max_num_kept`` finite entries.

        ``counts`` must equal ``max_num_kept``, not 1: this is the regression
        guard for the triton-ascend int1 reduction.
        """
        vocab_size = 32000
        logits = torch.full((2, vocab_size), -float("inf"), dtype=torch.bfloat16, device=DEVICE_TYPE)
        logits[:, :max_num_kept] = torch.randn(2, max_num_kept, dtype=torch.bfloat16, device=DEVICE_TYPE)
        num_sampled_tokens = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_TYPE)

        packed_mask, counts = _launch(logits, num_sampled_tokens, (vocab_size + 7) // 8)
        torch.npu.synchronize()

        cpu_counts = counts.cpu()
        assert cpu_counts[0].item() == 0
        assert cpu_counts[1].item() == max_num_kept, "counts truncated (int1 reduction bug)"

        # Ids 0..max_num_kept-1 are bit-packed little-endian, 8 ids per byte.
        expected = bytearray((vocab_size + 7) // 8)
        for tid in range(max_num_kept):
            expected[tid // 8] |= 1 << (tid % 8)
        assert packed_mask.cpu()[1].numpy().tobytes() == bytes(expected)
        assert packed_mask.cpu()[0].eq(0).all().item()

    def test_non_finite_logits_filtered(self):
        vocab_size = 16
        logits = torch.full((1, vocab_size), -float("inf"), dtype=torch.bfloat16, device=DEVICE_TYPE)
        logits[0, 2] = 1.0
        logits[0, 5] = float("inf")
        logits[0, 7] = float("nan")
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        packed_mask, counts = _launch(logits, num_sampled_tokens, (vocab_size + 7) // 8)
        torch.npu.synchronize()

        # Only token 2 is finite (excludes -inf, +inf and NaN) and active.
        assert counts.cpu().item() == 1
        assert packed_mask.cpu()[0, 0].item() == (1 << 2)

    def test_num_sampled_zero_clears_row(self):
        vocab_size = 32
        logits = torch.randn(2, vocab_size, dtype=torch.bfloat16, device=DEVICE_TYPE)
        num_sampled_tokens = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_TYPE)

        packed_mask, counts = _launch(logits, num_sampled_tokens, (vocab_size + 7) // 8)
        torch.npu.synchronize()

        # Inactive request: count 0 and fully-zero mask.
        assert counts.cpu()[0].item() == 0
        assert packed_mask.cpu()[0].eq(0).all().item()
        # Active request with all-finite logits: every token is in the support.
        assert counts.cpu()[1].item() == vocab_size
        assert packed_mask.cpu()[1].eq(0xFF).all().item()

    def test_non_contiguous_logits(self):
        vocab_size = 32
        base = torch.randn(1, vocab_size * 2, dtype=torch.bfloat16, device=DEVICE_TYPE)
        logits = base[:, ::2]  # stride-2 view exercises logits_col_stride
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)
        _assert_matches(logits, num_sampled_tokens, (vocab_size + 7) // 8)

    @pytest.mark.parametrize("cudagraph_rows", [1, 8])
    def test_capture_replay(self, cudagraph_rows):
        """The kernel must be capturable in an ACLGraph and replay correctly.

        This is the regression guard for the construct that forced a device
        ``rtMemcpy`` at launch (``rtMemcpy ... not permitted when a stream is
        capturing``), which surfaced as an AIVEC vector-core timeout on the
        first graphed decode step. A kernel that only works eagerly fails here.
        """
        vocab_size = 4096
        torch.manual_seed(0)
        logits = torch.randn(cudagraph_rows, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = torch.ones(cudagraph_rows, dtype=torch.int32, device=DEVICE_TYPE)
        packed_width = (vocab_size + 7) // 8

        packed_mask = torch.empty(
            (cudagraph_rows, packed_width), dtype=torch.uint8, device=DEVICE_TYPE
        )
        counts = torch.empty(cudagraph_rows, dtype=torch.int32, device=DEVICE_TYPE)

        def launch():
            _compact_sampling_mask_kernel_ascend[(cudagraph_rows,)](
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

        # JIT outside capture, then capture, then replay -- the sequence the
        # engine itself performs.
        launch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()

        ref_packed, ref_counts = _reference(logits, num_sampled_tokens, packed_width)
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))
        assert torch.equal(packed_mask.cpu(), torch.from_numpy(ref_packed))

    def test_from_logits_patch_matches_upstream_shape(self):
        """The patched ``SamplingMaskTensors.from_logits`` must stay wired up.

        Guards the semantic half of the merge: ``SamplingMaskTensors`` is a
        4-field NamedTuple upstream, so a rebind that still builds the old
        3-field form would raise at replay time.
        """
        import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa: F401  applies the patch
        from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
        from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
            compact_sampling_mask_from_logits,
        )

        assert (
            SamplingMaskTensors.from_logits.__func__ is compact_sampling_mask_from_logits
        ), "SamplingMaskTensors.from_logits is not rebound to the Ascend implementation"

        vocab_size = 64
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)
        logits = torch.randn(1, vocab_size, dtype=torch.bfloat16, device=DEVICE_TYPE)
        tensors = SamplingMaskTensors.from_logits(logits, num_sampled_tokens, 8)
        torch.npu.synchronize()

        assert tensors.vocab_size == vocab_size
        assert tensors.packed_mask.shape == (1, (vocab_size + 7) // 8)
        assert tensors.counts.shape == (1,)
        assert tensors.counts.cpu().item() == vocab_size

        # ``to_cpu_nonblocking`` is an async D2H; production waits on the copy
        # event before ``tolists()`` (see async_utils.py), so do the same here.
        cpu_tensors = tensors.to_cpu_nonblocking()
        torch.npu.synchronize()
        lists = cpu_tensors.tolists()
        assert lists.token_ids.tolist() == list(range(vocab_size))
