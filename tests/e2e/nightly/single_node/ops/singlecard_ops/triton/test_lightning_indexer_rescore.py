# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.lightning_indexer_rescore import gather_candidate_keys_triton


@pytest.fixture
def device():
    if not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("Requires Triton on Ascend A5")
    return torch.device("npu")


@pytest.mark.parametrize(("rows", "count"), [(0, 17), (3, 0), (3, 17), (3, 2048)])
@torch.inference_mode()
def test_lightning_indexer_rescore_matches_reference(device, rows, count):
    heads, dim, page_size = 32, 128, 128
    generator = torch.Generator().manual_seed(42)
    query = torch.randn(rows, heads, dim, generator=generator).bfloat16()
    weights = torch.randn(rows, heads, generator=generator).bfloat16()
    table = torch.tensor([[3, 1], [2, -1]], dtype=torch.int32)
    sequence = torch.tensor([0, 1, 1])[:rows]
    prefix = torch.tensor([[16], [0], [0]], dtype=torch.int32)[:rows]
    end = torch.tensor([[200], [256], [0]], dtype=torch.int32)[:rows]
    cache = torch.randn(4, page_size + 2, 1, dim, generator=generator).bfloat16()
    candidates = (torch.arange(count, dtype=torch.int32) % (2 * page_size + 1) - 1).repeat(rows, 1)
    if count > 1:
        candidates[:, 1] = page_size

    table_storage = torch.empty((2, 4), dtype=torch.int32, device=device)
    candidate_storage = torch.empty((rows, 2 * count), dtype=torch.int32, device=device)
    device_table, device_candidates = table_storage[:, ::2], candidate_storage[:, ::2]
    device_table.copy_(table)
    device_candidates.copy_(candidates)
    cache = cache.to(device)[:, :page_size]
    sequence = sequence.to(device)
    actual_keys = gather_candidate_keys_triton(cache, device_table, sequence, device_candidates)
    assert actual_keys.shape == (rows, count, dim)
    assert actual_keys.dtype == torch.float32
    if not rows or not count:
        return

    query = query.to(device).float()
    weights = weights.to(device).float()
    valid = (device_candidates >= prefix.to(device)) & (device_candidates < end.to(device))
    safe = device_candidates.clamp_min(0).long()
    pages = device_table[sequence[:, None], safe // page_size].clamp_min(0).long()
    expected_keys = cache[pages, safe % page_size, 0].float()
    assert torch.equal(actual_keys, expected_keys)
    expected_scores = (torch.bmm(query, expected_keys.transpose(1, 2)).relu() * weights[:, :, None]).sum(dim=1)
    actual_scores = (torch.bmm(query, actual_keys.transpose(1, 2)).relu() * weights[:, :, None]).sum(dim=1)
    expected_scores = expected_scores.masked_fill(~valid, -torch.inf)
    actual_scores = actual_scores.masked_fill(~valid, -torch.inf)
    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_scores.topk(count).indices, expected_scores.topk(count).indices)
