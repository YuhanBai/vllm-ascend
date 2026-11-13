# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.lightning_indexer_rescore import rescore_candidates_triton


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
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    sequence = torch.tensor([0, 1, 1])[:rows]
    prefix = torch.tensor([[16], [0], [0]], dtype=torch.int32)[:rows]
    end = torch.tensor([[200], [256], [0]], dtype=torch.int32)[:rows]
    cache = torch.randn(4, page_size + 2, 1, dim, generator=generator).bfloat16()
    candidates = (torch.arange(count, dtype=torch.int32) % (2 * page_size + 2) - 1).repeat(rows, 1)
    if count > 1:
        candidates[:, 1] = page_size

    valid = (candidates >= prefix) & (candidates < end)
    safe = torch.where(valid, candidates, 0).long()
    pages = table[sequence[:, None], safe // page_size].long()
    keys = cache[pages, safe % page_size, 0].float()
    dot = torch.bmm(query.float(), keys.transpose(1, 2))
    expected = (dot.relu() * weights.float()[:, :, None]).sum(dim=1).masked_fill(~valid, -torch.inf)

    table_storage = torch.empty((2, 4), dtype=torch.int32, device=device)
    candidate_storage = torch.empty((rows, 2 * count), dtype=torch.int32, device=device)
    device_table, device_candidates = table_storage[:, ::2], candidate_storage[:, ::2]
    device_table.copy_(table)
    device_candidates.copy_(candidates)
    result = rescore_candidates_triton(
        query.to(device),
        cache.to(device)[:, :page_size],
        weights.to(device),
        device_table,
        sequence.to(device),
        device_candidates,
        prefix.to(device),
        end.to(device),
    ).cpu()
    torch.testing.assert_close(result.isneginf(), ~valid)
    torch.testing.assert_close(result[valid], expected[valid], rtol=2e-5, atol=3e-4)
