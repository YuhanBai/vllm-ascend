# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.over_embedding_ids import over_embedding_ids_triton


@pytest.fixture
def device():
    if not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("Requires Triton on Ascend A5")
    return torch.device("npu")


@pytest.mark.parametrize("rows", [0, 1, 9, 2057])
@torch.inference_mode()
def test_over_embedding_ids_matches_reference(device, rows):
    vocab_size, eos = 163840, 2
    # LongCat 2.5 has four splits for each of the 2–5 gram orders.
    moduli = torch.arange(16476898, 16476930, 2, dtype=torch.int32).reshape(4, 4)
    weights = torch.tensor(
        [[[pow(vocab_size, delta, int(m)) for delta in range(5)] for m in group] for group in moduli],
        dtype=torch.int32,
    )
    offsets = torch.cat((moduli.new_zeros(1), moduli.flatten().cumsum(0, dtype=torch.int32)))
    tokens = torch.tensor([vocab_size - 1, eos, 35, 301], dtype=torch.int64)
    history = torch.tensor(
        [[163830, 163829, 163828, 163827], [7, eos, 123, 456], [-1, 12, 90, 88], [9, 8, -1, 55]],
        dtype=torch.int32,
    )
    pattern = torch.arange(rows) % len(tokens)
    tokens, history = tokens[pattern], history[pattern]
    # The high-ID row requires an intermediate wider than int32. The
    # reference stops at EOS or the request boundary before adding older IDs.
    valid = ((history >= 0) & (history != eos)).int().cumprod(-1).bool()
    all_tokens = torch.cat((tokens[:, None], history.masked_fill(~valid, 0)), dim=-1)
    orders = torch.arange(2, 6)
    factors = (weights * (torch.arange(5) < orders[:, None])[:, None, :]).reshape(16, 5)
    expected = ((all_tokens[:, None, :] * factors).sum(-1).remainder(moduli.flatten()) + offsets[:-1]).int()

    token_storage = torch.empty((rows, 2), dtype=tokens.dtype, device=device)
    history_storage = torch.empty((rows, 8), dtype=history.dtype, device=device)
    input_ids, input_history = token_storage[:, 0], history_storage[:, ::2]
    input_ids.copy_(tokens)
    input_history.copy_(history)
    result = over_embedding_ids_triton(
        input_ids, input_history, eos, weights.to(device), moduli.to(device), offsets.to(device)
    )
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
