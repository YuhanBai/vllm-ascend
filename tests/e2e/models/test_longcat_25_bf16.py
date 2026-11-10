# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in generation smoke test for an already running multi-node BF16 deployment."""

import os

import pytest
from openai import OpenAI


@pytest.mark.parametrize("prompt_length", [32, 2304])
def test_longcat_25_bf16_generation(prompt_length):
    base_url = os.environ.get("LONGCAT25_SMOKE_BASE_URL")
    if not base_url:
        pytest.skip("Set LONGCAT25_SMOKE_BASE_URL to the LongCat-2.5 server's /v1 URL.")

    def generate(endpoint):
        with OpenAI(base_url=endpoint, api_key="EMPTY") as client:
            result = client.completions.create(
                model="longcat-2.5-bf16",
                prompt=[1] + [42] * (prompt_length - 1),
                temperature=0,
                max_tokens=4,
                extra_body={"ignore_eos": True, "return_token_ids": True},
            )
        assert result.usage is not None
        assert result.usage.prompt_tokens == prompt_length
        assert result.usage.completion_tokens == 4
        token_ids = result.choices[0].model_extra.get("token_ids")
        assert token_ids is not None and len(token_ids) == 4
        return token_ids

    output = generate(base_url)
    if eager_url := os.environ.get("LONGCAT25_SMOKE_EAGER_BASE_URL"):
        assert output == generate(eager_url)
