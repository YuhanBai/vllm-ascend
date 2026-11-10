# LongCat-2.5 BF16 text inference

This experimental path loads the original `meituan-longcat/LongCat-2.5` checkpoint on
Ascend with Model Runner V2. It covers eager text prefill and generation.
The checkpoint has approximately 3.56 TB of weights. Plan the tensor, expert
and pipeline parallel layout around the per-rank weight and KV allocation.
The first pipeline stage owns the sharded over-embedding tables.

The BF16 indexer requires `npu_lightning_indexer` with both `init_tensor` and
`local_tensor` in its operator schema. If these are absent from the installed
`torch_npu`, install the matching `flash_ops` wheel and activate its CANN OPP
as described in the [FluentLLM NPU setup](https://github.com/meituan-longcat/SGLang-FluentLLM/blob/790693d2abe6af6925722fef06fefb6d7488a294/npu_test/ReadMe.md).
Use the vLLM revision recorded in `.github/vllm-main-verified.commit`.

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name longcat-2.5-bf16 \
  --dtype bfloat16 --enforce-eager \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --enable-expert-parallel --max-model-len 4096
```

Set `MODEL_PATH`, `TP`, and `PP` for the BF16 checkpoint and deployed ranks.
For the initial validation, send a short text prompt and a prompt with more
than 2048 input tokens. Generate multiple tokens and compare against the
FluentLLM BF16 path with the same token IDs. The longer prompt exercises the
mandatory initial and recent-token selection in the sparse indexer.

With the server running, the opt-in smoke test exercises both context lengths.

```bash
LONGCAT25_SMOKE_BASE_URL=http://127.0.0.1:8000/v1 \
  pytest -q tests/e2e/models/test_longcat_25_bf16.py
```

Images, video, MTP, KV transfer, context parallelism, sequence parallelism,
graph capture, and the W4A4 checkpoint are outside this text path.
