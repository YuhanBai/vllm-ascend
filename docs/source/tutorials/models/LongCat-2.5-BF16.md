# LongCat-2.5 BF16 text inference

This experimental path loads the original `meituan-longcat/LongCat-2.5` checkpoint on
Ascend with Model Runner V2. Prefill runs in eager mode. Uniform decode can
use ACLGraph via `FULL_DECODE_ONLY`.
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

For ACLGraph decode, replace `--enforce-eager` with the following option.
The `mode` value keeps Torch compilation disabled while enabling ACLGraph
capture and replay.

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'
```

For the initial validation, send a short text prompt and a prompt with more
than 2048 input tokens. Generate multiple tokens and compare against the
FluentLLM BF16 path with the same token IDs. The longer prompt exercises the
mandatory initial and recent-token selection in the sparse indexer.

With the server running, the opt-in smoke test exercises both context lengths.

```bash
LONGCAT25_SMOKE_BASE_URL=http://127.0.0.1:8000/v1 \
  pytest -q tests/e2e/models/test_longcat_25_bf16.py
```

For graph validation, point `LONGCAT25_SMOKE_BASE_URL` to the graph server and
`LONGCAT25_SMOKE_EAGER_BASE_URL` to an eager server using the same checkpoint
and parallel layout. The test compares the generated token IDs. Check the
graph server log for `ACL graph replay is active for the V2 target model`.

Images, video, MTP, KV transfer, context parallelism, sequence parallelism,
prefill graph capture, and the W4A4 checkpoint are outside this text path.
