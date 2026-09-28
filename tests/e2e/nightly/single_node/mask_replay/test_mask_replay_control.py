#!/usr/bin/env python
"""Mask-replay end-to-end test -- phase 2 (mask OFF control).

Same model, TP and graph mode, but with ``return_sampling_mask=False``. Verifies
that turning mask replay on does not change what the model produces, by
comparing against the token stream phase 1 dumped.
"""

import json
import os
import sys
import time
import traceback

import torch

MODEL = os.environ.get("MODEL_PATH", "/home/data/weights/Qwen3-30B-A3B")
TP = int(os.environ.get("TP_SIZE", "4"))
TOP_K = int(os.environ.get("TOP_K", "20"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.90"))
MODE = os.environ.get("CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
TOKENS_IN = os.environ.get("TOKENS_OUT", "/tmp/mask_on_tokens.json")

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Explain in one sentence why the sky is blue:",
    "Q: What is the largest planet in the solar system?\nA:",
    "Write a haiku about the ocean.",
    "def fibonacci(n):",
]


def main():
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams

    print("=" * 78, flush=True)
    print("MASK-REPLAY E2E (phase 2, mask OFF control)  TP=%d graph=%s" % (TP, MODE), flush=True)
    print("=" * 78, flush=True)

    with open(TOKENS_IN) as f:
        mask_on = json.load(f)

    devs = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    n_vis = len([d for d in devs.split(",") if d.strip()]) or None

    t0 = time.time()
    llm = LLM(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM, trust_remote_code=True,
        seed=1234, max_num_seqs=64,
        device_ids=list(range(n_vis)) if n_vis else None,
        compilation_config={"cudagraph_mode": MODE,
                            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
    )
    print("    built in %.1fs" % (time.time() - t0), flush=True)

    sp = SamplingParams(temperature=1.0, top_k=TOP_K, top_p=1.0,
                        max_tokens=MAX_TOKENS, seed=7)
    t = time.time()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print("    [mask-off] %d reqs / %d tokens in %.1fs" % (len(outs), n, time.time() - t), flush=True)

    same = 0
    diffs = []
    for o in outs:
        a = mask_on.get(o.prompt)
        b = list(o.outputs[0].token_ids)
        if a == b:
            same += 1
        else:
            diffs.append((o.prompt[:36], (a or [])[:10], b[:10]))

    ok = same == len(outs)
    print("\n  [%s] token stream identical with and without mask replay (%d/%d requests)"
          % ("PASS" if ok else "FAIL", same, len(outs)), flush=True)
    for d in diffs[:3]:
        print("      diff: %r\n        mask_on =%s\n        mask_off=%s" % d, flush=True)

    print("\nPHASE2 RESULT: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
