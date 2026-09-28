#!/usr/bin/env python
"""Mask-replay end-to-end test -- phase 1 (mask ON).

Runs the real engine with ``return_sampling_mask=True`` and checks the mask
invariants, then dumps the token streams to JSON so phase 2 can compare a
mask-OFF engine in a separate process (building two engines in one process
trips a PyTorch OpenMP thread-pool assert).
"""

import json
import os
import sys
import time
import traceback
from collections import Counter

import torch

MODEL = os.environ.get("MODEL_PATH", "/home/data/weights/Qwen3-30B-A3B")
TP = int(os.environ.get("TP_SIZE", "4"))
TOP_K = int(os.environ.get("TOP_K", "20"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
GPU_MEM = float(os.environ.get("GPU_MEM", "0.90"))
MODE = os.environ.get("CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
REPORT = os.environ.get("REPORT", "/tmp/mask_replay_report.json")
TOKENS_OUT = os.environ.get("TOKENS_OUT", "/tmp/mask_on_tokens.json")

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "Explain in one sentence why the sky is blue:",
    "Q: What is the largest planet in the solar system?\nA:",
    "Write a haiku about the ocean.",
    "def fibonacci(n):",
]

FAILURES = []


def check(name, cond, detail=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, detail), flush=True)
    if not cond:
        FAILURES.append("%s %s" % (name, detail))
    return cond


def main():
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams
    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa: F401
    from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
    from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
        compact_sampling_mask_from_logits,
    )

    print("=" * 78, flush=True)
    print("MASK-REPLAY E2E (phase 1, mask ON)  TP=%d EP=on graph=%s top_k=%d"
          % (TP, MODE, TOP_K), flush=True)
    print("=" * 78, flush=True)

    print("\n[R1] replay config + patch wiring", flush=True)
    check("SamplingMaskTensors.from_logits -> Ascend impl",
          SamplingMaskTensors.from_logits.__func__ is compact_sampling_mask_from_logits,
          "-> %s" % SamplingMaskTensors.from_logits.__func__.__name__)

    devs = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    n_vis = len([d for d in devs.split(",") if d.strip()]) or None

    t0 = time.time()
    llm = LLM(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM, trust_remote_code=True,
        return_sampling_mask=True, logprobs_mode="processed_logprobs",
        seed=1234, max_num_seqs=64,
        device_ids=list(range(n_vis)) if n_vis else None,
        compilation_config={"cudagraph_mode": MODE,
                            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
    )
    vc = llm.llm_engine.vllm_config
    print("    built in %.1fs | graph=%s | v2=%s | tp=%d ep=%s"
          % (time.time() - t0, vc.compilation_config.cudagraph_mode,
             vc.use_v2_model_runner, vc.parallel_config.tensor_parallel_size,
             vc.parallel_config.enable_expert_parallel), flush=True)
    check("Model Runner V2 active", vc.use_v2_model_runner is True)
    check("graph mode active (not NONE)",
          str(vc.compilation_config.cudagraph_mode) != "NONE",
          "(%s)" % vc.compilation_config.cudagraph_mode)
    check("expert parallelism active", vc.parallel_config.enable_expert_parallel is True)
    check("tensor parallel size == %d" % TP,
          vc.parallel_config.tensor_parallel_size == TP)

    sp = SamplingParams(temperature=1.0, top_k=TOP_K, top_p=1.0,
                        max_tokens=MAX_TOKENS, logprobs=TOP_K, seed=7)

    def run(tag):
        t = time.time()
        outs = llm.generate(PROMPTS, sp, use_tqdm=False)
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        print("    [%s] %d reqs / %d tokens in %.1fs" % (tag, len(outs), n, time.time() - t),
              flush=True)
        return outs

    print("\n[R2-R4] mask invariants", flush=True)
    outs = run("mask-on")

    total = missing = not_in_mask = empty = dups = 0
    size_hist = Counter()
    per_req = []
    for o in outs:
        co = o.outputs[0]
        toks = list(co.token_ids)
        sm = co.sampling_mask
        if sm is None:
            missing += 1
            per_req.append(dict(prompt=o.prompt[:40], gen=len(toks), masks=None))
            continue
        masks = sm.token_ids
        bad_here = 0
        for i, tok in enumerate(toks):
            if i >= len(masks):
                break
            total += 1
            sup = masks[i]
            size_hist[len(sup)] += 1
            if not sup:
                empty += 1
            if tok not in sup:
                not_in_mask += 1
                bad_here += 1
            if len(set(sup)) != len(sup):
                dups += 1
        per_req.append(dict(prompt=o.prompt[:40], gen=len(toks), masks=len(masks),
                            bad=bad_here))

    upper = TOP_K * TP
    oversize = sum(c for s, c in size_hist.items() if s > upper)
    check("all requests returned a sampling_mask", missing == 0, "(missing=%d)" % missing)
    check("positions checked > 0", total > 0, "(total=%d)" % total)
    check("no empty support sets", empty == 0, "(empty=%d)" % empty)
    check("sampled token always inside its replay mask",
          not_in_mask == 0, "(violations=%d / %d)" % (not_in_mask, total))
    check("no support wider than top_k * tp_size (%d)" % upper,
          oversize == 0, "(oversize=%d)" % oversize)
    check("no duplicate ids inside a support set", dups == 0, "(dups=%d)" % dups)
    print("    support-size histogram: %s" % dict(sorted(size_hist.items())), flush=True)

    print("\n[R6] support non-degeneracy", flush=True)
    single = size_hist.get(1, 0)
    union = set()
    for o in outs:
        sm = o.outputs[0].sampling_mask
        if sm is not None:
            for sup in sm.token_ids:
                union.update(sup)
    check("union of supports is wide (not collapsed)", len(union) > 20,
          "(distinct ids=%d)" % len(union))
    check("support is never a single candidate", single == 0,
          "(single-candidate steps=%d)" % single)

    print("\n[sample of returned masks]", flush=True)
    co = outs[0].outputs[0]
    if co.sampling_mask is not None:
        m = co.sampling_mask.token_ids
        toks = list(co.token_ids)
        print("    prompt: %r" % outs[0].prompt[:50], flush=True)
        for i in range(min(4, len(m))):
            print("      step %d: sampled=%-7d support_size=%-4d first_ids=%s"
                  % (i, toks[i], len(m[i]), m[i][:8]), flush=True)

    with open(TOKENS_OUT, "w") as f:
        json.dump({o.prompt: list(o.outputs[0].token_ids) for o in outs}, f)

    report = dict(tp=TP, graph_mode=str(vc.compilation_config.cudagraph_mode),
                  top_k=TOP_K, total_positions=total, size_hist=dict(size_hist),
                  union_size=len(union), per_req=per_req, failures=FAILURES)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    if FAILURES:
        print("PHASE1 RESULT: FAIL (%d)" % len(FAILURES), flush=True)
        for f in FAILURES:
            print("   - %s" % f, flush=True)
        return 1
    print("PHASE1 RESULT: PASS -- mask replay functional (TP%d + EP + graph %s)"
          % (TP, MODE), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
