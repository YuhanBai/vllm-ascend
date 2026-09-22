#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#
# =============================================================================
# Sampling-mask replay (mask_replay) server launcher -- vLLM Ascend
# =============================================================================
#
# Default topology: Qwen3-30B-A3B (MoE), TP=4, DP=1, expert parallel enabled.
#
# WHAT THIS ENABLES
#   For every generated token the server also returns the exact token-id
#   support set that survived top-k / top-p / min-p truncation: the "sampling
#   mask". RL training (e.g. GRPO) uses it to normalize pi_theta over the same
#   nucleus that pi_old actually sampled from, removing the off-policy
#   mismatch. Upstream reference: vllm/docs/training/sampling_mask.md
#
# HARD REQUIREMENTS (vLLM rejects the config at startup otherwise)
#   1. VLLM_USE_V2_MODEL_RUNNER=1
#      The mask producer exists only in the Model Runner V2 tree
#      (vllm/v1/worker/gpu/...). vLLM raises
#      "sampling distribution replay requires Model Runner V2" when v2 is off.
#      On Ascend this env var is the ONLY switch: patch_use_v2_model_runner.py
#      rebinds VllmConfig.use_v2_model_runner to read it directly and returns
#      False when unset (upstream would auto-enable v2 for some architectures).
#   2. --return-sampling-mask
#   3. --logprobs-mode processed_logprobs
#      Returned logprobs must be normalized over the same nucleus as the mask.
#   4. No speculative decoding, no diffusion model, no --logits-processors.
#   5. Request side: temperature > 0 and top_k > 0.
#
# WHY AN ASCEND PATCH IS NEEDED AT ALL
#   vllm_ascend/ops/triton/v2/sample/pack_sampling_mask.py replaces the
#   upstream packing kernel because
#     (a) triton-ascend does not upcast the int1 result of tl.sum to int32,
#         which truncates `counts` to 0/1, and
#     (b) the upstream launcher hard-codes BLOCK_SIZE=8192, which the NPU
#         backend cannot launch; 1024 is used instead.
#   Both symbols are rebound in
#   vllm_ascend/patch/worker/patch_v2/patch_triton.py.
#
# WHERE THE MASK COMES BACK
#   POST /inference/v1/generate  -> choices[].sampling_mask
#   The OpenAI-compatible /v1/chat/completions endpoint does NOT expose it.
#
# USAGE
#   ./run_mask_replay_server.sh
#   MODEL_PATH=/data/models/Qwen3-30B-A3B ./run_mask_replay_server.sh
#   MODEL_PATH=vllm-ascend/Qwen3-30B-A3B-W8A8 QUANTIZATION=ascend \
#       ./run_mask_replay_server.sh
# =============================================================================

set -euo pipefail

# ------------------------------- user knobs ---------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

TP_SIZE="${TP_SIZE:-4}"
DP_SIZE="${DP_SIZE:-1}"
# 1 -> --enable-expert-parallel. Ascend requires EP to spread the MoE experts
# of Qwen3-30B-A3B across the NPUs of the TP group.
ENABLE_EP="${ENABLE_EP:-1}"

ASCEND_DEVICES="${ASCEND_DEVICES:-0,1,2,3}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-37364}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-100}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

# "ascend" for W8A8 / quantized checkpoints, empty for BF16.
QUANTIZATION="${QUANTIZATION:-}"

# Repository default for this model is prefix caching OFF. Set to 1 to enable.
# Note it in your benchmark notes if you do: random synthetic data yields a
# zero hit rate, and cache hits change the sampled distributions you replay.
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"

# Optional perf knob (0/1/2, empty = vLLM Ascend default of 1). Kept off by
# default so a mask-replay accuracy issue cannot be confused with NZ layout.
WEIGHT_NZ_MODE="${WEIGHT_NZ_MODE:-}"

# ------------------------- environment (Ascend / NPU) -----------------------
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_DEVICES}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"   # required on A3, not needed on A2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"

# >>> The single most important line in this script. <<<
# Without it vLLM raises "sampling distribution replay requires Model Runner V2".
export VLLM_USE_V2_MODEL_RUNNER=1

# ------------------------------ build argv ----------------------------------
ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --trust-remote-code

    # ---- parallelism: TP4 / DP1 / EP on ----
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --distributed-executor-backend mp

    # ---- scheduling / memory ----
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

    # ---- mask-replay: the two required flags ----
    --return-sampling-mask
    --logprobs-mode processed_logprobs
)

if [[ "${ENABLE_EP}" == "1" ]]; then
    ARGS+=(--enable-expert-parallel)
fi

if [[ "${ENABLE_PREFIX_CACHING}" != "1" ]]; then
    ARGS+=(--no-enable-prefix-caching)
fi

if [[ -n "${QUANTIZATION}" ]]; then
    ARGS+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${WEIGHT_NZ_MODE}" ]]; then
    ARGS+=(--additional-config "{\"weight_nz_mode\": ${WEIGHT_NZ_MODE}}")
fi

# Deliberately NOT passed:
#   --speculative-config   mask-replay rejects speculative decoding
#   --logits-processors    mask-replay rejects custom logits processors

# -------------------------------- launch ------------------------------------
cat <<EOF
=============================================================================
 vLLM Ascend -- sampling-mask replay (mask_replay)
-----------------------------------------------------------------------------
 model            : ${MODEL_PATH}
 served name      : ${SERVED_MODEL_NAME}
 listen           : ${HOST}:${PORT}
 topology         : TP=${TP_SIZE}  DP=${DP_SIZE}  EP=$([[ "${ENABLE_EP}" == "1" ]] && echo on || echo off)
 devices          : ${ASCEND_RT_VISIBLE_DEVICES}
 max-model-len    : ${MAX_MODEL_LEN}
 max-num-seqs     : ${MAX_NUM_SEQS}
 quantization     : ${QUANTIZATION:-<none / BF16>}
 prefix caching   : $([[ "${ENABLE_PREFIX_CACHING}" == "1" ]] && echo on || echo off)
 model runner     : V2 (VLLM_USE_V2_MODEL_RUNNER=1)
 mask replay      : ON  (--return-sampling-mask, logprobs-mode=processed_logprobs)
=============================================================================
EOF

exec vllm serve "${MODEL_PATH}" "${ARGS[@]}"

# =============================================================================
# VERIFICATION  (substitute your own PORT / SERVED_MODEL_NAME)
# =============================================================================
#
# 1) Liveness. Works, but returns NO mask -- the OpenAI-compatible schema has
#    no sampling_mask field:
#
#    curl -s http://localhost:8000/v1/chat/completions \
#      -H 'Content-Type: application/json' \
#      -d '{"model":"qwen3",
#           "messages":[{"role":"user","content":"hi"}],
#           "max_tokens":16}'
#
# 2) The mask itself. /inference/v1/generate takes RAW token ids, not text,
#    so tokenize first:
#
#    python -c "
#    from transformers import AutoTokenizer
#    t = AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B')
#    print(t('The capital of France is', add_special_tokens=False)['input_ids'])
#    "
#
#    then POST those ids:
#
#    curl -s http://localhost:8000/inference/v1/generate \
#      -H 'Content-Type: application/json' \
#      -d '{"model":"qwen3",
#           "token_ids":[785,6722,315,9625,374],
#           "sampling_params":{"temperature":1.0,"top_p":0.95,"top_k":50,
#                              "max_tokens":16},
#           "stream":false}'
#
#    Expected: len(choices[0].token_ids) == len(choices[0].sampling_mask),
#    i.e. one support set per generated token.
#
# 3) Sanity checks on the mask contents
#    - 0 < len(mask[i]) <= top_k when top_k > 0.
#    - The sampled token id of step i is a member of mask[i].
#    - Rows are not all-zero and counts are not stuck at 0/1.
#
#    An all-zero mask with counts stuck at 0/1 is the signature of the
#    upstream (broken) kernel still being in use: triton-ascend does not
#    upcast the int1 tl.sum result. Check that
#    vllm_ascend.patch.worker.patch_v2.patch_triton actually got imported --
#    it is loaded under `if HAS_TRITON:` in
#    vllm_ascend/patch/worker/__init__.py, so a triton-less environment
#    silently skips it.
# =============================================================================
