# SFT Training

## M6 smoke test

M6 uses full-parameter bf16 fine-tuning of the pinned Qwen3-0.6B-Base revision. The
loss is computed only on the normalized response and EOS token; prompt tokens and
padding are masked with `-100`. Samples are never silently truncated: a sample over
the configured 16,384-token total-length limit is rejected.

The cloud entry point is `scripts/cloud_train_sft.sh`. It uses the verl virtual
environment and the project-local uv/Hugging Face caches. `SFT_GPU_COUNT` controls
the number of local processes. Multi-GPU runs use PyTorch DDP; the script gives all
ranks one shared timestamp and therefore one artifact directory. The entry point
also enables PyTorch expandable CUDA allocator segments because the variable-length
16K SFT workload otherwise can accumulate enough reserved fragmentation to fail a
later long-sequence backward pass even when total VRAM is sufficient.

Run the correctness/overfit smoke test on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 SFT_GPU_COUNT=1 \
  bash scripts/cloud_train_sft.sh smoke \
  2>&1 | tee /tmp/qwen3-m6-smoke.log
```

The smoke configuration uses the 64 shortest SFT-1K samples for 100 optimizer
steps, global batch 8, and writes checkpoints every 25 steps. A checkpoint contains
model, optimizer, scheduler, RNG, and Trainer state. Resume an interrupted run with:

```bash
SFT_GPU_COUNT=1 bash scripts/cloud_train_sft.sh smoke \
  --resume outputs/training/<run>/checkpoint-<step>
```

After the smoke test, reload the saved model and generate once:

```bash
.third_party/verl/.venv/bin/python scripts/verify_sft_checkpoint.py \
  outputs/training/<run>/final
```

Before M7, benchmark 1, 2, and 4 GPUs separately with the throughput configuration.
It keeps global batch 16 fixed, so the comparison does not change optimization
semantics:

```bash
CUDA_VISIBLE_DEVICES=0 SFT_GPU_COUNT=1 bash scripts/cloud_train_sft.sh throughput
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 bash scripts/cloud_train_sft.sh throughput
CUDA_VISIBLE_DEVICES=0,1,2,3 SFT_GPU_COUNT=4 bash scripts/cloud_train_sft.sh throughput
```

Each run writes `config.yaml`, `environment.json`, `train_metrics.json`, Trainer
state/checkpoints when enabled, and `final/` under `outputs/training/`. Record loss,
runtime, steps/tokens per second, peak GPU memory, GPU model/count, git commit, and
the artifact path. M7 formal SFT-1K starts only after M6 passes and the card count is
chosen from these measurements.

M6 selected 2-GPU DDP for formal SFT: it delivered about 1.62x single-GPU speedup
and 81% parallel efficiency while 4 GPUs delivered 2.55x speedup and 64% efficiency.

## M7 SFT-1K

M7 is an independent run from the pinned Base checkpoint, not a continuation of
the M6 overfit model. The frozen recipe uses all 1,000 ordered SFT samples, 3 epochs,
global batch 16, cosine learning-rate decay from 2e-5 with 3% warmup, and 2-GPU DDP.
The 3% warmup is frozen as 6 steps for the expected 189 optimizer steps. It saves
one checkpoint per epoch.

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh sft1k \
  2>&1 | tee /tmp/qwen3-m7-sft1k.log
```

If interrupted, resume from the newest checkpoint with the same mode and GPU count:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh sft1k \
  --resume outputs/training/<m7-run>/checkpoint-<step>
```

Reload the final checkpoint, then run the fixed 10-problem smoke before the full
399-problem evaluation:

```bash
.third_party/verl/.venv/bin/python scripts/verify_sft_checkpoint.py \
  outputs/training/<m7-run>/final

CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_m7.sh \
  smoke outputs/training/<m7-run>/final

CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_m7.sh \
  full outputs/training/<m7-run>/final
```

For the 0.6B model, use problem-level sharding to occupy both GPUs rather than
tensor parallelism. Each GPU runs an independent single-GPU vLLM over alternating
problem IDs. The launcher validates both shard configs, rejects missing or duplicate
problem/sample records, restores frozen manifest order, and computes metrics over
the merged 399 records:

```bash
bash scripts/cloud_eval_m7.sh \
  sharded-full outputs/training/<m7-run>/final \
  2>&1 | tee /tmp/qwen3-m7-eval-sharded-full.log
```

Shard logs are `/tmp/qwen3-m7-eval-shard-{0,1}.log`; shard directories and the
merged formal directory are all retained under `outputs/eval/`.

The full evaluation protocol is identical to M4: the same frozen 399 problems,
greedy pass@1, 32,768-token context, and 16,384-token generation cap.

### M7-v2 bounded-response pilot

M7-v1 failed its gate: even checkpoint-63 produced 10/10 length-capped smoke
responses with zero extraction. Do not formally evaluate any M7-v1 checkpoint.
M7-v2 changes only the response-length data recipe before the pilot: derive 1,000
ordered samples with `response <= 4096` from the frozen SFT-10K, without truncation.
The preparation command verifies the frozen input SHA-256 and deterministically
writes the same output (`32383cd65d4fc74ec9ec8c8a055665db786622173f404cdc5c5ba695ae5d7ff6`):

```bash
.third_party/verl/.venv/bin/python scripts/prepare_sft_short.py
```

The derived set has response P50 2,044, P90 3,607, max 4,088 and total-token max
5,039. The first gate trains only its first 256 rows for one epoch from Base:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh sft1k-short-pilot \
  2>&1 | tee /tmp/qwen3-m7-short-pilot.log
```

Run `verify_sft_checkpoint.py` and the fixed 10-problem smoke on the pilot final.
Only if responses close normally and code extraction recovers should a full bounded
SFT-1K configuration be created.

The 4,096-token pilot improved but failed the gate under both greedy and sampled
diagnostics. Derive the next controlled 2,048-token variant (still without
truncation), then run its 256-row one-epoch pilot:

```bash
.third_party/verl/.venv/bin/python scripts/prepare_sft_short.py \
  --response-max-tokens 2048 \
  --output data/processed/sft_1k_compact_v1.jsonl \
  --stats data/processed/sft_1k_compact_v1_stats.json

CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh sft1k-compact-pilot \
  2>&1 | tee /tmp/qwen3-m7-compact-pilot.log
```

The deterministic compact output SHA-256 is
`ab1755fa5540b5e4f1873ee7bb7f4bb445f7573258772d1997386daa17e25e74`;
response P50/P90/max are 1,308/1,885/2,048 tokens and total max is 2,875.
Only one hard-labeled sample survives in the first 1K, so this is strictly a
stopping/format diagnostic, not a candidate final difficulty-balanced dataset.

The one-epoch compact pilot collapses on both one and two GPUs, while the original
M6 model recovers much of its format behavior after about 12.5 epochs. Measure the
recovery curve with a fresh four-epoch compact-256 run. It has about 16 optimizer
steps per epoch and retains checkpoints 16/32/48/64:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh compact-4epoch \
  2>&1 | tee /tmp/qwen3-m7-compact-4epoch.log
```

Evaluate every checkpoint with the same greedy smoke; do not select an epoch from
training loss alone. This run remains a diagnostic and does not authorize a formal
399-problem evaluation.

## Official Qwen3-0.6B capacity diagnostic

To separate model-size limitations from failures in this project's SFT recipe, run
the official post-trained `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca` on the fixed smoke set. This is not a
Base/SFT score comparison: the diagnostic correctly uses the official model's chat
template and thinking mode, plus sampled decoding (temperature 0.6, top-p 0.95,
top-k 20) with a 4,096-token cap.

```bash
bash scripts/local_eval_official_qwen3.sh \
  2>&1 | tee /tmp/qwen3-official-0.6b-smoke.log
```

If the official model produces stable code at the same parameter count, capacity
alone does not explain our collapse. Its metrics must not be placed in the frozen
M4/M7 greedy comparison table because both post-training and decoding protocol
differ.

## M7-v3 matched response-format data

The long-output diagnostics motivate a matched data experiment rather than another
epoch or generation-limit change. Derive two variants from the frozen SFT-10K with
the same 1,000 problem IDs:

- `short_reasoning_v2` retains only complete, naturally short reasoning targets;
- `code_only_v2` changes both prompt and target to request and emit only fenced C++.

No response is truncated or teacher-rewritten. Eligible source rows must have
reasoning at most 2,048 tokens, response at most 4,096 tokens, and prompt plus
response at most 8,192 tokens. Selection uses seed `20260823` and the existing
difficulty/platform-balanced ordering. The command verifies the frozen SFT-10K
SHA-256 before writing either variant:

```bash
bash scripts/cloud_prepare_sft_ab.sh \
  2>&1 | tee /tmp/qwen3-prepare-sft-ab.log
```

This entry point selects `cache/huggingface/` when present and forces offline
tokenizer loading. It never contacts Hugging Face. If the pinned tokenizer is not
in either the project or default cache, it exits with an explicit error; network
access requires deliberately running `prepare_sft_ab.py --allow-tokenizer-download`.

Generated, Git-ignored artifacts are:

- `data/processed/sft_1k_short_reasoning_v2.jsonl`;
- `data/processed/sft_1k_code_only_v2.jsonl`;
- `data/processed/sft_1k_ab_v2_manifest.json`.

The manifest freezes ordered problem IDs, source and output hashes, tokenizer
revision, selection limits, and length/difficulty/platform statistics. Both
variants must have identical ordered problem IDs before training. These data are
an A/B diagnostic; their five hard-labeled rows are insufficient for a final
difficulty-balanced training recipe.

Train the two matched 256-sample pilots independently from the same frozen Base
checkpoint. Both use two-GPU DDP, global batch 16, learning rate 2e-5, four epochs,
and retain one checkpoint per epoch (expected steps 16/32/48/64):

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh ab-short-pilot \
  2>&1 | tee /tmp/qwen3-m7-ab-short-train.log

CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh ab-code-pilot \
  2>&1 | tee /tmp/qwen3-m7-ab-code-train.log
```

Do not compare only final checkpoints. Evaluate checkpoints 16, 32, 48, and 64
with the matching prompt protocol. The short model receives Output Protocol v1 and
a 4,096-token generation cap; the code-only model receives the same direct-code
prompt used in training and a 2,048-token cap. Both use greedy decoding and the
same frozen 10-problem smoke set:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
  short outputs/training/<short-run>/checkpoint-16

CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
  code outputs/training/<code-run>/checkpoint-16
```

Repeat each command for checkpoints 32, 48, and 64. Select candidates using the
fixed smoke metrics and response inspection, not training loss. This native-prompt
A/B measures whether each response recipe can learn its intended behavior; it is
not directly comparable to the frozen M4 Base score until a common-prompt follow-up
is run.

### Full-1K diversity control

The 256-sample four-epoch pilots show that code-only targets immediately fix
stopping and compilation but do not solve a complete smoke problem, while short
reasoning recovers only after repeated exposure and retains mechanical loops. Run
one epoch over all 1,000 matched IDs before mixing the recipes. This keeps total
token exposure close to the pilot while increasing unique-problem coverage fourfold:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh ab-short-1k \
  2>&1 | tee /tmp/qwen3-m7-ab-short-1k-train.log

CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh ab-code-1k \
  2>&1 | tee /tmp/qwen3-m7-ab-code-1k-train.log
```

With global batch 16, each run is expected to end at checkpoint 63. Evaluate the
saved checkpoint (or the identical final weights) using the matching native prompt:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
  short outputs/training/<short-1k-run>/checkpoint-63 \
  2>&1 | tee /tmp/qwen3-m7-ab-short-1k-eval.log

CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
  code outputs/training/<code-1k-run>/checkpoint-63 \
  2>&1 | tee /tmp/qwen3-m7-ab-code-1k-eval.log
```

Do not construct a mixed dataset until this diversity control is inspected. A mix
would otherwise confound response format with unique-problem coverage.

The equal-token diversity control shows that one exposure to each short-reasoning
target is insufficient: 9/10 smoke generations reach the 4,096-token cap even
though optimization is numerically stable. Code-only remains format-stable but has
zero pass@1, so stop extending that branch. Run a fresh four-epoch short-reasoning
1K experiment from Base, with cosine decay spanning all 252 optimizer steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh ab-short-1k-4epoch \
  2>&1 | tee /tmp/qwen3-m7-ab-short-1k-4epoch-train.log
```

Expected epoch checkpoints are 63, 126, 189, and 252. Evaluate all four with the
same native short-reasoning smoke protocol:

```bash
for STEP in 63 126 189 252; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
    short outputs/training/<short-1k-4epoch-run>/checkpoint-${STEP}
done 2>&1 | tee /tmp/qwen3-m7-ab-short-1k-4epoch-eval.log
```

This run is deliberately fresh rather than resumed from the one-epoch control: the
one-epoch scheduler has already decayed its learning rate to zero, whereas this
experiment must define a reproducible four-epoch schedule from the start.

### Token-weighted short-reasoning SFT

The unweighted short-1K four-epoch curve peaks at checkpoint 189 (epoch 3) with
6/10 normal stops, extraction 0.7, compile 0.6, and pass@1 0.1, then regresses at
epoch 4 while teacher-forcing loss continues to fall. Keep the data and schedule
fixed and change only the response-token objective:

- reasoning tokens: 0.25;
- C++ code tokens: 1.0;
- `<think>`/`</think>` and code-fence boundary tokens: 2.0;
- EOS: 4.0;
- prompt and padding: 0.0.

The weighted cross entropy uses tokenizer offset mappings rather than approximate
string/token splitting. Under DDP, each micro-batch denominator is all-reduced and
the local numerator is scaled so DDP's gradient averaging yields the global
weighted-token mean.

First run the two-step technical smoke on the intended two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh weighted-smoke \
  2>&1 | tee /tmp/qwen3-m7-weighted-smoke.log
```

Only after it finishes without NaN, DDP errors, or missing-weight errors, launch a
fresh full run from Base:

```bash
CUDA_VISIBLE_DEVICES=0,1 SFT_GPU_COUNT=2 \
  bash scripts/cloud_train_sft.sh weighted-short-1k-4epoch \
  2>&1 | tee /tmp/qwen3-m7-weighted-short-1k-4epoch-train.log
```

Expected checkpoints remain 63/126/189/252. Evaluate each with
`cloud_eval_sft_ab.sh weighted` and the same frozen smoke protocol. Compare the whole
curve to the unweighted run; do not compare weighted and unweighted scalar loss
values directly because their token denominators differ.

```bash
for STEP in 63 126 189 252; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/cloud_eval_sft_ab.sh \
    weighted outputs/training/<weighted-run>/checkpoint-${STEP}
done 2>&1 | tee /tmp/qwen3-m7-weighted-short-1k-4epoch-eval.log
```
