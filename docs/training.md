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

The full evaluation protocol is identical to M4: the same frozen 399 problems,
greedy pass@1, 32,768-token context, and 16,384-token generation cap.
