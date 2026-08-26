from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.training.sft import SFTDataCollator, SFTDataset, weighted_causal_lm_loss
from src.utils.config import load_config, require_sections
from src.utils.experiment import collect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen SFT with response-only loss")
    parser.add_argument("--config", default="configs/training/m6_sft_smoke.yaml")
    parser.add_argument("--resume", help="Trainer checkpoint directory, e.g. checkpoint-25")
    return parser.parse_args()


def _run_directory(config: dict[str, Any], resume: str | None) -> Path:
    if resume:
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_dir() or not checkpoint.name.startswith("checkpoint-"):
            raise ValueError(f"Invalid Trainer checkpoint: {checkpoint}")
        return checkpoint.parent
    experiment = config["experiment"]
    timestamp = os.environ.get("SFT_RUN_TIMESTAMP")
    if not timestamp:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError(
                "SFT_RUN_TIMESTAMP must be set for distributed training so every rank "
                "uses the same output directory"
            )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(experiment["output_dir"]) / f"{experiment['name']}-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_sections(config, "experiment", "model", "data", "training")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    run_dir = _run_directory(config, args.resume)
    model_config = config["model"]
    training = config["training"]
    data_config = config["data"]

    micro_batch = int(training["per_device_train_batch_size"])
    global_batch = int(training["global_batch_size"])
    denominator = world_size * micro_batch
    if global_batch % denominator:
        raise ValueError(
            f"global_batch_size={global_batch} is not divisible by world_size({world_size}) "
            f"* per_device_train_batch_size({micro_batch})"
        )
    gradient_accumulation = global_batch // denominator

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"], revision=model_config["revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SFTDataset(
        data_config["path"],
        tokenizer,
        max_length=int(data_config["max_length"]),
        limit=data_config.get("limit"),
        selection=data_config.get("selection", "ordered"),
        loss_weights=training.get("loss_weights"),
    )
    dataset_sha256 = hashlib.sha256(Path(data_config["path"]).read_bytes()).hexdigest()
    eval_dataset = None
    eval_dataset_sha256 = None
    if data_config.get("eval_path"):
        eval_dataset = SFTDataset(
            data_config["eval_path"],
            tokenizer,
            max_length=int(data_config["max_length"]),
            limit=data_config.get("eval_limit"),
            selection=data_config.get("eval_selection", "ordered"),
            loss_weights=training.get("loss_weights"),
        )
        eval_dataset_sha256 = hashlib.sha256(
            Path(data_config["eval_path"]).read_bytes()
        ).hexdigest()

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config["revision"],
        dtype=getattr(torch, model_config.get("dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    model.config.use_cache = False

    max_steps = int(training.get("max_steps", -1))
    num_train_epochs = float(training.get("num_train_epochs", 3.0))
    if max_steps <= 0 and num_train_epochs <= 0:
        raise ValueError("Set a positive max_steps or num_train_epochs")

    trainer_args = TrainingArguments(
        output_dir=str(run_dir),
        do_train=True,
        do_eval=eval_dataset is not None,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=gradient_accumulation,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training.get("lr_scheduler_type", "linear"),
        warmup_steps=int(training.get("warmup_steps", 0)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        bf16=bool(training.get("bf16", True)),
        tf32=bool(training.get("tf32", True)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        logging_strategy="steps",
        logging_steps=int(training.get("logging_steps", 1)),
        logging_first_step=True,
        eval_strategy=training.get("eval_strategy", "epoch") if eval_dataset else "no",
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 25)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        report_to="none",
        skip_memory_metrics=False,
        remove_unused_columns=False,
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        ddp_find_unused_parameters=False,
        include_num_input_tokens_seen="all",
        seed=int(config["experiment"]["seed"]),
        data_seed=int(config["experiment"]["seed"]),
    )
    class WeightedSFTTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # The custom objective normalizes by weighted tokens, not by
            # Trainer's num_items_in_batch. This flag makes Trainer apply its
            # normal gradient-accumulation scaling and report a per-microbatch
            # mean instead of summing all microbatch losses.
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_causal_lm_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    trainer_class = WeightedSFTTrainer if training.get("loss_weights") else Trainer
    trainer = trainer_class(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(int(tokenizer.pad_token_id)),
        processing_class=tokenizer,
    )

    if rank == 0 and not args.resume:
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        metadata = collect_environment()
        metadata["training"] = {
            "world_size": world_size,
            "global_batch_size": global_batch,
            "gradient_accumulation_steps": gradient_accumulation,
            "dataset_sha256": dataset_sha256,
            "selected_samples": len(dataset),
            "selected_tokens": sum(example["length"] for example in dataset.examples),
            "selected_problem_ids": [example["problem_id"] for example in dataset.examples],
            "loss_weights": training.get("loss_weights"),
            "selected_loss_weight_sum": sum(
                sum(example.get("loss_weights", [])) for example in dataset.examples
            ),
            "eval_dataset_sha256": eval_dataset_sha256,
            "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_tokens": (
                sum(example["length"] for example in eval_dataset.examples)
                if eval_dataset is not None
                else 0
            ),
        }
        (run_dir / "environment.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    result = trainer.train(resume_from_checkpoint=args.resume or None)
    final_eval_metrics = (
        trainer.evaluate(metric_key_prefix="final_eval") if eval_dataset is not None else {}
    )
    trainer.save_model(str(run_dir / "final"))
    if rank == 0:
        tokenizer.save_pretrained(run_dir / "final")
        metrics = dict(result.metrics)
        metrics.update(final_eval_metrics)
        metrics["world_size"] = world_size
        metrics["global_batch_size"] = global_batch
        metrics["gradient_accumulation_steps"] = gradient_accumulation
        (run_dir / "train_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print(f"Artifacts: {run_dir}")
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
