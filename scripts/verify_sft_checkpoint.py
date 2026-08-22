from __future__ import annotations

import argparse
import json

from src.inference.prompts import build_code_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description="Reload an SFT checkpoint and run one loss/generation probe")
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="data/processed/sft_1k.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    with open(args.data, encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    prompt = build_code_prompt(row["problem"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "problem_id": row["problem_id"],
        "generated_tokens": int(output.shape[0]),
        "generated_text": text,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
