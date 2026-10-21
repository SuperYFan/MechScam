#!/usr/bin/env python3
"""ScamCourt LoRA SFT on role examples (sharegpt messages + images).

Used for both models:
  debater : --data .../false_opening.jsonl .../false_rebuttal.jsonl .../true_opening.jsonl .../true_rebuttal.jsonl
  judge   : --data .../judge.jsonl (teacher debates) scamcourt/data/judge_rollouts.jsonl (own debates)

Only videos of the training split are used (meta.sample_id is checked against --split-file;
anything else, and anything in the exclusion list, is dropped and counted). Loss is on the
assistant answer only. One rolling checkpoint (adapter_latest) is kept; the final adapter is
written to <output-dir>/adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, ROOT, check_config, load_exclusions,  # noqa: E402
                      load_split, read_jsonl)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--data", type=Path, nargs="+", required=True, help="Example jsonl files")
    p.add_argument("--max-per-file", type=int, help="Cap examples taken from each file (random, seeded)")
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-image-pixels", type=int, default=384 * 384)
    p.add_argument("--max-seq-len", type=int, default=16384, help="Skip longer examples (tokens)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10, help="Optimizer steps")
    p.add_argument("--save-every", type=int, default=500, help="Optimizer steps (rolling adapter_latest)")
    p.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps (smoke tests)")
    p.add_argument("--resume-adapter", type=Path)
    p.add_argument("--resume-step", type=int, default=0)
    return p


def load_examples(args, root: Path) -> tuple[list[dict], dict]:
    split = load_split(root, args.split_file)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    rng = random.Random(args.seed)
    examples, counts = [], Counter()
    for path in args.data:
        rows = list(read_jsonl(root / path))
        if args.max_per_file and len(rows) > args.max_per_file:
            rows = rng.sample(rows, args.max_per_file)
        for ex in rows:
            sid = ex.get("meta", {}).get("sample_id")
            if sid in excluded:
                counts["dropped_excluded"] += 1
            elif sid not in split["train"]:
                counts["dropped_not_in_train_split"] += 1
            else:
                examples.append(ex)
                counts[f"used:{Path(path).name}"] += 1
    if counts["dropped_not_in_train_split"]:
        print(f"WARNING: dropped {counts['dropped_not_in_train_split']} examples whose video is not in the "
              f"training split", file=sys.stderr)
    return examples, dict(counts)


def save_rolling(model, out_dir: Path, step: int) -> None:
    tmp, final = out_dir / "adapter_latest.tmp", out_dir / "adapter_latest"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp))
    (tmp / "step.txt").write_text(str(step), encoding="utf-8")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def main() -> int:
    args = parser().parse_args()
    root = args.project_root.expanduser().resolve()
    out_dir = (root / args.output_dir).resolve()
    examples, counts = load_examples(args, root)
    if not examples:
        raise SystemExit("ERROR: no training examples left after the split/exclusion filter")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("resume_adapter", "resume_step", "log_every", "save_every")}
    config["data"] = [str(p) for p in args.data]
    config["example_counts"] = counts
    out_dir.mkdir(parents=True, exist_ok=True)
    check_config(out_dir / "train_config.json", config)
    print(json.dumps({"examples": len(examples), "counts": counts}, indent=1), flush=True)

    import torch
    from transformers import get_cosine_schedule_with_warmup
    from sc_model import causal_lm_loss, encode_train, load_model, model_device, to_messages

    torch.manual_seed(args.seed)
    lora = {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    processor, model = load_model(args.model_path.expanduser().resolve(),
                                  adapter=args.resume_adapter.expanduser().resolve() if args.resume_adapter else None,
                                  trainable=True, lora=None if args.resume_adapter else lora,
                                  gradient_checkpointing=True)
    model.train()
    device = model_device(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_micro = int(math.ceil(len(examples) * args.epochs))
    total_steps = max(1, total_micro // args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    for _ in range(args.resume_step):
        scheduler.step()

    order: list[int] = []
    epoch = 0
    while len(order) < total_micro:
        idx = list(range(len(examples)))
        random.Random(args.seed + epoch).shuffle(idx)
        order += idx
        epoch += 1
    order = order[:total_steps * args.grad_accum]

    log_path = out_dir / "train_log.jsonl"
    step, micro, start = args.resume_step, args.resume_step * args.grad_accum, time.time()
    window = {"loss": 0.0, "n": 0, "skipped_long": 0, "oom": 0, "tokens": 0}
    optimizer.zero_grad(set_to_none=True)
    while micro < len(order):
        ex = examples[order[micro]]
        micro += 1
        try:
            messages, target = to_messages(ex, root, args.max_image_pixels)
            inputs, n_tokens = encode_train(processor, messages, target, device)
            if n_tokens > args.max_seq_len:
                window["skipped_long"] += 1
                del inputs
            else:
                loss = causal_lm_loss(model, inputs)
                (loss / args.grad_accum).backward()
                window["loss"] += float(loss.detach())
                window["n"] += 1
                window["tokens"] += n_tokens
                del inputs, loss
        except torch.cuda.OutOfMemoryError:
            window["oom"] += 1
            inputs = loss = None  # noqa: F841 - release the graph before emptying the cache
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                row = {"step": step, "total_steps": total_steps,
                       "loss": round(window["loss"] / max(1, window["n"]), 4), "examples": window["n"],
                       "skipped_long": window["skipped_long"], "oom": window["oom"],
                       "mean_tokens": round(window["tokens"] / max(1, window["n"])),
                       "lr": scheduler.get_last_lr()[0], "elapsed_min": round((time.time() - start) / 60, 1),
                       "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)
                       if torch.cuda.is_available() else 0}
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                window = {"loss": 0.0, "n": 0, "skipped_long": 0, "oom": 0, "tokens": 0}
            if step % args.save_every == 0 and step < total_steps:
                save_rolling(model, out_dir, step)
            if step >= total_steps:
                break
    model.save_pretrained(str(out_dir / "adapter"))
    shutil.rmtree(out_dir / "adapter_latest", ignore_errors=True)
    print(f"saved {out_dir / 'adapter'} after {step} steps", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
