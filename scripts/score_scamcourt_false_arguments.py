#!/usr/bin/env python3
"""ScamCourt cold start, step 2: best-of-N selection with a local zero-shot judge.

For each generated wrong-side argument, a local Qwen3-VL-8B-Instruct judge sees
the video inputs (sampled frames + ID-segmented text) and a four-turn debate
transcript (both openings and both rebuttals). The judge answers with a letter;
we read the next-token probabilities of the two option letters (order
counterbalanced per argument) and record P(wrong label). The argument that
pushes the judge furthest toward the wrong label is selected (Khan et al.,
2024, best-of-N by persuasiveness). A no-debate pass gives the baseline.

No weights are downloaded. Results are appended to scores.jsonl and resumable;
--dry-run checks inputs without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scamcourt_common import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT, SIDES, all_sources, debate_frames, load_manifest, load_sample, load_true_openings,
    read_json, read_jsonl, render_transcript, select_sources, taxonomy_maps, true_opening_for, video_block_text,
    within,
)

VERSION = "scamcourt_judge_best_of_n_v1"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/mechscam"))
    p.add_argument("--arguments-dir", type=Path, default=Path("data/arguments/teacher_arguments"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/arguments/qwen3_vl_8b_scores"))
    p.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-image-pixels", type=int, default=384 * 384)
    p.add_argument("--max-text-chars", type=int, default=16000)
    p.add_argument("--true-openings-dir", type=Path, default=Path("data/arguments/true_openings"))
    p.add_argument("--allow-template-true-opening", action="store_true",
                   help="Use the annotation-template true opening when no rewritten one exists (style leak!)")
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    return p


def option_order(key: str) -> tuple[str, str]:
    """Letters A/B -> labels, counterbalanced without using gold labels."""
    return ("Scam", "Non-scam") if hashlib.sha256(key.encode()).digest()[0] % 2 == 0 else ("Non-scam", "Scam")


def judge_messages(sample: dict, sample_dir: Path, args: argparse.Namespace, transcript: str | None,
                   order: tuple[str, str]) -> list[dict]:
    frames = debate_frames(sample, args.max_frames)
    sources = select_sources(all_sources(sample), args.max_text_chars)
    content: list[dict] = [{"type": "text", "text": video_block_text(sources, frames)}]
    for frame in frames:
        path = within(sample_dir, frame["file"])
        if not path.is_file():
            raise FileNotFoundError(path)
        content.append({"type": "text", "text": f"Frame {frame['frame_id']}:"})
        content.append({"type": "image", "image": path.as_uri(), "max_pixels": args.max_image_pixels})
    question = ("\n\nDEBATE TRANSCRIPT (advocacy, not evidence):\n" + transcript) if transcript else ""
    question += (f"\n\nWhich label is correct for this video?\nA: {order[0]}\nB: {order[1]}\n"
                 "Answer with only the letter A or B.")
    content.append({"type": "text", "text": question})
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": content}]


def transcripts_for(record: dict, names: dict, true_opening: dict | None = None) -> list[str]:
    wrong, true = record["wrong_side"], record["true_side"]
    result = []
    for arg in record["arguments"]:
        wrong_opening = {"claim_label": record["wrong_label"], "claimed_subtype": arg["claimed_subtype"],
                         **arg["opening"]}
        openings = {true: true_opening or record["reference_opening"], wrong: wrong_opening}
        rebuttals = {wrong: arg["rebuttal"], true: arg["true_side_reply"]}
        assert set(openings) == set(SIDES)
        result.append(render_transcript(openings, rebuttals, names))
    return result


class Judge:
    def __init__(self, model_path: Path):
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path), dtype="auto", device_map="auto", local_files_only=True).eval()
        tok = self.processor.tokenizer
        self.letter_ids = []
        for letter in ("A", "B"):
            ids = tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Option letter {letter} is not a single token: {ids}")
            self.letter_ids.append(ids[0])

    def p_options(self, messages: list[dict]) -> tuple[float, float, str]:
        import torch
        from qwen_vl_utils import process_vision_info
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)
        if videos is not None:
            raise ValueError("Only sampled images are supported")
        inputs = self.processor(text=[text], images=images, return_tensors="pt", do_resize=False).to(self.model.device)
        with torch.inference_mode():
            try:
                logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
            except TypeError:
                logits = self.model(**inputs).logits[0, -1].float()
        pair = torch.softmax(logits[self.letter_ids], dim=-1).tolist()
        top = self.processor.tokenizer.decode([int(logits.argmax())])
        del inputs, logits
        return pair[0], pair[1], top


def p_wrong(judge: Judge, messages: list[dict], order: tuple[str, str], wrong_label: str) -> dict:
    pa, pb, top = judge.p_options(messages)
    probs = {order[0]: pa, order[1]: pb}
    return {"p_wrong": round(probs[wrong_label], 6), "option_order": list(order), "top_token": top}


def summarize(records: list[dict]) -> dict:
    def block(items: list[dict]) -> dict:
        if not items:
            return {"n": 0}
        base = [r["baseline"]["p_wrong"] for r in items]
        best = [r["arguments"][r["selected_index"]]["p_wrong"] for r in items]
        return {"n": len(items),
                "baseline_judge_accuracy": round(sum(p < 0.5 for p in base) / len(items), 4),
                "selected_fool_rate": round(sum(p >= 0.5 for p in best) / len(items), 4),
                "mean_baseline_p_wrong": round(sum(base) / len(items), 4),
                "mean_selected_p_wrong": round(sum(best) / len(items), 4)}
    by_label, by_subtype, strat = defaultdict(list), defaultdict(list), defaultdict(list)
    strength = defaultdict(list)
    for r in records:
        by_label[r["gold_label"]].append(r)
        by_subtype[r["gold_subtype"]].append(r)
        for a in r["arguments"]:
            strat[a["strategy"]].append(a["p_wrong"])
            strength[a.get("argument_strength") or "unrated"].append(a["p_wrong"])
    return {"overall": block(records),
            "by_gold_label": {k: block(v) for k, v in sorted(by_label.items())},
            "by_gold_subtype": {k: block(v) for k, v in sorted(by_subtype.items())},
            "by_strategy_mean_p_wrong": {k: round(sum(v) / len(v), 4) for k, v in sorted(strat.items())},
            "by_strength_mean_p_wrong": {k: {"n": len(v), "mean": round(sum(v) / len(v), 4)}
                                         for k, v in sorted(strength.items())},
            "selected_strategy_counts": dict(Counter(r["arguments"][r["selected_index"]]["strategy"]
                                                     for r in records))}


def run(args: argparse.Namespace) -> int:
    root = args.project_root.expanduser().resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    arguments_dir = (root / args.arguments_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    taxonomy = read_json(dataset_dir / "taxonomy.json")
    _, names = taxonomy_maps(taxonomy)
    rows = {r["sample_id"]: r for r in load_manifest(dataset_dir)}
    true_openings = load_true_openings((root / args.true_openings_dir).resolve())
    jobs, missing_true = [], 0
    for path in sorted(arguments_dir.glob("*/*.json")):
        record = read_json(path)
        if record.get("review_status") in ("generated", "reviewed") and record.get("arguments"):
            if record["sample_id"] not in rows:
                raise ValueError(f"Unknown sample in arguments dir: {record['sample_id']}")
            found = true_opening_for(record, true_openings, args.allow_template_true_opening)
            if found is None:
                missing_true += 1
                continue
            record["_true_opening"], record["_true_provenance"] = found
            record["_score_key"] = record["input_fingerprint"] + "|" + found[1]
            jobs.append(record)
    jobs.sort(key=lambda r: r["sample_id"])
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps({"arguments_dir": str(arguments_dir), "skipped_without_true_opening": missing_true,
                      "scorable_videos": len(jobs),
                      "arguments": sum(len(r["arguments"]) for r in jobs)}, indent=1), flush=True)
    if not jobs:
        raise ValueError("No generated argument files found; run step 1 first")
    if args.dry_run:
        for record in jobs:
            sample, sample_path = load_sample(dataset_dir, rows[record["sample_id"]])
            for i, transcript in enumerate([None] + transcripts_for(record, names, record["_true_opening"])):
                judge_messages(sample, sample_path.parent, args, transcript, option_order(f"{record['sample_id']}|{i}"))
        print(f"Dry run passed for {len(jobs)} videos; no model loaded.", flush=True)
        return 0
    config = {"version": VERSION, "model_path": str(args.model_path.expanduser().resolve()),
              "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
              "max_frames": args.max_frames, "max_image_pixels": args.max_image_pixels,
              "max_text_chars": args.max_text_chars}
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, scores_path = output_dir / "run_config.json", output_dir / "scores.jsonl"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Output dir has a different run config; use another --output-dir")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    done = {}
    if scores_path.exists():
        for rec in read_jsonl(scores_path):
            done[rec["sample_id"]] = rec
    remaining = [r for r in jobs if done.get(r["sample_id"], {}).get("score_key") != r["_score_key"]]
    print(f"Resume: {len(jobs) - len(remaining)} done, {len(remaining)} to score", flush=True)
    if remaining:
        judge = Judge(args.model_path.expanduser().resolve())
        with scores_path.open("a", encoding="utf-8") as handle:
            for n, record in enumerate(remaining, 1):
                start = time.monotonic()
                sid = record["sample_id"]
                sample, sample_path = load_sample(dataset_dir, rows[sid])
                order = option_order(f"{sid}|0")
                baseline = p_wrong(judge, judge_messages(sample, sample_path.parent, args, None, order),
                                   order, record["wrong_label"])
                scored = []
                for i, (arg, transcript) in enumerate(zip(record["arguments"],
                                                          transcripts_for(record, names, record["_true_opening"])), 1):
                    order = option_order(f"{sid}|{i}")
                    result = p_wrong(judge, judge_messages(sample, sample_path.parent, args, transcript, order),
                                     order, record["wrong_label"])
                    scored.append({"index": i - 1, "strategy": arg["strategy"],
                                   "claimed_subtype": arg["claimed_subtype"],
                                   "argument_strength": arg.get("argument_strength"), **result})
                selected = max(range(len(scored)), key=lambda k: scored[k]["p_wrong"])
                out = {"sample_id": sid, "gold_label": record["gold_label"], "gold_subtype": record["gold_subtype"],
                       "wrong_label": record["wrong_label"], "input_fingerprint": record["input_fingerprint"],
                       "true_opening_provenance": record["_true_provenance"], "score_key": record["_score_key"],
                       "baseline": baseline, "arguments": scored, "selected_index": selected,
                       "selection_rule": "max_p_wrong_with_full_debate",
                       "elapsed_seconds": round(time.monotonic() - start, 3)}
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                handle.flush()
                done[sid] = out
                print(f"[{n}/{len(remaining)}] {sid} base={baseline['p_wrong']:.3f} "
                      f"best={scored[selected]['p_wrong']:.3f} ({scored[selected]['strategy']})", flush=True)
    current = [done[r["sample_id"]] for r in jobs if done.get(r["sample_id"], {}).get("score_key") == r["_score_key"]]
    summary = {"config": config, "summary": summarize(current)}
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["summary"]["overall"], indent=2), flush=True)
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
