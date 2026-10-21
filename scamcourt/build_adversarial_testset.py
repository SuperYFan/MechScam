#!/usr/bin/env python3
"""ScamCourt adversarial-argument test set (test-split videos only; never used for training).

One row per teacher-written wrong-side argument of a test video:
  wrong-side opening + rebuttal, the true side's reply, the rewritten true-side opening,
  strategy, argument strength (weak-case reruns), and - if step-2 scores exist - the zero-shot
  Qwen judge's p(wrong) and whether best-of-N selected it.
Evaluate with: python scamcourt/evaluate.py --mode adversarial --adv-file <output> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_DATASET, DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, ROOT, Dataset, load_exclusions,  # noqa: E402
                      load_split, read_jsonl, scc, write_json, write_jsonl)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--split", default="test")
    p.add_argument("--arguments-dir", type=Path, default=Path("data/arguments/teacher_arguments"))
    p.add_argument("--true-openings-dir", type=Path, default=Path("data/arguments/true_openings"))
    p.add_argument("--scores", type=Path,
                   default=Path("data/arguments/qwen3_vl_8b_scores/scores.jsonl"))
    p.add_argument("--output", type=Path, default=Path("data/adversarial_test.jsonl"))
    return p


def main() -> int:
    args = parser().parse_args()
    root = args.project_root.expanduser().resolve()
    ds = Dataset(root, args.dataset_dir)
    split = load_split(root, args.split_file)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    wanted = split[args.split] - excluded
    true_openings = scc.load_true_openings((root / args.true_openings_dir).resolve())
    scores = {}
    if (root / args.scores).exists():
        scores = {r["sample_id"]: r for r in read_jsonl(root / args.scores)}
    rows, counts = [], Counter()
    for path in sorted((root / args.arguments_dir).glob("*/*.json")):
        if path.parent.name == "logs":
            continue
        rec = scc.read_json(path)
        sid = rec.get("sample_id")
        if sid not in wanted:
            continue
        if rec.get("review_status") not in ("generated", "reviewed") or not rec.get("arguments"):
            counts["videos_without_arguments"] += 1
            continue
        found = scc.true_opening_for(rec, true_openings, allow_template=False)
        true_opening = None
        if found:
            ref = found[0]
            true_opening = {"claim_label": scc.SIDE_LABEL[rec["true_side"]], "claimed_subtype": ref["claimed_subtype"],
                            "argument": ref["argument"], "citations": ref["citations"]}
        else:
            counts["videos_without_true_opening"] += 1
        score = scores.get(sid)
        if score and score.get("input_fingerprint") != rec["input_fingerprint"]:
            score = None
            counts["stale_scores_ignored"] += 1
        counts["videos"] += 1
        for i, arg in enumerate(rec["arguments"]):
            s = score["arguments"][i] if score and i < len(score["arguments"]) else None
            rows.append({
                "item_id": f"{sid}#{i}", "sample_id": sid, "argument_index": i,
                "gold_label": rec["gold_label"], "gold_subtype": rec["gold_subtype"],
                "true_side": rec["true_side"], "wrong_side": rec["wrong_side"], "wrong_label": rec["wrong_label"],
                "strategy": arg["strategy"], "argument_strength": arg.get("argument_strength") or "normal",
                "weak_case_mode": bool(rec.get("weak_case_mode")),
                "wrong_opening": {"claim_label": rec["wrong_label"], "claimed_subtype": arg["claimed_subtype"],
                                  "argument": arg["opening"]["argument"], "citations": arg["opening"]["citations"]},
                "wrong_rebuttal": arg["rebuttal"], "true_reply": arg["true_side_reply"],
                "true_opening": true_opening,
                "qwen_zero_shot_p_wrong": s["p_wrong"] if s else None,
                "qwen_zero_shot_baseline_p_wrong": score["baseline"]["p_wrong"] if score else None,
                "qwen_selected": bool(score and score["selected_index"] == i)})
    out = (root / args.output).resolve()
    write_jsonl(out, rows)
    summary = {"output": str(out), "split": args.split, "arguments": len(rows), "counts": dict(counts),
               "by_gold_label": dict(Counter(r["gold_label"] for r in rows)),
               "by_strategy": dict(Counter(r["strategy"] for r in rows)),
               "by_strength": dict(Counter(r["argument_strength"] for r in rows)),
               "with_true_opening": sum(r["true_opening"] is not None for r in rows),
               "with_qwen_scores": sum(r["qwen_zero_shot_p_wrong"] is not None for r in rows)}
    write_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
