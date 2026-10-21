#!/usr/bin/env python3
"""ScamCourt step 6 (data): judge SFT examples from the model's OWN debates on training videos.

  --debates   debates.jsonl from run_debate.py --split train
  --no-transcript-fraction f : additionally, for a stratified fraction f of the training videos,
              a judge example WITHOUT a transcript (the same judge then also works without a debate,
              and the "no argument" baseline of the adversarial test is in-distribution)
  --classifier-only : instead, one no-transcript example for EVERY training video
              (the plain-SFT baseline: same backbone, same output format, no debate)

Target = gold label, subtype, content role, rationale (codes replaced by names) and gold evidence
citations visible in the input. Transcripts mark each citation [verified] / [UNVERIFIED].
Train together with the teacher-debate judge data of the cold-start export:
  python scamcourt/train_sft.py --data work/sft_coldstart/judge.jsonl <this output> ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_DATASET, DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, ROOT, Dataset, load_exclusions,  # noqa: E402
                      load_split, read_jsonl, scc, select_ids, stratified_subsample, write_json, write_jsonl)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--debates", type=Path, nargs="*", default=[])
    p.add_argument("--no-transcript-fraction", type=float, default=0.25)
    p.add_argument("--classifier-only", action="store_true")
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--max-text-chars", type=int, default=32000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    return p


def judge_example(ds: Dataset, sid: str, args, transcript_fn, meta: dict) -> dict:
    sample, ctx = ds.context(sid, args.max_frames, args.max_text_chars)
    gold_cits = scc.reference_opening(sample, ds.taxonomy, ctx["source_texts"], ctx["frame_ids"])["citations"]
    transcript = transcript_fn(ctx) if transcript_fn else None
    target = scc.judge_target(sample, gold_cits, ctx, ds.names)
    return scc.role_example(scc.judge_system_prompt(ds.taxonomy), ctx, scc.judge_extra(transcript), target,
                            {"sample_id": sid, "gold_label": target["label"], "gold_subtype": target["subtype"],
                             "role": "judge", **meta})


def main() -> int:
    args = parser().parse_args()
    root = args.project_root.expanduser().resolve()
    ds = Dataset(root, args.dataset_dir)
    split = load_split(root, args.split_file)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    train_ids = select_ids(ds.rows, split, "train", excluded)
    allowed = set(train_ids)
    counts, rows = Counter(), []

    if args.classifier_only:
        for sid in train_ids:
            rows.append(judge_example(ds, sid, args, None, {"source": "no_transcript"}))
        counts["no_transcript"] = len(rows)
    else:
        if not args.debates:
            raise SystemExit("ERROR: pass --debates (or --classifier-only)")
        for path in args.debates:
            for rec in read_jsonl(root / path):
                sid = rec["sample_id"]
                if sid not in allowed:
                    counts["dropped_not_train_or_excluded"] += 1
                    continue

                def transcript(ctx, rec=rec):
                    return scc.render_transcript(rec["openings"], rec["rebuttals"], ds.names,
                                                 scc.citation_checker(ctx))
                failed = any(v["opening"] or v["rebuttal"] for v in rec["format_failed"].values())
                rows.append(judge_example(ds, sid, args, transcript,
                                          {"source": "model_debate", "k": rec["k"], "debate_format_failed": failed}))
                counts["model_debate"] += 1
        n_plain = round(len(train_ids) * args.no_transcript_fraction)
        plain = stratified_subsample(train_ids, {i: ds.rows[i]["primary_subtype"] for i in train_ids},
                                     n_plain, args.seed)
        for sid in plain:
            rows.append(judge_example(ds, sid, args, None, {"source": "no_transcript"}))
        counts["no_transcript"] = len(plain)
    out = (root / args.output).resolve()
    write_jsonl(out, rows)
    labels = Counter(r["meta"]["gold_label"] for r in rows)
    summary = {"output": str(out), "examples": len(rows), "counts": dict(counts), "labels": dict(labels),
               "settings": {"max_frames": args.max_frames, "max_text_chars": args.max_text_chars,
                            "classifier_only": args.classifier_only,
                            "no_transcript_fraction": args.no_transcript_fraction}}
    write_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
