#!/usr/bin/env python3
"""ScamCourt cold start, step 3: export SFT data (LLaMA-Factory sharegpt/messages format).

Roles (choose with --roles):
  false_opening   wrong-side debater, opening stage        (teacher argument, best-of-N selected)
  false_rebuttal  wrong-side debater, rebuttal stage       (answers the reference opening)
  true_opening    reference-side debater, opening stage    (built from annotation evidence + rationale)
  true_rebuttal   reference-side debater, rebuttal stage   (teacher's true_side_reply)
  judge           judge verdict on the full four-turn transcript (gold label/subtype/role/rationale)

Debaters use the SAME system prompt whether they are on the true or wrong side,
so the model cannot tell from the prompt which side is correct. Inputs are
label-agnostic (uniform frame set + budgeted ID-segmented text). An example is
dropped if any citation in its target is not visible in its input.
Use --split-file to restrict to training videos; otherwise every video is exported
with its sample_id in `meta` so you can filter later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scamcourt_common import (  # noqa: E402
    SIDE_LABEL, citations_visible, debate_context, debater_extra, debater_system_prompt, judge_extra,
    judge_system_prompt, judge_target, load_manifest, load_sample, load_true_openings, read_json, read_jsonl,
    render_transcript, role_example, taxonomy_maps, true_opening_for, user_content,  # noqa: F401
)

ROLES = ("false_opening", "false_rebuttal", "true_opening", "true_rebuttal", "judge")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/mechscam"))
    p.add_argument("--arguments-dir", type=Path, default=Path("data/arguments/teacher_arguments"))
    p.add_argument("--scores", type=Path,
                   default=Path("data/arguments/qwen3_vl_8b_scores/scores.jsonl"))
    p.add_argument("--no-scores", action="store_true",
                   help="Skip best-of-N; use all valid arguments (or the first with --first-only)")
    p.add_argument("--all-arguments", action="store_true",
                   help="Export every valid argument as augmentation instead of only the selected one")
    p.add_argument("--first-only", action="store_true")
    p.add_argument("--tie-epsilon", type=float, default=1e-3,
                   help="Arguments within this p(wrong) of the best count as tied; ties are broken by a hash "
                        "of the sample ID (not always the first strategy)")
    p.add_argument("--true-openings-dir", type=Path, default=Path("data/arguments/true_openings"))
    p.add_argument("--allow-template-true-opening", action="store_true",
                   help="Fall back to the annotation-template true opening (leaks the correct side by style)")
    p.add_argument("--exclude-weak", action="store_true",
                   help="Drop arguments the teacher rated argument_strength=weak (weak-case reruns)")
    p.add_argument("--split-file", type=Path,
                   help='JSON {"train": [sample_id, ...], ...}; only --split ids are exported')
    p.add_argument("--split", default="train")
    p.add_argument("--roles", default="false_opening,false_rebuttal")
    p.add_argument("--output-dir", type=Path, default=Path("work/sft_coldstart"))
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--max-text-chars", type=int, default=32000)
    p.add_argument("--image-path-mode", choices=("absolute", "relative"), default="absolute",
                   help="relative = relative to --project-root")
    return p


def pick_argument(score: dict, eps: float) -> tuple[int, str]:
    """Best-of-N index. When several arguments are (near-)tied for the highest p(wrong) - typically all
    ~0 because none moves the judge - pick one of them by a hash of the sample ID instead of always the
    first, so the wrong-side strategies stay balanced in the SFT data."""
    ps = [a["p_wrong"] for a in score["arguments"]]
    best = max(ps)
    tied = [i for i, p in enumerate(ps) if p >= best - eps]
    if len(tied) == 1:
        return tied[0], "best"
    h = int(hashlib.sha256(score["sample_id"].encode("utf-8")).hexdigest(), 16)
    return tied[h % len(tied)], "tie_hash"


def main() -> int:
    args = parser().parse_args()
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    if bad := set(roles) - set(ROLES):
        print(f"ERROR: unknown roles {sorted(bad)}; choose from {ROLES}", file=sys.stderr)
        return 2
    root = args.project_root.expanduser().resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    arguments_dir = (root / args.arguments_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    taxonomy = read_json(dataset_dir / "taxonomy.json")
    _, names = taxonomy_maps(taxonomy)
    rows = {r["sample_id"]: r for r in load_manifest(dataset_dir)}
    allowed = None
    if args.split_file:
        split = read_json((root / args.split_file).resolve())
        if args.split not in split:
            print(f"ERROR: split {args.split!r} not in {args.split_file}", file=sys.stderr)
            return 2
        allowed = set(split[args.split])
    true_openings = load_true_openings((root / args.true_openings_dir).resolve())
    selected = {}
    if not args.no_scores:
        scores_path = (root / args.scores).resolve()
        if not scores_path.is_file():
            print(f"ERROR: {scores_path} not found; run step 2 or pass --no-scores", file=sys.stderr)
            return 2
        for rec in read_jsonl(scores_path):
            selected[rec["sample_id"]] = rec
    out = {role: [] for role in roles}
    counts = Counter()
    for path in sorted(arguments_dir.glob("*/*.json")):
        record = read_json(path)
        if record.get("review_status") not in ("generated", "reviewed") or not record.get("arguments"):
            continue
        sid = record["sample_id"]
        counts["videos_with_arguments"] += 1
        if allowed is not None and sid not in allowed:
            counts["outside_split"] += 1
            continue
        if sid not in rows:
            raise ValueError(f"Unknown sample: {sid}")
        found = true_opening_for(record, true_openings, args.allow_template_true_opening)
        if found is None:
            counts["skipped_without_true_opening"] += 1
            continue
        reference, true_provenance = found
        if args.no_scores:
            indices = [0] if args.first_only else list(range(len(record["arguments"])))
        else:
            score = selected.get(sid)
            if score is None or score.get("input_fingerprint") != record["input_fingerprint"] or \
                    score.get("true_opening_provenance", "template") != true_provenance:
                counts["missing_or_stale_score"] += 1
                continue
            if args.all_arguments:
                indices = list(range(len(record["arguments"])))
            else:
                index, how = pick_argument(score, args.tie_epsilon)
                indices = [index]
                counts[f"selected_{how}"] += 1
        sample, sample_path = load_sample(dataset_dir, rows[sid])
        ctx = debate_context(sample, sample_path.parent, args.max_frames, args.max_text_chars,
                             None if args.image_path_mode == "absolute" else root)
        wrong, true = record["wrong_side"], record["true_side"]
        ref_target = {"claim_label": SIDE_LABEL[true], "claimed_subtype": reference["claimed_subtype"],
                      "argument": reference["argument"], "citations": reference["citations"]}
        meta = {"sample_id": sid, "gold_label": record["gold_label"], "gold_subtype": record["gold_subtype"],
                "true_opening": true_provenance.split(":")[0]}

        def emit(role: str, system: str, extra: str, target: dict, citations: list, **more) -> None:
            if not citations_visible(citations, ctx["source_ids"], ctx["frame_ids"]):
                counts[f"{role}_dropped_citation_not_in_input"] += 1
                return
            out[role].append(role_example(system, ctx, extra, target, {**meta, "role": role, **more}))
            counts[role] += 1

        if "true_opening" in roles:
            emit("true_opening", debater_system_prompt(true, taxonomy), debater_extra("opening"), ref_target,
                 ref_target["citations"], side=true)
        for idx in indices:
            arg = record["arguments"][idx]
            if args.exclude_weak and arg.get("argument_strength") == "weak":
                counts["excluded_weak_argument"] += 1
                continue
            wrong_opening = {"claim_label": SIDE_LABEL[wrong], "claimed_subtype": arg["claimed_subtype"],
                             "argument": arg["opening"]["argument"], "citations": arg["opening"]["citations"]}
            more = {"side": wrong, "strategy": arg["strategy"], "argument_index": idx,
                    "argument_strength": arg.get("argument_strength"),
                    "weak_case_mode": bool(record.get("weak_case_mode"))}
            if "false_opening" in roles:
                emit("false_opening", debater_system_prompt(wrong, taxonomy), debater_extra("opening"),
                     wrong_opening, wrong_opening["citations"], **more)
            if "false_rebuttal" in roles:
                emit("false_rebuttal", debater_system_prompt(wrong, taxonomy),
                     debater_extra("rebuttal", wrong, wrong_opening, ref_target, names), arg["rebuttal"],
                     arg["rebuttal"]["citations"], **more)
            if "true_rebuttal" in roles:
                emit("true_rebuttal", debater_system_prompt(true, taxonomy),
                     debater_extra("rebuttal", true, ref_target, wrong_opening, names), arg["true_side_reply"],
                     arg["true_side_reply"]["citations"], **{**more, "side": true})
            if "judge" in roles:
                transcript = render_transcript({true: ref_target, wrong: wrong_opening},
                                               {wrong: arg["rebuttal"], true: arg["true_side_reply"]}, names)
                verdict = judge_target(sample, reference["citations"], ctx, names)
                emit("judge", judge_system_prompt(taxonomy), judge_extra(transcript), verdict, verdict["citations"],
                     **more)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = {}
    for role, examples in out.items():
        path = output_dir / f"{role}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for ex in examples:
                handle.write(json.dumps(ex, ensure_ascii=False) + "\n")
        dataset_info[f"scamcourt_{role}"] = {
            "file_name": str(path), "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                     "assistant_tag": "assistant", "system_tag": "system"}}
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
    summary = {"roles": roles, "counts": dict(counts), "output_dir": str(output_dir),
               "split_filter": args.split if args.split_file else None}
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
