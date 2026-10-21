#!/usr/bin/env python3
"""ScamCourt step 7: evaluate a judge (LoRA adapter, or --no-adapter for the zero-shot base model).

Modes
  no_debate   : the judge sees the video only (test split).
  debate      : the judge sees the video + a debate produced by run_debate.py on the test split.
  adversarial : the judge sees the video + teacher-written wrong-side arguments
                (build_adversarial_testset.py). Protocols:
                  single = only the wrong-side opening (one-sided persuasion)
                  full   = rewritten true opening + wrong opening + both rebuttals
                Each video is also judged once without any argument (baseline).

p(Scam) is read from the answer's first label token after '{"label": "' (two-way, no sampling);
in no_debate/debate mode the full answer is also generated to read subtype and content role.
Predictions are appended to predictions.jsonl (resumable); metrics go to metrics.json.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_DATASET, DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, LABELS, ROOT, Dataset,  # noqa: E402
                      adversarial_metrics, check_config, has_risk_cues, judge_metrics, load_exclusions, load_split,
                      mean, parse_verdict, read_jsonl, refresh_gold, scc, select_ids, write_json)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--split", default="test")
    p.add_argument("--mode", choices=("no_debate", "debate", "adversarial"), required=True)
    p.add_argument("--debates", type=Path, help="debates.jsonl (debate mode); sample k=0 is used")
    p.add_argument("--adv-file", type=Path, default=Path("data/adversarial_test.jsonl"))
    p.add_argument("--protocols", default="single,full")
    p.add_argument("--debate-view", choices=("full", "openings", "prosecution_only", "defense_only"),
                   default="full", help="debate mode: which turns of each debate the judge sees")
    p.add_argument("--no-verification-marks", action="store_true",
                   help="debate mode (ablation): show citations without [verified]/[UNVERIFIED] marks")
    p.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--adapter", type=Path)
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--no-generate", action="store_true", help="Probabilities only (no subtype / role)")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--max-text-chars", type=int, default=32000)
    p.add_argument("--max-image-pixels", type=int, default=384 * 384)
    p.add_argument("--limit", type=int)
    return p


def items_for(args, root: Path, ds: Dataset, excluded: set[str]) -> list[dict]:
    """Each item: id, sample_id, transcript spec (None / model debate / teacher protocol)."""
    split = load_split(root, args.split_file)
    if args.mode == "no_debate":
        return [{"id": sid, "sample_id": sid, "kind": "none"} for sid in select_ids(ds.rows, split, args.split, excluded)]
    if args.mode == "debate":
        if not args.debates:
            raise SystemExit("ERROR: --debates is required in debate mode")
        wanted = set(select_ids(ds.rows, split, args.split, excluded))
        recs = [r for r in read_jsonl(root / args.debates) if r["k"] == 0 and r["sample_id"] in wanted]
        missing = wanted - {r["sample_id"] for r in recs}
        if missing:
            print(f"WARNING: {len(missing)} {args.split} videos have no debate yet", file=sys.stderr)
        return [{"id": r["sample_id"], "sample_id": r["sample_id"], "kind": "model_debate", "debate": r,
                 "view": args.debate_view}
                for r in sorted(recs, key=lambda r: r["sample_id"])]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    rows = [r for r in read_jsonl(root / args.adv_file) if r["sample_id"] not in excluded]
    if any(r["sample_id"] in split["train"] for r in rows):
        raise SystemExit("ERROR: the adversarial file contains training videos")
    items = [{"id": f"{sid}#base", "sample_id": sid, "kind": "none"} for sid in sorted({r["sample_id"] for r in rows})]
    for r in rows:
        for proto in protocols:
            if proto == "full" and r["true_opening"] is None:
                continue
            items.append({"id": f"{r['item_id']}#{proto}", "sample_id": r["sample_id"], "kind": proto, "adv": r})
    return items


NO_MARKS = False  # set by --no-verification-marks


def transcript_of(item: dict, ctx: dict, names: dict) -> str | None:
    if item["kind"] == "none":
        return None
    if item["kind"] == "model_debate":
        d, view = item["debate"], item["view"]
        if view == "full":
            openings, rebuttals = d["openings"], d["rebuttals"]
        elif view == "openings":
            openings, rebuttals = d["openings"], {}
        else:  # one-sided persuasion ("consultancy"): only that side's opening
            side = view.split("_")[0]
            openings, rebuttals = {side: d["openings"][side]}, {}
        checker = None if NO_MARKS else scc.citation_checker(ctx)
        return scc.render_transcript(openings, rebuttals, names, checker)
    a = item["adv"]
    if item["kind"] == "single":
        return scc.render_transcript({a["wrong_side"]: a["wrong_opening"]}, {}, names)
    return scc.render_transcript({a["true_side"]: a["true_opening"], a["wrong_side"]: a["wrong_opening"]},
                                 {a["wrong_side"]: a["wrong_rebuttal"], a["true_side"]: a["true_reply"]}, names)


def main() -> int:
    args = parser().parse_args()
    global NO_MARKS
    if args.no_verification_marks:
        if args.mode != "debate":
            raise SystemExit("ERROR: --no-verification-marks applies to --mode debate")
        NO_MARKS = True
        scc.format_citations = functools.partial(scc.format_citations, verified=False)  # no marks at all
    if bool(args.adapter) == bool(args.no_adapter):
        raise SystemExit("ERROR: pass exactly one of --adapter or --no-adapter")
    root = args.project_root.expanduser().resolve()
    ds = Dataset(root, args.dataset_dir)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    items = items_for(args, root, ds, excluded)
    if args.limit:
        keep = sorted({it["sample_id"] for it in items})[:args.limit]
        items = [it for it in items if it["sample_id"] in set(keep)]
    generate_text = args.mode != "adversarial" and not args.no_generate
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"mode": args.mode, "split": args.split, "adapter": str(args.adapter) if args.adapter else None,
              "debates": str(args.debates) if args.debates else None,
              "debate_view": args.debate_view if args.mode == "debate" else None,
              "adv_file": str(args.adv_file) if args.mode == "adversarial" else None,
              "protocols": args.protocols if args.mode == "adversarial" else None, "generate": generate_text,
              "max_new_tokens": args.max_new_tokens, "max_frames": args.max_frames,
              "max_text_chars": args.max_text_chars, "max_image_pixels": args.max_image_pixels,
              "model_path": str(args.model_path)}
    if args.no_verification_marks:
        config["verification_marks"] = False
    check_config(out_dir / "eval_config.json", config)
    pred_path = out_dir / "predictions.jsonl"
    done = {r["id"]: r for r in read_jsonl(pred_path)} if pred_path.exists() else {}
    done, refreshed = refresh_gold(done, ds)  # saved rows follow the current dataset labels
    todo = [it for it in items if it["id"] not in done]
    print(json.dumps({"mode": args.mode, "items": len(items), "done": len(done), "todo": len(todo),
                      "gold_refresh": dict(refreshed)}), flush=True)

    if todo:
        import torch  # noqa: F401
        from sc_model import generate, load_model, option_continuation, option_probs, to_messages
        processor, model = load_model(args.model_path.expanduser().resolve(),
                                      adapter=args.adapter.expanduser().resolve() if args.adapter else None)
        model.eval()
        shared, ids = option_continuation(processor.tokenizer, scc.JUDGE_LABEL_PREFIX, LABELS)
        print(json.dumps({"answer_prefix_tokens": len(shared),
                          "label_tokens": [processor.tokenizer.decode([i]) for i in ids]}), flush=True)
        system = scc.judge_system_prompt(ds.taxonomy)
        cache: dict[str, tuple[dict, dict]] = {}
        start = time.time()
        with pred_path.open("a", encoding="utf-8") as fh:
            for b in range(0, len(todo), args.batch_size):
                batch = todo[b:b + args.batch_size]
                msgs, samples = [], []
                for it in batch:
                    if it["sample_id"] not in cache:
                        cache.clear()
                        cache[it["sample_id"]] = ds.context(it["sample_id"], args.max_frames, args.max_text_chars)
                    sample, ctx = cache[it["sample_id"]]
                    ex = scc.role_example(system, ctx, scc.judge_extra(transcript_of(it, ctx, ds.names)), None, {})
                    msgs.append(to_messages(ex, root, args.max_image_pixels)[0])
                    samples.append(sample)
                p_scam = option_probs(model, processor, msgs, scc.JUDGE_LABEL_PREFIX, LABELS)
                texts = generate(model, processor, msgs, args.max_new_tokens) if generate_text else [None] * len(batch)
                for it, sample, p, text in zip(batch, samples, p_scam, texts):
                    ann = sample["annotation"]
                    row = {"id": it["id"], "sample_id": it["sample_id"], "kind": it["kind"],
                           "label": ann["binary_label"], "subtype": ann["primary_subtype"],
                           "content_role": ann.get("content_role"), "has_risk_cues": has_risk_cues(sample),
                           "p_scam": p, "generated": text}
                    if text is not None:
                        v = parse_verdict(text, ds.labels)
                        row.update(pred_label_generated=v["label"], pred_subtype=v["subtype"],
                                   pred_role=v["content_role"], parsed=v["parsed"])
                    if it["kind"] in ("single", "full"):
                        a = it["adv"]
                        row.update(item_id=a["item_id"], protocol=it["kind"], strategy=a["strategy"],
                                   argument_strength=a["argument_strength"], weak_case_mode=a["weak_case_mode"],
                                   qwen_selected=a["qwen_selected"], gold_label=a["gold_label"],
                                   gold_subtype=a["gold_subtype"])
                    if it["kind"] == "model_debate":
                        d = it["debate"]
                        row.update(debate_format_failed=d["format_failed"], debate_citations=d["citations"])
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done[it["id"]] = row
                fh.flush()
                n = b + len(batch)
                if n % (args.batch_size * 10) < args.batch_size or n == len(todo):
                    rate = (time.time() - start) / n
                    print(f"[{n}/{len(todo)}] {rate:.2f}s/item, eta {rate * (len(todo) - n) / 3600:.1f}h", flush=True)

    preds = [done[it["id"]] for it in items if it["id"] in done]
    subtype_ids = list(ds.labels)
    if args.mode == "adversarial":
        base = {p["sample_id"]: (p["p_scam"] if p["label"] == "Non-scam" else 1 - p["p_scam"])
                for p in preds if p["kind"] == "none"}
        rows = [dict(p, p_wrong=p["p_scam"] if p["label"] == "Non-scam" else 1 - p["p_scam"])
                for p in preds if p["kind"] in ("single", "full")]
        metrics = {"adversarial": adversarial_metrics(rows, base),
                   "no_argument_judge": judge_metrics([p for p in preds if p["kind"] == "none"], subtype_ids)}
    else:
        metrics = judge_metrics(preds, subtype_ids)
        if args.mode == "debate":
            def rate(side_key: str, part: str) -> float | None:
                vals = []
                for p in preds:
                    true_side = scc.LABEL_SIDE[p["label"]]
                    side = true_side if side_key == "true" else scc.SIDES[1 - scc.SIDES.index(true_side)]
                    c = p["debate_citations"][side][part]
                    if c["n"]:
                        vals.append(c["verified"] / c["n"])
                return mean(vals)
            metrics["debate"] = {
                "true_side_win_rate": metrics["binary"]["accuracy"],
                "format_failure_rate": mean([float(any(v["opening"] or v["rebuttal"]
                                                       for v in p["debate_format_failed"].values())) for p in preds]),
                "verified_citation_rate": {f"{s}_{part}": rate(s, part) for s in ("true", "wrong")
                                           for part in ("opening", "rebuttal")}}
    metrics["config"] = config
    metrics["gold_refresh"] = dict(refreshed)
    metrics["n_items"] = len(preds)
    write_json(out_dir / "metrics.json", metrics)
    brief = metrics.get("adversarial", {}) if args.mode == "adversarial" else {
        "accuracy": metrics["binary"]["accuracy"], "macro_f1": metrics["binary"]["macro_f1"],
        "fpr": metrics["false_positive_rate"], "fine_macro_f1": metrics.get("fine_grained", {}).get("macro_f1")}
    print(json.dumps(brief, indent=1)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
