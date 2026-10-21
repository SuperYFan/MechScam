#!/usr/bin/env python3
"""ScamCourt step 5: run full debates with a debater model (no teacher text involved).

For every video of --split (minus the exclusion list) and every sample k:
  1. OPENING  : prosecution and defense statements (same model, side-specific system prompt)
  2. REBUTTAL : each side answers the other's opening
Outputs are parsed and FORMAT-checked (side's label, a category of that side, non-empty text);
a failed output is regenerated up to --attempts times, then replaced by a marked fallback.
Citation truth is not enforced here: every citation is checked against the video inputs and
recorded, and transcripts later mark it [verified] or [UNVERIFIED] for the judge.

Prompts come from scripts/scamcourt_common.py (same as the SFT export); keep --max-frames /
--max-text-chars equal to the export settings. Results go to <output-dir>/debates.jsonl
(resumable, keyed by sample_id and k).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_DATASET, DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, FALLBACK_REBUTTAL, ROOT,  # noqa: E402
                      Dataset, check_config, citation_stats, fallback_opening, load_exclusions, load_split,
                      parse_opening, parse_rebuttal, read_jsonl, scc, select_ids, stratified_subsample)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--subsample", type=int, help="Stratified (by subtype) subset of N videos")
    p.add_argument("--limit", type=int, help="First N videos (smoke tests)")
    p.add_argument("--model-path", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--adapter", type=Path, help="Debater LoRA; omit with --no-adapter for the base model")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-per-video", type=int, default=1)
    p.add_argument("--temperature", type=float, help="Default 0 (greedy) for 1 sample, 0.8 otherwise")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--max-text-chars", type=int, default=32000)
    p.add_argument("--max-image-pixels", type=int, default=384 * 384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--role-reminder", action="store_true",
                   help="Append a 'keep your assigned side' reminder to every debater prompt (training-free "
                        "baseline: an untrained model otherwise often concedes in the rebuttal)")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; no model")
    return p


def role_reminder(side: str) -> str:
    label = scc.SIDE_LABEL[side]
    return (f"Reminder: you are the {side.upper()}. In this stage you must argue that the video is {label} "
            f"content, even if the evidence for {label} looks weak; never concede, never argue for the other "
            f"side, and never mention category codes in the prose. Reply with the JSON object only.")


def chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run_stage(items: list[dict], build, parse, generate_fn, args) -> None:
    """items get 'result', 'raw', 'error', 'attempts'. build(item)->messages; parse(item, text)->(value, err)."""
    pending = list(items)
    for attempt in range(1, args.attempts + 1):
        if not pending:
            break
        temperature = args.temperature if attempt == 1 else max(args.temperature, 0.7)
        for batch in chunks(pending, args.batch_size):
            texts = generate_fn([build(it) for it in batch], temperature)
            for it, text in zip(batch, texts):
                value, err = parse(it, text)
                it["raw"], it["attempts"] = text, attempt
                if value is not None:
                    it["result"], it["error"] = value, None
                else:
                    it["error"] = err
        pending = [it for it in pending if it.get("result") is None]


def main() -> int:
    args = parser().parse_args()
    if not args.dry_run and bool(args.adapter) == bool(args.no_adapter):
        raise SystemExit("ERROR: pass exactly one of --adapter or --no-adapter")
    if args.temperature is None:
        args.temperature = 0.0 if args.samples_per_video == 1 else 0.8
    if args.samples_per_video > 1 and args.temperature <= 0:
        raise SystemExit("ERROR: several samples per video need --temperature > 0")
    root = args.project_root.expanduser().resolve()
    ds = Dataset(root, args.dataset_dir)
    split = load_split(root, args.split_file)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    ids = select_ids(ds.rows, split, args.split, excluded)
    ids = stratified_subsample(ids, {i: ds.rows[i]["primary_subtype"] for i in ids}, args.subsample, args.seed)
    if args.limit:
        ids = ids[:args.limit]
    out_dir = (root / args.output_dir).resolve()
    config = {"split": args.split, "split_file": str(args.split_file), "subsample": args.subsample,
              "adapter": str(args.adapter) if args.adapter else None, "model_path": str(args.model_path),
              "samples_per_video": args.samples_per_video, "temperature": args.temperature, "top_p": args.top_p,
              "max_new_tokens": args.max_new_tokens, "attempts": args.attempts, "max_frames": args.max_frames,
              "max_text_chars": args.max_text_chars, "max_image_pixels": args.max_image_pixels,
              "seed": args.seed, "excluded": sorted(excluded), "role_reminder": args.role_reminder}
    out_path = out_dir / "debates.jsonl"
    done = set()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        check_config(out_dir / "debate_config.json", config)
        if out_path.exists():
            done = {(r["sample_id"], r["k"]) for r in read_jsonl(out_path)}
    jobs = [(sid, k) for sid in ids for k in range(args.samples_per_video) if (sid, k) not in done]
    print(json.dumps({"split": args.split, "videos": len(ids), "debates_done": len(done),
                      "debates_todo": len(jobs)}), flush=True)

    systems = {side: scc.debater_system_prompt(side, ds.taxonomy) for side in scc.SIDES}
    cats = {side: ds.categories(scc.SIDE_LABEL[side]) for side in scc.SIDES}
    contexts: dict[str, tuple[dict, dict]] = {}

    def ctx_of(sid: str) -> tuple[dict, dict]:
        if sid not in contexts:
            contexts[sid] = ds.context(sid, args.max_frames, args.max_text_chars)
        return contexts[sid]

    if args.dry_run:
        for sid in ids[:50]:
            _, ctx = ctx_of(sid)
            for side in scc.SIDES:
                scc.role_example(systems[side], ctx, scc.debater_extra("opening"), None, {})
        print(f"dry run ok ({min(50, len(ids))} videos checked)")
        return 0
    if not jobs:
        return 0

    import torch
    from sc_model import generate, load_model, to_messages
    torch.manual_seed(args.seed)
    processor, model = load_model(args.model_path.expanduser().resolve(),
                                  adapter=args.adapter.expanduser().resolve() if args.adapter else None)
    model.eval()

    def gen(messages_list, temperature):
        return generate(model, processor, messages_list, args.max_new_tokens, temperature, args.top_p)

    def messages_for(item: dict, extra: str) -> list[dict]:
        _, ctx = ctx_of(item["sid"])
        if args.role_reminder:
            extra += "\n\n" + role_reminder(item["side"])
        ex = scc.role_example(systems[item["side"]], ctx, extra, None, {})
        return to_messages(ex, root, args.max_image_pixels)[0]

    start, n_done = time.time(), 0
    with out_path.open("a", encoding="utf-8") as fh:
        for group in chunks(jobs, max(1, args.batch_size)):
            openings = [{"sid": sid, "k": k, "side": side} for sid, k in group for side in scc.SIDES]
            run_stage(openings, lambda it: messages_for(it, scc.debater_extra("opening")),
                      lambda it, t: parse_opening(t, it["side"], cats[it["side"]]), gen, args)
            by_job = {}
            for it in openings:
                by_job.setdefault((it["sid"], it["k"]), {})[it["side"]] = it
            rebuttals = []
            for (sid, k), sides in by_job.items():
                for side in scc.SIDES:
                    other = scc.SIDES[1 - scc.SIDES.index(side)]
                    own = sides[side].get("result") or fallback_opening(side, cats[side])
                    opp = sides[other].get("result") or fallback_opening(other, cats[other])
                    rebuttals.append({"sid": sid, "k": k, "side": side, "own": own, "opp": opp})
            run_stage(rebuttals, lambda it: messages_for(it, scc.debater_extra(
                "rebuttal", it["side"], it["own"], it["opp"], ds.names, scc.citation_checker(ctx_of(it["sid"])[1]))),
                lambda it, t: parse_rebuttal(t), gen, args)
            for sid, k in group:
                sample, ctx = ctx_of(sid)
                check = scc.citation_checker(ctx)
                record = {"sample_id": sid, "k": k, "gold_label": sample["annotation"]["binary_label"],
                          "gold_subtype": sample["annotation"]["primary_subtype"],
                          "true_side": scc.LABEL_SIDE[sample["annotation"]["binary_label"]],
                          "openings": {}, "rebuttals": {}, "format_failed": {}, "citations": {}, "raw": {}}
                for side in scc.SIDES:
                    o = by_job[(sid, k)][side]
                    r = next(x for x in rebuttals if x["sid"] == sid and x["k"] == k and x["side"] == side)
                    record["openings"][side] = o.get("result") or fallback_opening(side, cats[side])
                    record["rebuttals"][side] = r.get("result") or dict(FALLBACK_REBUTTAL)
                    record["format_failed"][side] = {"opening": o.get("result") is None,
                                                     "rebuttal": r.get("result") is None,
                                                     "opening_attempts": o.get("attempts"),
                                                     "rebuttal_attempts": r.get("attempts")}
                    record["citations"][side] = {
                        "opening": citation_stats(record["openings"][side]["citations"], check),
                        "rebuttal": citation_stats(record["rebuttals"][side]["citations"], check)}
                    record["raw"][side] = {"opening": o.get("raw"), "rebuttal": r.get("raw"),
                                           "opening_error": o.get("error"), "rebuttal_error": r.get("error")}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                contexts.pop(sid, None)
            n_done += len(group)
            rate = (time.time() - start) / n_done
            print(f"[{n_done}/{len(jobs)}] {rate:.1f}s/debate, eta {rate * (len(jobs) - n_done) / 3600:.1f}h",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
