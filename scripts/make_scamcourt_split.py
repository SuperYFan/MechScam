#!/usr/bin/env python3
"""ScamCourt: stratified train/test split (4:1, no dev) for scam_video_dataset_v3.

Strata = primary_subtype x dataset_id (the inherited `split` field of the manifest is ignored).
Per stratum the videos are shuffled with a fixed seed and allocated by largest-remainder
rounding; every stratum with at least 3 videos gets >= 1 test video. Deterministic for a seed.
Output: {"train": [sample_id, ...], "test": [...], "stats": ..., "per_stratum": ...}, used as
--split-file by build_scamcourt_coldstart_sft.py and the later ScamCourt scripts.
Refuses to overwrite an existing (frozen) split unless --overwrite.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scamcourt_common import load_manifest  # noqa: E402


def allocate(n: int, test_ratio: float) -> tuple[int, int]:
    """(n_train, n_test) with largest-remainder rounding; >= 1 test video when n >= 3."""
    raw = [n * (1 - test_ratio), n * test_ratio]
    base = [int(x) for x in raw]
    for i in sorted(range(2), key=lambda i: raw[i] - base[i], reverse=True)[: n - sum(base)]:
        base[i] += 1
    if n >= 3 and base[1] == 0 and test_ratio > 0:
        base = [base[0] - 1, 1]
    return base[0], base[1]


def make_split(rows: list[dict], test_ratio: float, seed: int) -> dict:
    strata = defaultdict(list)
    for row in rows:
        strata[(row["primary_subtype"], row["dataset_id"])].append(row["sample_id"])
    rng = random.Random(seed)
    split, table = {"train": [], "test": []}, {}
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_train, n_test = allocate(len(ids), test_ratio)
        split["test"] += ids[:n_test]
        split["train"] += ids[n_test:]
        table[f"{key[0]}|{key[1]}"] = {"train": n_train, "test": n_test}
    for name in split:
        split[name].sort()
    if set(split["train"]) & set(split["test"]) or len(split["train"]) + len(split["test"]) != len(rows):
        raise AssertionError("split is not a partition of the manifest")
    label = {r["sample_id"]: r["binary_label"] for r in rows}
    sub = {r["sample_id"]: r["primary_subtype"] for r in rows}
    stats = {name: {"n": len(ids), "labels": dict(Counter(label[i] for i in ids)),
                    "subtypes": dict(sorted(Counter(sub[i] for i in ids).items()))}
             for name, ids in split.items()}
    return {"seed": seed, "test_ratio": test_ratio, "strata": "primary_subtype x dataset_id",
            **split, "stats": stats, "per_stratum": table}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--dataset-dir", type=Path, default=Path("data/mechscam"))
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("data/splits/scamcourt_v3_seed42.json"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not 0 < args.test_ratio < 1:
        p.error("--test-ratio must be in (0, 1)")
    root = args.project_root.expanduser().resolve()
    out = (root / args.output).resolve()
    if out.exists() and not args.overwrite:
        p.error(f"{out} exists; pass --overwrite or another --output (never silently change a frozen split)")
    payload = make_split(load_manifest((root / args.dataset_dir).resolve()), args.test_ratio, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "stats": payload["stats"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
