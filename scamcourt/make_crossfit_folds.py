#!/usr/bin/env python3
"""ScamCourt: 2-fold cross-fitting split of the TRAINING videos, for out-of-sample judge data.

Why: the judge is trained on debates produced by our own debater. If those debates are generated
on the same training videos the debater was fine-tuned on, they are in-sample (the debater has
seen the teacher arguments for exactly these videos) and look better than the debates the judge
will see on the test set. With cross-fitting, debater A is trained on half A and debates half B,
debater B is trained on half B and debates half A, so every training video gets an out-of-sample
debate. The test-set debates still come from the debater trained on the full training split.

Each fold file is a normal split file, so the existing scripts work unchanged:
  fold_a.json = {"train": half A, "test": half B}   (train_sft on A, run_debate --split test on B)
  fold_b.json = {"train": half B, "test": half A}
"test" in a fold file is the held-out training half, never the real test split (checked).
Halves are stratified by primary_subtype x dataset_id, like the main split. Deterministic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scamcourt_common import load_manifest  # noqa: E402


def halves(rows: list[dict], train_ids: set[str], seed: int) -> tuple[list[str], list[str]]:
    strata = defaultdict(list)
    for r in rows:
        if r["sample_id"] in train_ids:
            strata[(r["primary_subtype"], r["dataset_id"])].append(r["sample_id"])
    rng = random.Random(seed)
    a, b, odd = [], [], 0
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        cut = len(ids) // 2 + (len(ids) % 2 if odd % 2 == 0 else 0)
        odd += len(ids) % 2
        a += ids[:cut]
        b += ids[cut:]
    return sorted(a), sorted(b)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=Path("data/mechscam"))
    p.add_argument("--split-file", type=Path, default=Path("data/splits/scamcourt_v3_seed42.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("data/splits/crossfit"))
    args = p.parse_args()
    root = args.project_root.expanduser().resolve()
    main_split = json.loads((root / args.split_file).read_text(encoding="utf-8"))
    train, test = set(main_split["train"]), set(main_split["test"])
    rows = load_manifest((root / args.dataset_dir).resolve())
    a, b = halves(rows, train, args.seed)
    assert set(a) | set(b) == train and not set(a) & set(b) and not (set(a) | set(b)) & test
    label = {r["sample_id"]: r["binary_label"] for r in rows}
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, tr, held in (("fold_a", a, b), ("fold_b", b, a)):
        payload = {"train": tr, "test": held, "seed": args.seed, "parent_split": str(args.split_file),
                   "note": "cross-fitting fold: 'test' is the HELD-OUT HALF OF THE TRAINING SPLIT, "
                           "not the real test set"}
        path = out / f"{name}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8"))["train"] != tr:
            raise SystemExit(f"ERROR: {path} exists with different content; refusing to overwrite a frozen fold")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        summary[name] = {"path": str(path), "train": len(tr), "heldout": len(held),
                         "train_labels": dict(Counter(label[i] for i in tr)),
                         "heldout_labels": dict(Counter(label[i] for i in held))}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
