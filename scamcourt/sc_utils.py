"""ScamCourt training/evaluation utilities (pure Python, no torch).

Prompt formats are NOT defined here: they come from scripts/scamcourt_common.py, which the
cold-start export also uses, so every stage feeds the models exactly one format.
"""

from __future__ import annotations

import json
import re
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scamcourt_common as scc  # noqa: E402

LABELS = ("Scam", "Non-scam")
DEFAULT_SPLIT = Path("data/splits/scamcourt_v3_seed42.json")
DEFAULT_EXCLUSIONS = Path("data/splits/exclusions.json")
DEFAULT_DATASET = Path("data/mechscam")


# ---------------------------------------------------------------- IO

def read_jsonl(path: Path) -> Iterable[dict]:
    yield from scc.read_jsonl(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp, n = path.with_name(path.name + ".tmp"), 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def check_config(path: Path, config: dict) -> None:
    """Refuse to resume into an output dir made with different settings."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        # A setting added in a newer script version (absent from the old file) counts as unset (None).
        diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
        if diff:
            raise SystemExit(f"ERROR: {path.parent} was created with different settings ({diff}); "
                             f"use another --output-dir")
    write_json(path, config)


# ---------------------------------------------------------------- split / exclusions

def load_split(root: Path, split_file: Path) -> dict[str, set[str]]:
    data = json.loads((root / split_file).read_text(encoding="utf-8"))
    return {k: set(data[k]) for k in ("train", "test")}


def load_exclusions(root: Path, path: Path | None) -> set[str]:
    if path is None:
        return set()
    full = root / path
    if not full.exists():
        raise SystemExit(f"ERROR: exclusion list {full} not found (pass --exclusions '' to disable)")
    return {e["sample_id"] for e in json.loads(full.read_text(encoding="utf-8"))["excluded"]}


def select_ids(all_ids: Iterable[str], split: dict[str, set[str]], part: str, excluded: set[str]) -> list[str]:
    if part not in split:
        raise SystemExit(f"ERROR: unknown split {part!r}")
    return sorted(i for i in all_ids if i in split[part] and i not in excluded)


def stratified_subsample(ids: list[str], key: dict[str, str], n: int | None, seed: int) -> list[str]:
    """Up to n ids, proportional per stratum (largest remainder), deterministic."""
    if not n or n >= len(ids):
        return list(ids)
    groups = defaultdict(list)
    for i in ids:
        groups[key[i]].append(i)
    rng = random.Random(seed)
    quotas = {g: n * len(v) / len(ids) for g, v in groups.items()}
    take = {g: int(q) for g, q in quotas.items()}
    for g in sorted(quotas, key=lambda g: quotas[g] - take[g], reverse=True)[: n - sum(take.values())]:
        take[g] += 1
    out = []
    for g in sorted(groups):
        members = sorted(groups[g])
        rng.shuffle(members)
        out += members[: take[g]]
    return sorted(out)


# ---------------------------------------------------------------- dataset access

class Dataset:
    def __init__(self, root: Path, dataset_dir: Path):
        self.root = root
        self.dir = (root / dataset_dir).resolve()
        self.taxonomy = scc.read_json(self.dir / "taxonomy.json")
        self.labels, self.names = scc.taxonomy_maps(self.taxonomy)
        self.rows = {r["sample_id"]: r for r in scc.load_manifest(self.dir)}

    def sample(self, sid: str) -> tuple[dict, Path]:
        return scc.load_sample(self.dir, self.rows[sid])

    def context(self, sid: str, max_frames: int, max_text_chars: int) -> tuple[dict, dict]:
        sample, path = self.sample(sid)
        return sample, scc.debate_context(sample, path.parent, max_frames, max_text_chars)

    def categories(self, label: str) -> list[str]:
        return [k for k, v in self.labels.items() if v == label]


def refresh_gold(rows: dict[str, dict], ds: "Dataset") -> tuple[dict[str, dict], Counter]:
    """Bring saved prediction rows in line with the CURRENT dataset labels (e.g. after a dataset update).

    A model's answer does not depend on the gold label, so plain / debate / consultancy rows keep their
    prediction and only get the new label, subtype and content role (true/wrong-side bookkeeping is
    swapped accordingly). Adversarial rows carry the gold label their wrong-side argument was written
    against; when that label changed, the row is dropped so the item is run again with the new argument.
    """
    cache: dict[str, tuple] = {}
    out, stats = {}, Counter()
    for key, r in rows.items():
        sid = r["sample_id"]
        if sid not in cache:
            ann = ds.sample(sid)[0]["annotation"]
            cache[sid] = (ann["binary_label"], ann["primary_subtype"], ann.get("content_role"))
        label, subtype, role = cache[sid]
        if r.get("gold_label") is not None and r["gold_label"] != label:
            stats["dropped_stale_adversarial"] += 1
            continue
        if r.get("label") != label:
            stats["relabelled"] += 1
            r = dict(r, label=label)
            cits = (r.get("debate") or {}).get("citations")
            if isinstance(cits, dict) and set(cits) == {"true", "wrong"}:
                r["debate"] = dict(r["debate"], citations={"true": cits["wrong"], "wrong": cits["true"]})
            if isinstance(r.get("consultant"), dict):
                r["consultant"] = dict(r["consultant"],
                                       argues_true_side=r["consultant"]["side"] == scc.LABEL_SIDE[label])
        if r.get("subtype") != subtype or r.get("content_role") != role:
            stats["subtype_or_role_updated"] += 1
            r = dict(r, subtype=subtype, content_role=role)
        out[key] = r
    return out, stats


def has_risk_cues(sample: dict) -> bool:
    pe = sample.get("potential_elements") or {}
    return bool(pe.get("textual") or pe.get("visual"))


# ---------------------------------------------------------------- parsing model outputs

def parse_json_object(text: str) -> dict | None:
    """First JSON object in a generation (tolerates code fences / surrounding text)."""
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    return None


def _clean_citations(value: Any, limit: int) -> list[dict]:
    out = []
    for c in value if isinstance(value, list) else []:
        if isinstance(c, dict) and isinstance(c.get("source_id"), str) and isinstance(c.get("quote", ""), str):
            out.append({"source_id": c["source_id"], "quote": c.get("quote", "")})
    return out[:limit]


def parse_opening(text: str, side: str, categories: list[str]) -> tuple[dict | None, str | None]:
    """(opening, error). Only the FORMAT is checked here; citation truth is checked separately."""
    value = parse_json_object(text)
    if value is None:
        return None, "no JSON object"
    label = scc.SIDE_LABEL[side]
    if value.get("claim_label") != label:
        return None, f"claim_label must be {label}"
    if value.get("claimed_subtype") not in categories:
        return None, "claimed_subtype not a category of this side"
    if not isinstance(value.get("argument"), str) or not value["argument"].strip():
        return None, "empty argument"
    return {"claim_label": label, "claimed_subtype": value["claimed_subtype"],
            "argument": scc.normalize_text(value["argument"]),
            "citations": _clean_citations(value.get("citations"), 5)}, None


def parse_rebuttal(text: str) -> tuple[dict | None, str | None]:
    value = parse_json_object(text)
    if value is None:
        return None, "no JSON object"
    if not isinstance(value.get("rebuttal"), str) or not value["rebuttal"].strip():
        return None, "empty rebuttal"
    return {"rebuttal": scc.normalize_text(value["rebuttal"]),
            "citations": _clean_citations(value.get("citations"), 4)}, None


def fallback_opening(side: str, categories: list[str]) -> dict:
    return {"claim_label": scc.SIDE_LABEL[side], "claimed_subtype": categories[0],
            "argument": "No valid statement was produced.", "citations": []}


FALLBACK_REBUTTAL = {"rebuttal": "No valid rebuttal was produced.", "citations": []}


_LABEL_ALIASES = {"scam": "Scam", "non-scam": "Non-scam", "non scam": "Non-scam", "nonscam": "Non-scam",
                  "non_scam": "Non-scam", "not scam": "Non-scam", "not a scam": "Non-scam"}


_TRUNC_LABEL = re.compile(r'"label"\s*:\s*"(Scam|Non-scam)"')
_TRUNC_SUBTYPE = re.compile(r'"subtype"\s*:\s*"([SN]\d\d)"')


def _label_prefix(raw: str) -> str | None:
    """'Non-scam | General news ...' -> 'Non-scam' (the label followed by a separator, not more letters)."""
    t = raw.strip()
    for lab in ("Non-scam", "Scam"):
        if t.lower().startswith(lab.lower()) and len(t) > len(lab) and not t[len(lab)].isalnum() \
                and t[len(lab)] != "-":
            return lab
    return None


def parse_verdict(text: str, code_labels: dict | None = None) -> dict:
    """Judge answer -> label / subtype / role. One lenient rule set, applied to every model alike:
    a label written in another case ("scam", "non scam") is normalized; a label given as a category code
    (e.g. {"label": "N06"}) or missing while a valid subtype code is given takes that category's side
    (code_labels: category code -> "Scam" / "Non-scam"). `label_recovered` records which rule was used."""
    value = parse_json_object(text) or {}
    truncated = False
    if "label" not in value and text:
        # Answer cut off by the token limit before the JSON closed (the first complete object found is
        # then a nested citation, or nothing): read the leading fields as written.
        m = _TRUNC_LABEL.search(text)
        if m:
            value, truncated = {"label": m.group(1)}, True
            s = _TRUNC_SUBTYPE.search(text)
            if s:
                value["subtype"] = s.group(1)
    raw = value.get("label")
    sub = value.get("subtype") if isinstance(value.get("subtype"), str) else None
    label, how = None, None
    if raw in LABELS:
        label, how = raw, ("truncated_json" if truncated else None)
    elif isinstance(raw, str) and raw.strip().lower() in _LABEL_ALIASES:
        label, how = _LABEL_ALIASES[raw.strip().lower()], "case_or_spelling"
    elif isinstance(raw, str) and _label_prefix(raw):
        label, how = _label_prefix(raw), "label_with_suffix"
    elif code_labels and isinstance(raw, str) and raw.strip() in code_labels:
        label, how = code_labels[raw.strip()], "label_was_category_code"
        sub = sub or raw.strip()
    elif code_labels and raw is None and sub in code_labels:
        label, how = code_labels[sub], "label_from_subtype"
    return {"label": label, "subtype": sub,
            "content_role": value.get("content_role") if isinstance(value.get("content_role"), str) else None,
            "parsed": bool(value), "label_recovered": how}


def citation_stats(citations: list[dict], checker) -> dict:
    ok = sum(bool(checker(c)) for c in citations)
    return {"n": len(citations), "verified": ok}


# ---------------------------------------------------------------- metrics

def _r(x: float) -> float:
    return round(x, 4)


def mean(xs: list[float]) -> float | None:
    return _r(sum(xs) / len(xs)) if xs else None


def prf(gold: list[str], pred: list[str | None], classes: list[str]) -> dict:
    per, conf = {}, {g: {c: 0 for c in classes} for g in classes}
    for g, p in zip(gold, pred):
        if g in conf and p in conf[g]:
            conf[g][p] += 1
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        npred = sum(p == c for p in pred)
        sup = sum(g == c for g in gold)
        pr, rc = (tp / npred if npred else 0.0), (tp / sup if sup else 0.0)
        per[c] = {"support": sup, "predicted": npred, "precision": _r(pr), "recall": _r(rc),
                  "f1": _r(2 * pr * rc / (pr + rc)) if pr + rc else 0.0}
    present = [c for c in classes if per[c]["support"]]
    return {"n": len(gold), "accuracy": _r(sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))),
            "macro_f1": _r(sum(per[c]["f1"] for c in present) / max(1, len(present))),
            "per_class": per, "confusion": conf}


def judge_metrics(preds: list[dict], subtype_ids: list[str]) -> dict:
    """preds: {label, subtype, content_role, has_risk_cues, p_scam, pred_subtype, pred_role, parsed}."""
    gold = [p["label"] for p in preds]
    # A local judge gives p(Scam); an API model gives a parsed label ("pred_label", None = unparsable,
    # counted as wrong).
    pred = [p["pred_label"] if "pred_label" in p else ("Scam" if p["p_scam"] >= 0.5 else "Non-scam")
            for p in preds]
    out = {"binary": prf(gold, pred, list(LABELS))}
    non = [(p, q) for p, q in zip(preds, pred) if p["label"] == "Non-scam"]
    out["false_positive_rate"] = {
        "all_nonscam": mean([float(q == "Scam") for _, q in non]),
        "nonscam_with_risk_cues": mean([float(q == "Scam") for p, q in non if p["has_risk_cues"]]),
        "nonscam_without_risk_cues": mean([float(q == "Scam") for p, q in non if not p["has_risk_cues"]]),
        "n_with_risk_cues": sum(1 for p, _ in non if p["has_risk_cues"])}
    out["false_negative_rate"] = mean([float(q != "Scam") for p, q in zip(preds, pred) if p["label"] == "Scam"])
    out["unparsable_label_rate"] = mean([float(q is None) for q in pred])
    gen = [p for p in preds if p.get("generated") is not None]
    if gen:
        out["fine_grained"] = prf([p["subtype"] for p in gen], [p.get("pred_subtype") for p in gen], subtype_ids)
        out["content_role_accuracy"] = mean([float((p.get("pred_role") or "").strip().lower()
                                                   == (p.get("content_role") or "").strip().lower()) for p in gen])
        out["parse_rate"] = mean([float(bool(p.get("parsed"))) for p in gen])
        local = [(p, q) for p, q in zip(preds, pred) if p.get("generated") is not None and "pred_label" not in p]
        if local:
            out["generated_label_agrees_with_p"] = mean([float(p.get("pred_label_generated") == q) for p, q in local])
    by_sub = defaultdict(list)
    for p, q in zip(preds, pred):
        by_sub[p["subtype"]].append(float(q == p["label"]))
    out["binary_accuracy_by_subtype"] = {k: mean(v) for k, v in sorted(by_sub.items())}
    return out


def adversarial_metrics(rows: list[dict], baselines: dict[str, float]) -> dict:
    """rows: one per (argument, protocol) with p_wrong; baselines: sample_id -> p_wrong without arguments."""
    out = {"no_argument": {"n_videos": len(baselines),
                           "misled_rate": mean([float(v >= 0.5) for v in baselines.values()]),
                           "mean_p_wrong": mean(list(baselines.values()))}}
    for protocol in sorted({r["protocol"] for r in rows}):
        items = [r for r in rows if r["protocol"] == protocol]

        def summary(sub: list[dict]) -> dict:
            right_before = [r for r in sub if baselines.get(r["sample_id"], 1.0) < 0.5]
            return {"n": len(sub), "misled_rate": mean([float(r["p_wrong"] >= 0.5) for r in sub]),
                    "induced_error_rate": mean([float(r["p_wrong"] >= 0.5) for r in right_before]),
                    "mean_p_wrong": mean([r["p_wrong"] for r in sub]),
                    "mean_shift_p_wrong": mean([r["p_wrong"] - baselines[r["sample_id"]] for r in sub
                                                if r["sample_id"] in baselines])}

        block = {"all_arguments": summary(items),
                 "normal_mode_only": summary([r for r in items if not r.get("weak_case_mode")]),
                 "qwen_selected_hardest": summary([r for r in items if r.get("qwen_selected")])}
        for key in ("gold_label", "strategy", "argument_strength", "gold_subtype"):
            groups = defaultdict(list)
            for r in items:
                groups[str(r.get(key))].append(r)
            block[f"by_{key}"] = {k: summary(v) for k, v in sorted(groups.items())}
        out[protocol] = block
    return out


def counter(values: Iterable) -> dict:
    return dict(Counter(values))
