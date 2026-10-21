#!/usr/bin/env python3
"""ScamCourt training-free baselines with any OpenAI-compatible chat model (closed or open source).

Same video inputs and the same role prompts as the trained ScamCourt models
(scripts/scamcourt_common.py); frames are sent as base64 images. Protocols:

  direct       the model judges the video alone                              (1 call / video)
  self_debate  the same model plays prosecution and defense (opening + rebuttal) and then judges
               the transcript; citations are checked and marked [verified]/[UNVERIFIED]
                                                                              (5 calls / video)
  consultancy  one-sided persuasion: the judge sees only the prosecution opening, or only the defense
               opening, of the self_debate run (one advocate is on the true side, one on the wrong side;
               no extra debater calls)                                        (2 judge calls / video)
  attack       the judge sees teacher-written wrong-side arguments of the adversarial test set
               (build_adversarial_testset.py): single = wrong opening only, full = complete
               four-turn debate; compare with `direct` (no argument)          (2 calls / argument)

Debater prompts end with the same "keep your assigned side" reminder as the local training-free
baseline (run_debate.py --role-reminder); --no-role-reminder turns it off.
Labels come from the parsed JSON answer (no log-probabilities); an unparsable answer is retried
once with feedback and then counted as wrong. Every call is cached in <output-dir>/<protocol>.jsonl,
so runs resume after interruption. Usage (tokens) is summed in metrics.json.

  read -rsp 'API key: ' OPENAI_API_KEY; export OPENAI_API_KEY; echo
  python scamcourt/api_baselines.py --list-models
  python scamcourt/api_baselines.py --model <model> --probe
  python scamcourt/api_baselines.py --model <model> --protocols direct,self_debate,consultancy,attack \
      --output-dir scamcourt/runs/api/<model>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_utils import (DEFAULT_DATASET, DEFAULT_EXCLUSIONS, DEFAULT_SPLIT, FALLBACK_REBUTTAL, LABELS, ROOT,  # noqa: E402
                      Dataset, adversarial_metrics, check_config, citation_stats, refresh_gold, fallback_opening, has_risk_cues,
                      judge_metrics, load_exclusions, load_split, mean, parse_opening, parse_rebuttal,
                      parse_verdict, read_jsonl, scc, select_ids, stratified_subsample, write_json)

from run_debate import role_reminder  # noqa: E402

PROTOCOLS = ("direct", "self_debate", "consultancy", "attack")  # run in this order (consultancy reuses debates)
CONSULTANCY_VIEWS = ("prosecution_only", "defense_only")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument("--split", default="test")
    p.add_argument("--subsample", type=int, help="Stratified subset of N videos (same subset for every model)")
    p.add_argument("--limit", type=int, help="First N videos (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocols", default="direct,self_debate,consultancy,attack")
    p.add_argument("--adv-file", type=Path, default=Path("data/adversarial_test.jsonl"))
    p.add_argument("--attack-arguments", choices=("all", "qwen_selected"), default="all",
                   help="all = every teacher argument (main metric); qwen_selected = 1 per video (cheaper)")
    p.add_argument("--attack-protocols", default="single,full")
    # model / endpoint
    p.add_argument("--model", help="Model name as the endpoint expects it")
    p.add_argument("--request-model", help="Name sent to the endpoint if it differs from --model (alias); not recorded")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0, help="Negative = do not send (model default)")
    p.add_argument("--max-tokens", type=int, default=1200, help="0 = do not send any token limit")
    p.add_argument("--max-tokens-param", choices=("max_completion_tokens", "max_tokens"),
                   default="max_completion_tokens")
    p.add_argument("--reasoning-effort",
                   help="Thinking budget sent to the endpoint (e.g. none / low / medium / high). Not sent when omitted, "
                        "so the model's own default applies - record it, and use a separate --output-dir per setting")
    p.add_argument("--json-mode", action="store_true", help="Send response_format=json_object")
    p.add_argument("--extra-body", help='Provider-specific JSON merged into the request body, e.g. '
                                        '\'{"thinking_mode": false}\' or \'{"enable_thinking": false}\'')
    p.add_argument("--no-role-reminder", action="store_true",
                   help="Do not append the 'keep your assigned side' reminder to debater prompts")
    p.add_argument("--image-detail", choices=("auto", "low", "high"), help="Sent only when given")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--quota-retry", action="store_true",
                   help="Treat 429 'exceeded your current quota' as a temporary token-rate limit: wait (up to 2 min "
                        "between tries, about an hour in total) and retry instead of failing the item")
    p.add_argument("--attempts", type=int, default=2, help="Tries per role output (2 = one retry with feedback)")
    # inputs (keep equal to the trained models' evaluation)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--max-text-chars", type=int, default=32000)
    p.add_argument("--max-image-pixels", type=int, default=384 * 384)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--list-models", action="store_true", help="Print the endpoint's model list and exit")
    p.add_argument("--probe", action="store_true", help="One judge call on one video, printed; nothing saved")
    p.add_argument("--rescore", action="store_true",
                   help="No API calls: re-read the saved answers in --output-dir with the current parser and "
                        "rewrite metrics.json (settings are taken from its api_config.json)")
    p.add_argument("--dry-run", action="store_true", help="Build all requests, print sizes; no API calls")
    return p


# ---------------------------------------------------------------- API

_local = threading.local()


def client(args):
    if not hasattr(_local, "client"):
        from openai import OpenAI
        key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit(f"ERROR: set {args.api_key_env} (read -rsp 'API key: ' {args.api_key_env}; "
                             f"export {args.api_key_env})")
        _local.client = OpenAI(api_key=key, base_url=args.base_url.rstrip("/"), timeout=args.timeout, max_retries=0)
    return _local.client


QUOTA_RETRY = False  # set by --quota-retry


def transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    body = str(getattr(exc, "body", "") or exc).lower()
    if status == 429 and any(t in body for t in ("insufficient_quota", "insufficient_balance", "credit")):
        # Usually an exhausted balance (do not retry). Some providers (e.g. Alibaba Model Studio) use the same
        # code for a token-rate quota that resets after a while: --quota-retry waits and retries instead.
        return QUOTA_RETRY and "insufficient_balance" not in body
    if status == 400:
        return "could not be processed" in body
    return status in (408, 409, 429) or (isinstance(status, int) and status >= 500) or \
        type(exc).__name__ in ("APIConnectionError", "APITimeoutError")


class ContentRefused(Exception):
    """The provider's safety / content filter rejected the request or blocked the answer."""


_REFUSAL_MARKERS = ("data_inspection_failed", "inappropriate content", "content_filter", "content filter",
                    "content_policy", "content policy", "responsible ai", "safety", "blocked", "prohibited",
                    "sensitive", "violat", "harmful")


def refusal(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    body = str(getattr(exc, "body", "") or exc).lower()
    return status in (400, 403, 422, 451) and any(m in body for m in _REFUSAL_MARKERS)


def chat(args, messages: list[dict], stop: threading.Event) -> tuple[str, dict]:
    kwargs = {"model": args.request_model or args.model, "messages": messages}
    if args.max_tokens > 0:  # 0 = do not send a token limit (endpoint default)
        kwargs[args.max_tokens_param] = args.max_tokens
    if args.temperature >= 0:
        kwargs["temperature"] = args.temperature
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    if args.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if args.extra_body:
        kwargs["extra_body"] = json.loads(args.extra_body)
    delay = 5.0
    retries = max(args.retries, 35) if args.quota_retry else args.retries
    for attempt in range(1, retries + 1):
        if stop.is_set():
            raise InterruptedError
        try:
            r = client(args).chat.completions.create(**kwargs)
            text = (r.choices[0].message.content or "") if r.choices else ""
            usage = getattr(r, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            finish = getattr(r.choices[0], "finish_reason", None) if r.choices else None
            u = {"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                 # thinking tokens (when the endpoint reports them) and answers cut off by --max-tokens
                 "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                 "truncated": int(finish == "length")}
            if finish == "content_filter" and not text.strip():
                raise ContentRefused("answer blocked by the provider's content filter (finish_reason=content_filter)")
            return text, u
        except ContentRefused:
            raise
        except Exception as exc:  # noqa: BLE001
            if refusal(exc):
                raise ContentRefused(str(getattr(exc, "body", "") or exc)[:500]) from exc
            if attempt == retries or not transient(exc):
                raise
            time.sleep(min(delay, 120) + random.random())
            delay *= 2
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- messages

class Images:
    """base64 JPEG data URIs of the selected frames, resized to max_pixels (cached per video)."""

    def __init__(self, max_pixels: int):
        self.max_pixels, self.cache, self.lock = max_pixels, {}, threading.Lock()

    def uri(self, path: str) -> str:
        with self.lock:
            if path in self.cache:
                return self.cache[path]
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w * h > self.max_pixels:
                scale = (self.max_pixels / (w * h)) ** 0.5
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        with self.lock:
            if len(self.cache) > 2000:
                self.cache.clear()
            self.cache[path] = uri
        return uri


def api_messages(example: dict, images: Images, detail: str | None) -> list[dict]:
    system, user = example["messages"][0]["content"], example["messages"][1]["content"]
    parts = user.split("<image>")
    if len(parts) - 1 != len(example["images"]):
        raise ValueError("placeholder / image count mismatch")
    content = []
    for i, chunk in enumerate(parts):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if i < len(example["images"]):
            image = {"url": images.uri(example["images"][i])}
            if detail:
                image["detail"] = detail
            content.append({"type": "image_url", "image_url": image})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def ask(args, messages: list[dict], parse, stop, usage: Counter) -> tuple[object | None, str, str | None]:
    """Call, parse; on a parse error retry once with feedback. Returns (value, raw_text, error)."""
    convo, raw, err = list(messages), "", None
    for attempt in range(args.attempts):
        raw, u = chat(args, convo, stop)
        usage.update(u)
        usage["calls"] += 1
        value, err = parse(raw)
        if value is not None:
            return value, raw, None
        convo = convo + [{"role": "assistant", "content": raw or "(empty)"},
                         {"role": "user", "content": f"Your reply is not usable ({err}). Reply again with ONE JSON "
                                                     f"object exactly in the required output format."}]
    return None, raw, err


# ---------------------------------------------------------------- protocols

def verdict_parser(code_labels: dict):
    def parse(raw: str):
        v = parse_verdict(raw, code_labels)
        return (v, None) if v["label"] else (None, "no valid label")
    return parse


def reparse(row: dict, code_labels: dict) -> dict:
    """Re-read a stored judge answer with the current parser, so every model is scored by the same rules
    (also rows saved before a parser change)."""
    if not row.get("generated") and row.get("generated") != "":
        return row
    v = parse_verdict(row["generated"], code_labels)
    return dict(row, pred_label=v["label"], pred_subtype=v["subtype"], pred_role=v["content_role"],
                parsed=v["parsed"], label_recovered=v["label_recovered"])


class Runner:
    def __init__(self, args, root: Path, ds: Dataset):
        self.args, self.root, self.ds = args, root, ds
        self.images = Images(args.max_image_pixels)
        self.judge_system = scc.judge_system_prompt(ds.taxonomy)
        self.systems = {side: scc.debater_system_prompt(side, ds.taxonomy) for side in scc.SIDES}
        self.cats = {side: ds.categories(scc.SIDE_LABEL[side]) for side in scc.SIDES}

    def ctx(self, sid: str):
        return self.ds.context(sid, self.args.max_frames, self.args.max_text_chars)

    def messages(self, system: str, ctx: dict, extra: str) -> list[dict]:
        return api_messages(scc.role_example(system, ctx, extra, None, {}), self.images, self.args.image_detail)

    def judge(self, ctx, transcript, stop, usage) -> dict:
        try:
            v, raw, err = ask(self.args, self.messages(self.judge_system, ctx, scc.judge_extra(transcript)),
                              verdict_parser(self.ds.labels), stop, usage)
        except ContentRefused as exc:  # recorded as an outcome (counted as wrong in strict metrics)
            usage["refused"] += 1
            return {"pred_label": None, "pred_subtype": None, "pred_role": None, "parsed": False,
                    "label_recovered": None, "generated": None, "error": "refused", "refused": True,
                    "refusal": str(exc)[:300]}
        v = v or {"label": None, "subtype": None, "content_role": None, "parsed": False, "label_recovered": None}
        return {"pred_label": v["label"], "pred_subtype": v["subtype"], "pred_role": v["content_role"],
                "parsed": v["parsed"], "label_recovered": v["label_recovered"], "generated": raw, "error": err}

    def debater_extra(self, side: str, extra: str) -> str:
        return extra if self.args.no_role_reminder else extra + "\n\n" + role_reminder(side)

    def opening(self, side, ctx, stop, usage):
        extra = self.debater_extra(side, scc.debater_extra("opening"))
        try:
            v, raw, err = ask(self.args, self.messages(self.systems[side], ctx, extra),
                              lambda t: parse_opening(t, side, self.cats[side]), stop, usage)
        except ContentRefused:
            usage["refused"] += 1
            v, raw = None, "(refused by the provider's content filter)"
        return (v or fallback_opening(side, self.cats[side])), v is None, raw

    def rebuttal(self, side, own, opp, ctx, stop, usage):
        extra = self.debater_extra(side, scc.debater_extra("rebuttal", side, own, opp, self.ds.names,
                                                           scc.citation_checker(ctx)))
        try:
            v, raw, err = ask(self.args, self.messages(self.systems[side], ctx, extra), parse_rebuttal, stop, usage)
        except ContentRefused:
            usage["refused"] += 1
            v, raw = None, "(refused by the provider's content filter)"
        return (v or dict(FALLBACK_REBUTTAL)), v is None, raw

    def base_row(self, item_id: str, sid: str, sample: dict) -> dict:
        ann = sample["annotation"]
        return {"id": item_id, "sample_id": sid, "label": ann["binary_label"], "subtype": ann["primary_subtype"],
                "content_role": ann.get("content_role"), "has_risk_cues": has_risk_cues(sample)}

    def run(self, protocol: str, item: dict, stop) -> dict:
        usage = Counter()
        sid = item["sample_id"]
        sample, ctx = self.ctx(sid)
        row = self.base_row(item["id"], sid, sample)
        check = scc.citation_checker(ctx)
        if protocol == "direct":
            row.update(self.judge(ctx, None, stop, usage))
        elif protocol == "self_debate":
            openings, rebuttals, failed, raws = {}, {}, {}, {}
            for side in scc.SIDES:
                openings[side], failed[side + "_opening"], raws[side + "_opening"] = self.opening(side, ctx, stop, usage)
            for side in scc.SIDES:
                other = scc.SIDES[1 - scc.SIDES.index(side)]
                rebuttals[side], failed[side + "_rebuttal"], raws[side + "_rebuttal"] = self.rebuttal(
                    side, openings[side], openings[other], ctx, stop, usage)
            transcript = scc.render_transcript(openings, rebuttals, self.ds.names, check)
            row.update(self.judge(ctx, transcript, stop, usage))
            true_side = scc.LABEL_SIDE[row["label"]]
            row["debate"] = {"openings": openings, "rebuttals": rebuttals, "format_failed": failed, "raw": raws,
                             "citations": {("true" if s == true_side else "wrong"): {
                                 "opening": citation_stats(openings[s]["citations"], check),
                                 "rebuttal": citation_stats(rebuttals[s]["citations"], check)} for s in scc.SIDES}}
        elif protocol == "consultancy":
            side = item["view"].split("_")[0]
            op = item["debate"]["openings"][side]
            transcript = scc.render_transcript({side: op}, {}, self.ds.names, check)
            row.update(self.judge(ctx, transcript, stop, usage))
            row["consultant"] = {"view": item["view"], "side": side,
                                 "argues_true_side": side == scc.LABEL_SIDE[row["label"]],
                                 "citations": citation_stats(op["citations"], check)}
        else:  # attack
            a = item["adv"]
            if item["kind"] == "single":
                transcript = scc.render_transcript({a["wrong_side"]: a["wrong_opening"]}, {}, self.ds.names)
            else:
                transcript = scc.render_transcript(
                    {a["true_side"]: a["true_opening"], a["wrong_side"]: a["wrong_opening"]},
                    {a["wrong_side"]: a["wrong_rebuttal"], a["true_side"]: a["true_reply"]}, self.ds.names)
            row.update(self.judge(ctx, transcript, stop, usage))
            row.update(item_id=a["item_id"], protocol=item["kind"], strategy=a["strategy"],
                       argument_strength=a["argument_strength"], weak_case_mode=a["weak_case_mode"],
                       qwen_selected=a["qwen_selected"], gold_label=a["gold_label"], gold_subtype=a["gold_subtype"],
                       wrong_label=a["wrong_label"])
        row["usage"] = dict(usage)
        return row


# ---------------------------------------------------------------- items / metrics

def build_items(args, root: Path, ids: list[str], protocol: str, debates: dict | None = None) -> list[dict]:
    if protocol == "consultancy":
        debates = debates or {}
        return [{"id": f"{sid}#{view}", "sample_id": sid, "view": view, "debate": debates[sid]["debate"]}
                for sid in ids if sid in debates for view in CONSULTANCY_VIEWS]
    if protocol != "attack":
        return [{"id": sid, "sample_id": sid} for sid in ids]
    wanted = set(ids)
    rows = [r for r in read_jsonl(root / args.adv_file) if r["sample_id"] in wanted]
    if args.attack_arguments == "qwen_selected":
        rows = [r for r in rows if r["qwen_selected"]]
    kinds = [k.strip() for k in args.attack_protocols.split(",") if k.strip()]
    return [{"id": f"{r['item_id']}#{k}", "sample_id": r["sample_id"], "kind": k, "adv": r}
            for r in rows for k in kinds if not (k == "full" and r["true_opening"] is None)]


def protocol_metrics(protocol: str, rows: list[dict], subtype_ids: list[str], direct: dict[str, dict]) -> dict:
    usage = Counter()
    for r in rows:
        usage.update(r.get("usage", {}))
    refused = [r for r in rows if r.get("refused")]
    out = {"n": len(rows), "usage": dict(usage),
           "label_recovered": dict(Counter(r.get("label_recovered") for r in rows if r.get("label_recovered"))),
           "refused": len(refused), "refusal_rate": mean([float(bool(r.get("refused"))) for r in rows]),
           "refused_by_gold_label": dict(Counter(r["label"] for r in refused))}
    if protocol == "attack":
        def wrong(r):  # strict: refused / unparsable answers count as misled (also counted separately)
            return float(r["pred_label"] != r["label"])
        scored = [dict(r, p_wrong=wrong(r)) for r in rows]
        base = {sid: wrong(d) for sid, d in direct.items()}
        out["unparsable"] = sum(r["pred_label"] is None for r in rows)
        out["attack"] = adversarial_metrics(scored, base)
        if not base:
            out["note"] = "run the direct protocol for the no-argument baseline"
        return out
    out.update(judge_metrics(rows, subtype_ids))  # strict: refused / unparsable answers count as wrong
    if refused:
        answered = [r for r in rows if not r.get("refused")]
        out["answered_only"] = {k: v for k, v in judge_metrics(answered, subtype_ids).items()
                                if k in ("binary", "false_positive_rate", "fine_grained")}
    if protocol == "self_debate":
        fails = [r["debate"]["format_failed"] for r in rows]
        out["debate"] = {
            "format_failure_rate": mean([float(any(f.values())) for f in fails]),
            "verified_citation_rate": {
                f"{who}_{part}": mean([r["debate"]["citations"][who][part]["verified"] /
                                       r["debate"]["citations"][who][part]["n"]
                                       for r in rows if r["debate"]["citations"][who][part]["n"]])
                for who in ("true", "wrong") for part in ("opening", "rebuttal")}}
    if protocol == "consultancy":
        for flag, name in ((True, "consultant_argues_true_side"), (False, "consultant_argues_wrong_side")):
            sub = [r for r in rows if r["consultant"]["argues_true_side"] is flag]
            out[name] = {"n": len(sub), "accuracy": mean([float(r["pred_label"] == r["label"]) for r in sub])}
        for view in CONSULTANCY_VIEWS:
            sub = [r for r in rows if r["consultant"]["view"] == view]
            out[view] = {"n": len(sub), "accuracy": mean([float(r["pred_label"] == r["label"]) for r in sub]),
                         "predicted_scam_rate": mean([float(r["pred_label"] == "Scam") for r in sub])}
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    args = parser().parse_args()
    global QUOTA_RETRY
    QUOTA_RETRY = args.quota_retry
    root = args.project_root.expanduser().resolve()
    if args.list_models:
        for m in client(args).models.list().data:
            print(m.id)
        return 0
    if not args.model:
        raise SystemExit("ERROR: --model is required")
    ds = Dataset(root, args.dataset_dir)
    split = load_split(root, args.split_file)
    excluded = load_exclusions(root, args.exclusions if str(args.exclusions) else None)
    ids = select_ids(ds.rows, split, args.split, excluded)
    ids = stratified_subsample(ids, {i: ds.rows[i]["primary_subtype"] for i in ids}, args.subsample, args.seed)
    if args.limit:
        ids = ids[:args.limit]
    runner = Runner(args, root, ds)
    stop = threading.Event()

    if args.probe:
        sample, ctx = runner.ctx(ids[0])
        usage = Counter()
        row = runner.judge(ctx, None, stop, usage)
        print(json.dumps({"sample_id": ids[0], "gold": sample["annotation"]["binary_label"], **row,
                          "usage": dict(usage)}, ensure_ascii=False, indent=1))
        return 0

    requested = {p.strip() for p in args.protocols.split(",") if p.strip()}
    if bad := requested - set(PROTOCOLS):
        raise SystemExit(f"ERROR: unknown protocols {sorted(bad)}")
    protocols = [p for p in PROTOCOLS if p in requested]
    if args.rescore:
        protocols = []
    calls_per = {"direct": 1, "self_debate": 5, "consultancy": 1, "attack": 1}
    plan = {p: build_items(args, root, ids, p) for p in protocols if p != "consultancy"}
    if "consultancy" in protocols:  # filled in after self_debate; 2 judge calls per video
        plan["consultancy"] = [None] * (2 * len(ids))
    print(json.dumps({"model": args.model, "videos": len(ids),
                      "items": {p: len(v) for p, v in plan.items()},
                      "approx_calls": sum(len(v) * calls_per[p] for p, v in plan.items())}, indent=1), flush=True)
    if args.dry_run:
        sample, ctx = runner.ctx(ids[0])
        msgs = runner.messages(runner.judge_system, ctx, scc.judge_extra(None))
        text_chars = sum(len(c.get("text", "")) for c in msgs[1]["content"]) + len(msgs[0]["content"])
        img_bytes = sum(len(c["image_url"]["url"]) for c in msgs[1]["content"] if c["type"] == "image_url")
        print(json.dumps({"example_video": ids[0], "text_chars": text_chars,
                          "images": len(ctx["images"]), "image_base64_chars": img_bytes}, indent=1))
        return 0
    if args.output_dir is None:
        raise SystemExit("ERROR: --output-dir is required")
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"model": args.model, "base_url": args.base_url, "split": args.split, "subsample": args.subsample,
              "seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens,
              "reasoning_effort": args.reasoning_effort, "json_mode": args.json_mode,
              "image_detail": args.image_detail, "max_frames": args.max_frames,
              "max_text_chars": args.max_text_chars, "max_image_pixels": args.max_image_pixels,
              "role_reminder": not args.no_role_reminder, "extra_body": args.extra_body,
              "judge_prompt_sha256": hashlib.sha256(runner.judge_system.encode()).hexdigest()}
    if args.rescore:
        if not (out_dir / "api_config.json").exists():
            raise SystemExit(f"ERROR: nothing to rescore in {out_dir}")
        config = json.loads((out_dir / "api_config.json").read_text(encoding="utf-8"))
    else:
        check_config(out_dir / "api_config.json", config)

    refresh_stats: dict[str, dict] = {}

    def load_rows(path: Path) -> dict[str, dict]:
        """Saved rows, with gold labels refreshed from the current dataset (see sc_utils.refresh_gold)."""
        rows, stats = refresh_gold({r["id"]: r for r in read_jsonl(path)}, ds)
        if stats and path.name not in refresh_stats:
            refresh_stats[path.name] = dict(stats)
            print(f"[{path.name}] gold refreshed from the dataset: {dict(stats)}", flush=True)
        return rows

    results: dict[str, dict[str, dict]] = {}
    lock = threading.Lock()
    for protocol in protocols:
        if protocol == "consultancy":
            deb_path = out_dir / "self_debate.jsonl"
            debates = results.get("self_debate") or (
                load_rows(deb_path) if deb_path.exists() else {})
            plan[protocol] = build_items(args, root, ids, protocol, debates)
            if len(plan[protocol]) < 2 * len(ids):
                print(f"[consultancy] {len(ids) - len(plan[protocol]) // 2} videos have no self_debate yet; "
                      f"they are skipped (run self_debate first)", flush=True)
        path = out_dir / f"{protocol}.jsonl"
        done = load_rows(path) if path.exists() else {}
        todo = [it for it in plan[protocol] if it["id"] not in done]
        print(f"[{protocol}] {len(done)} done, {len(todo)} to run", flush=True)
        start, n_ok, n_err = time.time(), 0, Counter()
        with path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(runner.run, protocol, it, stop): it for it in todo}
            try:
                for fut in as_completed(futures):
                    it = futures[fut]
                    try:
                        row = fut.result()
                    except InterruptedError:
                        continue
                    except Exception as exc:  # noqa: BLE001 - keep going; the item is retried next run
                        n_err[type(exc).__name__] += 1
                        if sum(n_err.values()) <= 5:
                            print(f"  error on {it['id']}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                        continue
                    with lock:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        fh.flush()
                        done[row["id"]] = row
                    n_ok += 1
                    if n_ok % 25 == 0 or n_ok == len(todo):
                        rate = (time.time() - start) / n_ok
                        print(f"  [{protocol}] {n_ok}/{len(todo)} {rate:.1f}s/item, "
                              f"eta {rate * (len(todo) - n_ok) / 3600:.1f}h", flush=True)
            except KeyboardInterrupt:
                stop.set()
                for f in futures:
                    f.cancel()
                raise
        if n_err:
            print(f"  [{protocol}] {sum(n_err.values())} items failed ({dict(n_err)}); rerun the same command "
                  f"to retry them", flush=True)
        results[protocol] = done

    direct_path = out_dir / "direct.jsonl"
    direct = results.get("direct") or (load_rows(direct_path) if direct_path.exists() else {})
    metrics = {"config": config, "videos": len(ids), "protocols_run": protocols, "gold_refresh": refresh_stats,
               "attack_selection": {"arguments": args.attack_arguments, "protocols": args.attack_protocols}}
    # Metrics for every protocol that has results in this directory (also ones run in earlier commands).
    reported = []
    for protocol in PROTOCOLS:
        path = out_dir / f"{protocol}.jsonl"
        if protocol not in results:
            if not path.exists():
                continue
            results[protocol] = load_rows(path)
        if protocol not in plan:
            deb = results.get("self_debate") or (load_rows(out_dir / "self_debate.jsonl")
                                                 if (out_dir / "self_debate.jsonl").exists() else {})
            plan[protocol] = build_items(args, root, ids, protocol, deb)
        reported.append(protocol)
    for protocol in reported:
        wanted = {it["id"] for it in plan[protocol]}
        rows = [r for k, r in results[protocol].items() if k in wanted]
        rows = [reparse(r, ds.labels) for r in rows]
        metrics[protocol] = protocol_metrics(protocol, rows, list(ds.labels),
                                             {k: reparse(r, ds.labels) for k, r in direct.items()})
        metrics[protocol]["complete"] = len(rows) == len(wanted)
    write_json(out_dir / "metrics.json", metrics)
    brief = {p: ({k: metrics[p]["attack"][k]["all_arguments"]["misled_rate"] for k in metrics[p]["attack"]
                  if k != "no_argument"} if p == "attack" else
                 {"acc": metrics[p]["binary"]["accuracy"], "macro_f1": metrics[p]["binary"]["macro_f1"],
                  "fpr_risk_cues": metrics[p]["false_positive_rate"]["nonscam_with_risk_cues"],
                  "fine_macro_f1": metrics[p].get("fine_grained", {}).get("macro_f1")})
             for p in reported if metrics[p]["n"]}
    print(json.dumps(brief, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
