"""Offline tests for the ScamCourt wrong-side argument pipeline (no API, no GPU)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_scamcourt_false_arguments as gen  # noqa: E402
import generate_scamcourt_true_openings as tro  # noqa: E402
import scamcourt_common as sc  # noqa: E402

TAXONOMY = {"binary_labels": ["Scam", "Non-scam"], "subtypes": [
    {"id": "S03", "label": "Scam", "stage": "II", "name": "Investment pool returns", "definition": "Pool."},
    {"id": "S06", "label": "Scam", "stage": "IV", "name": "Recruitment commissions", "definition": "Downlines."},
    {"id": "N02", "label": "Non-scam", "stage": None, "name": "Financial news", "definition": "News."},
    {"id": "N03", "label": "Non-scam", "stage": None, "name": "Education", "definition": "Tutorials."},
]}

ASR_SCAM = ("Deposit 100 dollars into the staking pool and you will earn three percent every day. "
            "Invite your friends to earn more. This is not financial advice, do your own research.")
ASR_NON = ("Today we review how staking works on major exchanges. Some platforms promise three percent "
           "daily returns, which is unrealistic. Always check the terms before you deposit anything.")


def make_sample(sample_id: str, label: str, subtype: str, asr: str) -> dict:
    dataset, vid = sample_id.split(":")
    scam = label == "Scam"
    return {
        "sample_id": sample_id, "dataset_id": dataset, "video_id": vid, "split": "train",
        "title": "Staking video", "description": "Link in description.",
        "asr": {"language": "en", "full_text": asr, "segments": [{"start": 0, "end": 5, "text": asr}]},
        "video_caption": "caption",
        "frames": [{"frame_id": f"F{i:04d}", "file": f"frames/{i:06d}.jpg"} for i in range(1, 5)],
        "frame_sets": {"caption": ["F0001"], "potential_visual_elements": ["F0001"],
                       "label_evidence": ["F0001", "F0002", "F0003", "F0004"]},
        "text_sources": {
            "potential_elements": [],
            "label_evidence": [
                {"source_id": "TITLE_0001", "kind": "TITLE", "text": "Staking video", "location": {}},
                {"source_id": "DESCRIPTION_0001", "kind": "DESCRIPTION", "text": "Link in description.",
                 "location": {}},
                {"source_id": "ASR_0001", "kind": "ASR", "text": asr, "location": {}}]},
        "potential_elements": {
            "textual": [{"element_id": "T001", "observation": "A daily return figure is stated.",
                         "dimensions": ["reward_or_return_claim"],
                         "evidence": [{"source_id": "ASR_0001", "quote": "three percent"}],
                         "context": "Framed as a review." if not scam else "Promotional framing.",
                         "potential_relevance": "Return claim.", "limitations": "No deposit shown."}],
            "visual": [{"element_id": "V001", "observation": "A dashboard.", "dimensions": [],
                        "frame_ids": ["F0002"], "context": "", "potential_relevance": "", "limitations": "x"}],
            "textual_context_notes": [{"observation": "A disclaimer appears.",
                                       "evidence": [{"source_id": "ASR_0001", "quote": "do your own research"}]}]
            if scam else [],
            "visual_context_notes": []},
        "annotation": {
            "binary_label": label, "primary_subtype": subtype, "secondary_subtypes": ["S06"] if scam else [],
            "content_role": "promotion_or_instruction" if scam else "discussion_or_education",
            "contextual_rationale": ("This video is scam content because S03 deposits are linked to returns."
                                     if scam else "This video is non-scam content because it reviews staking."),
            "fine_grained_rationale": "Deposits are linked to daily returns." if scam else "Educational review.",
            "evidence": ([{"evidence_id": "E001", "observation": "Deposit linked to return.",
                           "source_id": "ASR_0001", "quote": "Deposit 100 dollars into the staking pool"},
                          {"evidence_id": "E002", "observation": "Dashboard.", "source_id": "F0003", "quote": ""}]
                         if scam else
                         [{"evidence_id": "E001", "observation": "Review framing.", "source_id": "ASR_0001",
                           "quote": "Today we review how staking works"}])}}


def build_dataset(root: Path) -> Path:
    from PIL import Image
    ds = root / "data/mechscam"
    (ds / "samples").mkdir(parents=True)
    (ds / "taxonomy.json").write_text(json.dumps(TAXONOMY), encoding="utf-8")
    manifest = []
    for sid, label, sub, asr in (("vidscamnet:scamA", "Scam", "S03", ASR_SCAM),
                                 ("youtube_scam:nonB", "Non-scam", "N03", ASR_NON)):
        sample = make_sample(sid, label, sub, asr)
        folder = ds / "samples" / sample["dataset_id"] / sample["video_id"]
        (folder / "frames").mkdir(parents=True)
        for frame in sample["frames"]:
            Image.new("RGB", (32, 32), (10, 20, 30)).save(folder / frame["file"])
        (folder / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
        manifest.append({"sample_id": sid, "dataset_id": sample["dataset_id"], "video_id": sample["video_id"],
                         "split": "train", "binary_label": label, "primary_subtype": sub,
                         "sample_path": f"samples/{sample['dataset_id']}/{sample['video_id']}/sample.json",
                         "frame_count": 4})
    (ds / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n", encoding="utf-8")
    return ds


def good_result(job: dict) -> dict:
    if job["wrong_side"] == "defense":  # gold Scam
        quotes = ["This is not financial advice, do your own research.", "Invite your friends to earn more.",
                  "you will earn three percent every day"]
        sub = "N03"
    else:
        quotes = ["Some platforms promise three percent daily returns", "before you deposit anything",
                  "Today we review how staking works on major exchanges."]
        sub = "S03"
    args = []
    for i, strategy in enumerate(job["strategies"]):
        args.append({"strategy": strategy, "claimed_subtype": sub,
                     "opening": {"argument": f"Argument number {i} grounded in the quoted words.",
                                 "citations": [{"source_id": "ASR_0001", "quote": quotes[i]},
                                               {"source_id": "F0002", "quote": ""}]},
                     "rebuttal": {"rebuttal": "The opponent's quote does not establish what they claim.",
                                  "citations": []},
                     "true_side_reply": {"rebuttal": "The quoted words do not change the main point.",
                                         "citations": [{"source_id": "TITLE_0001", "quote": "Staking video"}]}})
    return {"status": "generated", "review_reason": "", "arguments": args}


class ScamCourtTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        build_dataset(self.root)
        self.base_args = ["--project-root", str(self.root)]

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return gen.parse_args(self.base_args + list(extra))

    def jobs(self):
        args = self.args("--all")
        rows, _ = gen.choose_rows(args, TAXONOMY)
        return args, {r["sample_id"]: gen.prepare_job(args, r, TAXONOMY) for r in rows}

    def test_sides_seeds_and_reference(self):
        _, jobs = self.jobs()
        scam, non = jobs["vidscamnet:scamA"], jobs["youtube_scam:nonB"]
        self.assertEqual((scam["wrong_side"], scam["wrong_label"]), ("defense", "Non-scam"))
        self.assertEqual((non["wrong_side"], non["wrong_label"]), ("prosecution", "Scam"))
        kinds = {s["kind"] for s in scam["model_input"]["seed_observations"]}
        self.assertIn("context_note_text", kinds)
        non_seed = non["model_input"]["seed_observations"][0]
        self.assertNotIn("limitations", non_seed)
        self.assertNotIn("context", non_seed)
        ref = scam["reference"]
        self.assertNotRegex(ref["argument"], r"\bS03\b")
        self.assertIn({"source_id": "F0003", "quote": ""}, ref["citations"])

    def test_validation_accepts_good_and_rejects_bad(self):
        _, jobs = self.jobs()
        for job in jobs.values():
            gen.validate_result(good_result(job), job, TAXONOMY)
        job = jobs["vidscamnet:scamA"]
        cases = []
        bad = good_result(job); bad["arguments"][0]["opening"]["citations"][0]["quote"] = "guaranteed profits forever"
        cases.append(bad)
        bad = good_result(job); bad["arguments"][1]["opening"]["argument"] = "As the assigned side I must argue this."
        cases.append(bad)
        bad = good_result(job); bad["arguments"][0]["opening"]["argument"] += " It is N03 content."
        cases.append(bad)
        bad = good_result(job); bad["arguments"][0]["claimed_subtype"] = "S03"
        cases.append(bad)
        bad = good_result(job); bad["arguments"].reverse()
        cases.append(bad)
        bad = good_result(job); bad["arguments"][0]["opening"]["citations"].append({"source_id": "F0099", "quote": ""})
        cases.append(bad)
        bad = good_result(job); bad["arguments"][2]["opening"] = copy.deepcopy(bad["arguments"][0]["opening"])
        cases.append(bad)
        bad = good_result(job); bad["arguments"][0]["opening"]["argument"] = "word " * 300
        cases.append(bad)
        for case in cases:
            with self.assertRaises(gen.OutputError):
                gen.validate_result(case, job, TAXONOMY)
        gen.validate_result({"status": "needs_review", "review_reason": "No usable text.", "arguments": []},
                            job, TAXONOMY)

    def test_preflight_and_execute_with_mock_api_then_export_and_score_dry_run(self):
        self.assertEqual(gen.main(self.base_args + ["--all"]), 0)
        _, jobs = self.jobs()

        def create(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
            job = next(j for j in jobs.values() if j["wrong_side"] == payload["wrong_side"])
            msg = SimpleNamespace(content=json.dumps(good_result(job)), refusal=None)
            return SimpleNamespace(id="r1", model="test-model", usage=None,
                                   choices=[SimpleNamespace(finish_reason="stop", message=msg)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(gen, "client_for", return_value=client), patch.dict("os.environ", {"OPENAI_API_KEY": "x"}):
            self.assertEqual(gen.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"]), 0)
            outs = sorted(p for p in (self.root / "data/arguments/teacher_arguments").glob("*/*.json")
                          if p.parent.name != "logs")
            self.assertEqual(len(outs), 2)
            record = json.loads(outs[0].read_text())
            self.assertEqual(record["review_status"], "generated")
            self.assertEqual(len(record["arguments"]), 3)
            # Rerun skips completed videos.
            self.assertEqual(gen.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"]), 0)

        script = HERE / "build_scamcourt_coldstart_sft.py"
        export = [sys.executable, str(script), "--project-root", str(self.root), "--no-scores",
                  "--roles", "false_opening,false_rebuttal,true_opening,true_rebuttal,judge"]
        out_dir = self.root / "work/sft_coldstart"
        # Without step-2 rewritten openings nothing is exported (template style would leak the answer).
        res = subprocess.run(export, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        summary = json.loads((out_dir / "export_summary.json").read_text())
        self.assertEqual(summary["counts"]["skipped_without_true_opening"], 2)

        # Step 2: rewrite the true openings (mock API).
        def create_true(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
            item = next(i for i in payload["evidence_items"] if i["quote"])
            text = ("The decisive facts are plain on the record. " * 6) + "The judge should accept this position."
            value = {"status": "generated", "review_reason": "",
                     "opening": {"argument": text, "citations": [{"source_id": item["source_id"],
                                                                  "quote": item["quote"].lower()}]}}
            msg = SimpleNamespace(content=json.dumps(value), refusal=None)
            return SimpleNamespace(id="t", model="test-model", usage=None,
                                   choices=[SimpleNamespace(finish_reason="stop", message=msg)])

        tclient = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_true)))
        with patch.object(tro, "client_for", return_value=tclient), patch.dict("os.environ", {"OPENAI_API_KEY": "x"}):
            self.assertEqual(tro.main(self.base_args + ["--all"]), 0)  # preflight
            self.assertEqual(tro.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"]), 0)
            self.assertEqual(tro.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"]), 0)  # resume
        saved = json.loads((self.root / "data/arguments/true_openings/vidscamnet/scamA.json").read_text())
        self.assertEqual(saved["review_status"], "generated")
        self.assertEqual(saved["citation_repairs"][0]["kind"], "case_punctuation")  # lower-cased quote snapped back

        res = subprocess.run(export, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        rows = [json.loads(l) for l in (out_dir / "false_opening.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        ex = rows[0]
        self.assertEqual(ex["messages"][1]["content"].count("<image>"), len(ex["images"]))
        target = json.loads(ex["messages"][2]["content"])
        self.assertNotEqual(target["claim_label"], ex["meta"]["gold_label"])
        judge = [json.loads(l) for l in (out_dir / "judge.jsonl").read_text().splitlines()]
        self.assertEqual(json.loads(judge[0]["messages"][2]["content"])["label"], judge[0]["meta"]["gold_label"])
        # Same system prompt for a side regardless of truthfulness.
        trues = [json.loads(l) for l in (out_dir / "true_opening.jsonl").read_text().splitlines()]
        pros_true = next(t for t in trues if t["meta"]["side"] == "prosecution")
        pros_false = next(r for r in rows if r["meta"]["side"] == "prosecution")
        self.assertEqual(pros_true["messages"][0]["content"], pros_false["messages"][0]["content"])
        self.assertEqual(pros_true["meta"]["true_opening"], "rewritten")
        self.assertTrue(json.loads(pros_true["messages"][2]["content"])["argument"].startswith("The decisive facts"))

        score = HERE / "score_scamcourt_false_arguments.py"
        res = subprocess.run([sys.executable, str(score), "--project-root", str(self.root), "--dry-run"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr + res.stdout)

    def test_weak_case_mode_reruns_only_needs_review(self):
        _, jobs = self.jobs()
        calls = []

        def make_client(weak_answer: bool):
            def create(**kwargs):
                payload = json.loads(kwargs["messages"][1]["content"][0]["text"])
                job = next(j for j in jobs.values() if j["wrong_side"] == payload["wrong_side"])
                calls.append((job["row"]["sample_id"], "WEAK-CASE MODE" in kwargs["messages"][0]["content"]))
                if job["wrong_side"] == "prosecution" and not weak_answer:
                    value = {"status": "needs_review", "review_reason": "No scam mechanism.", "arguments": []}
                else:
                    value = good_result(job)
                    if weak_answer:
                        for arg in value["arguments"]:
                            arg["argument_strength"] = "weak"
                msg = SimpleNamespace(content=json.dumps(value), refusal=None)
                return SimpleNamespace(id="r", model="test-model", usage=None,
                                       choices=[SimpleNamespace(finish_reason="stop", message=msg)])
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        out = self.root / "data/arguments/teacher_arguments/youtube_scam/nonB.json"
        with patch.dict("os.environ", {"OPENAI_API_KEY": "x"}):
            with patch.object(gen, "client_for", return_value=make_client(False)):
                gen.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"])
            self.assertEqual(json.loads(out.read_text())["review_status"], "needs_review")
            calls.clear()
            with patch.object(gen, "client_for", return_value=make_client(True)):
                self.assertEqual(gen.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1",
                                                            "--weak-case-mode"]), 0)
            self.assertEqual(calls, [("youtube_scam:nonB", True)])
            saved = json.loads(out.read_text())
            self.assertEqual(saved["review_status"], "generated")
            self.assertTrue(saved["weak_case_mode"])
            self.assertEqual({a["argument_strength"] for a in saved["arguments"]}, {"weak"})
            calls.clear()
            # A later normal run treats the weak-case output as complete, not stale.
            with patch.object(gen, "client_for", return_value=make_client(False)):
                self.assertEqual(gen.main(self.base_args + ["--all", "--execute", "--model", "test-model", "--workers", "1"]), 0)
            self.assertEqual(calls, [])
        # Weak mode requires the strength rating.
        weak_args = gen.parse_args(self.base_args + ["--all", "--weak-case-mode"])
        row = next(r for r in gen.choose_rows(weak_args, TAXONOMY)[0] if r["video_id"] == "nonB")
        job = gen.prepare_job(weak_args, row, TAXONOMY)
        with self.assertRaises(gen.OutputError):
            gen.validate_result(good_result(job), job, TAXONOMY)

    def test_citation_repair_snaps_only_safe_near_matches(self):
        texts = {"ASR_0001": "Something close to 40 percent which isn't that bad. I'm telling you to do the same.",
                 "ASR_0002": "we have some people making hundreds of dollars every single day"}
        fixed, kind = sc.repair_citation({"source_id": "ASR_0002", "quote": "something close to 40 percent, which isn't"},
                                         texts, set())
        self.assertEqual((fixed["source_id"], kind), ("ASR_0001", "case_punctuation+source_id"))
        self.assertIn(fixed["quote"], texts["ASR_0001"])
        fixed, kind = sc.repair_citation({"source_id": "ASR_0001", "quote": "I am telling you to do the same"},
                                         texts, set())
        self.assertEqual((fixed["quote"], kind), ("I'm telling you to do the same", "fuzzy"))
        for bad in ("we have some people making thousands of dollars every single day",
                    "guaranteed profits for everyone who joins today"):
            _, kind = sc.repair_citation({"source_id": "ASR_0002", "quote": bad}, texts, set())
            self.assertIsNone(kind)
        # Omitted words are restored; a quote across a chunk cut keeps the longer verbatim part.
        texts = {"DESCRIPTION_0001": "an affiliate advertising program designed to provide a means for sites to "
                                     "earn advertising fees by advertising and linking to amazon.com",
                 "ASR_0001": "who was performing well in the industry One thing I like",
                 "ASR_0002": "about Citi is that they are a leader and they always keep innovating",
                 "ASR_0003": "you should not",
                 "ASR_0004": "deposit money into the pool today friends"}
        fixed, kind = sc.repair_citation({"source_id": "DESCRIPTION_0001", "quote":
                                          "an affiliate advertising program designed to provide advertising fees"},
                                         texts, set())
        self.assertEqual(kind, "elision")
        self.assertEqual(fixed["quote"], "an affiliate advertising program designed to provide a means for sites "
                                         "to earn advertising fees")
        fixed, kind = sc.repair_citation({"source_id": "ASR_0002", "quote": "One thing I like about Citi is that "
                                          "they are a leader and they always keep innovating"}, texts, set())
        self.assertEqual((fixed["quote"], kind), (texts["ASR_0002"], "chunk_boundary"))
        _, kind = sc.repair_citation({"source_id": "ASR_0004", "quote": "you should not deposit money into the pool"},
                                     texts, set())
        self.assertIsNone(kind)  # dropping "not" would flip the meaning
        _, kind = sc.repair_citation({"source_id": "DESCRIPTION_0001", "quote":
                                      "an affiliate advertising program designed to earn 500 advertising fees"},
                                     texts, set())
        self.assertIsNone(kind)
        _, jobs = self.jobs()
        job = jobs["vidscamnet:scamA"]
        value = good_result(job)
        value["arguments"][0]["opening"]["citations"][0]["quote"] = "this is NOT financial advice do your own research"
        repairs = gen.apply_citation_repairs(value, job, 0.85)
        self.assertEqual(len(repairs), 1)
        gen.validate_result(value, job, TAXONOMY)

    def test_true_opening_validation(self):
        _, jobs = self.jobs()
        gen_job = jobs["vidscamnet:scamA"]
        record = {"sample_id": "vidscamnet:scamA", "input_fingerprint": "fp", "true_side": "prosecution",
                  "reference_opening": gen_job["reference"], "input_coverage": {
                      "source_ids": [s["source_id"] for s in gen_job["sources"]],
                      "frame_ids": sorted(gen_job["frame_ids"])},
                  "arguments": good_result(gen_job)["arguments"]}
        args = tro.parse_args(self.base_args + ["--all"])
        job = tro.prepare_job(args, record, gen_job["row"], TAXONOMY)
        text = "The record shows a deposit tied to daily returns and an invitation to recruit others. " * 3

        def result(argument=text, cits=None):
            return {"status": "generated", "review_reason": "", "opening": {
                "argument": argument,
                "citations": cits or [{"source_id": "ASR_0001", "quote": "Deposit 100 dollars into the staking pool"},
                                      {"source_id": "F0003", "quote": ""}]}}
        tro.validate_result(result(), job)
        bad_cases = [
            result("This video is scam content because deposits earn returns. " * 4),
            result(text + " It is S03."),
            result("Too short."),
            result(cits=[{"source_id": "ASR_0001", "quote": "Invite your friends to earn more."}]),  # not evidence
            result(cits=[{"source_id": "F0001", "quote": ""}]),  # frame not in reference evidence
            result("Your Honor, " + text),
            result(text + " This supports the prosecution's position."),
            result(text + " The transcript (ASR_0001) shows this."),
        ]
        for case in bad_cases:
            with self.assertRaises(tro.OutputError):
                tro.validate_result(case, job)
        # A side word that the video itself uses (e.g. "Defence Investiture Ceremony") is allowed in prose,
        # but naming a debate side is still rejected.
        job["source_texts"] = dict(job["source_texts"], DESCRIPTION_0001="held at the Defence Investiture Ceremony")
        tro.validate_result(result(text + " The award was given at the Defence Investiture Ceremony."), job)
        for case in (result(text + " The defence argues otherwise."), result(text + " The prosecution claims more."),
                     result(text + " My opponent ignores this.")):
            with self.assertRaises(tro.OutputError):
                tro.validate_result(case, job)

    def test_best_of_n_tie_break(self):
        import build_scamcourt_coldstart_sft as export
        score = {"sample_id": "vidscamnet:a", "arguments": [{"p_wrong": 0.1}, {"p_wrong": 0.7}, {"p_wrong": 0.2}]}
        self.assertEqual(export.pick_argument(score, 1e-3), (1, "best"))
        picks = {export.pick_argument({"sample_id": f"vidscamnet:v{i}", "arguments": [{"p_wrong": 0.0}] * 3},
                                      1e-3) for i in range(30)}
        self.assertEqual({p[0] for p in picks}, {0, 1, 2})  # ties spread over all strategies
        self.assertTrue(all(p[1] == "tie_hash" for p in picks))
        tie = {"sample_id": "x", "arguments": [{"p_wrong": 0.0}, {"p_wrong": 0.9}, {"p_wrong": 0.9004}]}
        self.assertIn(export.pick_argument(tie, 1e-3)[0], (1, 2))

    def test_option_order_is_counterbalanced(self):
        import score_scamcourt_false_arguments as score
        orders = {score.option_order(f"x|{i}") for i in range(20)}
        self.assertEqual(len(orders), 2)


if __name__ == "__main__":
    unittest.main()
