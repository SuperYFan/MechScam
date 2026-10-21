"""Offline tests for the ScamCourt training / debate / evaluation scripts (no GPU, no API).

A small project is built with the real step-1/step-2 scripts (mocked API), exported with the
real cold-start export, and then every scamcourt/ script is run end to end with a fake torch.
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SCRIPTS = PKG.parent / "scripts"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))
import sc_utils  # noqa: E402
import scamcourt_common as scc  # noqa: E402

TRAIN = ["vidscamnet:scamA", "youtube_scam:nonB", "vidscamnet:scamC", "youtube_scam:nonF"]
TEST = ["youtube_scam:nonD", "vidscamnet:scamE"]
EXCLUDED = ["youtube_scam:nonF"]


def clone_video(root: Path, src: str, dst: str) -> None:
    """Copy a fixture video (sample, frames, manifest row, step-1/2 outputs) under a new ID."""
    ds = root / "data/mechscam"
    s_ds, s_vid = src.split(":")
    d_ds, d_vid = dst.split(":")
    shutil.copytree(ds / "samples" / s_ds / s_vid, ds / "samples" / d_ds / d_vid)
    sp = ds / "samples" / d_ds / d_vid / "sample.json"
    sample = json.loads(sp.read_text())
    sample.update(sample_id=dst, dataset_id=d_ds, video_id=d_vid)
    sp.write_text(json.dumps(sample))
    rows = [json.loads(l) for l in (ds / "manifest.jsonl").read_text().splitlines()]
    row = dict(next(r for r in rows if r["sample_id"] == src), sample_id=dst, dataset_id=d_ds, video_id=d_vid,
               sample_path=f"samples/{d_ds}/{d_vid}/sample.json")
    (ds / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows + [row]) + "\n")
    for sub in ("teacher_arguments", "true_openings"):
        src_path = root / "data/arguments" / sub / s_ds / f"{s_vid}.json"
        rec = json.loads(src_path.read_text())
        rec.update(sample_id=dst, dataset_id=d_ds, video_id=d_vid)
        out = root / "data/arguments" / sub / d_ds / f"{d_vid}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))


def build_project(root: Path) -> None:
    import generate_scamcourt_false_arguments as gen
    import generate_scamcourt_true_openings as tro
    import test_scamcourt_false_arguments as t
    t.build_dataset(root)
    base = ["--project-root", str(root)]
    args = gen.parse_args(base + ["--all"])
    rows, _ = gen.choose_rows(args, t.TAXONOMY)
    jobs = {r["sample_id"]: gen.prepare_job(args, r, t.TAXONOMY) for r in rows}

    def reply(content):
        msg = SimpleNamespace(content=json.dumps(content), refusal=None)
        return SimpleNamespace(id="x", model="test-model", usage=None,
                               choices=[SimpleNamespace(finish_reason="stop", message=msg)])

    def create(**kw):
        payload = json.loads(kw["messages"][1]["content"][0]["text"])
        return reply(t.good_result(next(j for j in jobs.values() if j["wrong_side"] == payload["wrong_side"])))

    def create_true(**kw):
        payload = json.loads(kw["messages"][1]["content"].split("\n", 1)[1])
        item = next(i for i in payload["evidence_items"] if i["quote"])
        text = ("The decisive facts are plain on the record. " * 6) + "The judge should accept this position."
        return reply({"status": "generated", "review_reason": "", "opening": {
            "argument": text, "citations": [{"source_id": item["source_id"], "quote": item["quote"]}]}})

    client = lambda fn: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fn)))  # noqa: E731
    with patch.dict("os.environ", {"OPENAI_API_KEY": "x"}), redirect_stdout(io.StringIO()), \
            redirect_stderr(io.StringIO()):
        with patch.object(gen, "client_for", return_value=client(create)):
            assert gen.main(base + ["--all", "--execute", "--model", "test-model", "--workers", "1"]) == 0
        with patch.object(tro, "client_for", return_value=client(create_true)):
            assert tro.main(base + ["--all", "--execute", "--model", "test-model", "--workers", "1"]) == 0
    for src, dst in (("vidscamnet:scamA", "vidscamnet:scamC"), ("youtube_scam:nonB", "youtube_scam:nonF"),
                     ("youtube_scam:nonB", "youtube_scam:nonD"), ("vidscamnet:scamA", "vidscamnet:scamE")):
        clone_video(root, src, dst)
    splits = root / "data/splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "scamcourt_v3_seed42.json").write_text(json.dumps({"train": TRAIN, "test": TEST}))
    (root / "data/splits/exclusions.json").write_text(json.dumps(
        {"excluded": [{"sample_id": s} for s in EXCLUDED]}))


def run(main, argv: list[str]) -> int:
    with patch.object(sys, "argv", ["x"] + argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return main()


class UtilsTest(unittest.TestCase):
    def test_parse_outputs(self):
        op, err = sc_utils.parse_opening('```json\n{"claim_label":"Scam","claimed_subtype":"S03","argument":" a  b ",'
                                         '"citations":[{"source_id":"ASR_0001","quote":"x"},"bad"]}\n```',
                                         "prosecution", ["S03", "S06"])
        self.assertIsNone(err)
        self.assertEqual(op["argument"], "a b")
        self.assertEqual(len(op["citations"]), 1)
        self.assertIsNotNone(sc_utils.parse_opening('{"claim_label":"Non-scam"}', "prosecution", ["S03"])[1])
        self.assertIsNotNone(sc_utils.parse_opening(
            '{"claim_label":"Scam","claimed_subtype":"N03","argument":"x"}', "prosecution", ["S03"])[1])
        self.assertIsNotNone(sc_utils.parse_rebuttal("no json")[1])
        v = sc_utils.parse_verdict('prefix {"label": "Non-scam", "subtype": "N03", "content_role": "news"}')
        self.assertEqual((v["label"], v["subtype"], v["parsed"]), ("Non-scam", "N03", True))
        self.assertIsNone(sc_utils.parse_verdict("garbage")["label"])
        codes = {"S03": "Scam", "N06": "Non-scam"}
        v = sc_utils.parse_verdict('{"label": "N06"}', codes)  # label given as a category code
        self.assertEqual((v["label"], v["subtype"], v["label_recovered"]), ("Non-scam", "N06", "label_was_category_code"))
        self.assertEqual(sc_utils.parse_verdict('{"label": "non scam"}', codes)["label"], "Non-scam")
        self.assertEqual(sc_utils.parse_verdict('{"subtype": "S03"}', codes)["label"], "Scam")
        self.assertIsNone(sc_utils.parse_verdict('{"label": "X99"}', codes)["label"])

    def test_debate_views(self):
        import evaluate
        op = lambda lab, sub: {"claim_label": lab, "claimed_subtype": sub, "argument": "a", "citations": []}  # noqa
        d = {"openings": {"prosecution": op("Scam", "S03"), "defense": op("Non-scam", "N03")},
             "rebuttals": {"prosecution": {"rebuttal": "r1", "citations": []},
                           "defense": {"rebuttal": "r2", "citations": []}}}
        ctx = {"source_texts": {}, "frame_ids": set()}
        names = {"S03": "Pool", "N03": "Education"}
        get = lambda v: evaluate.transcript_of({"kind": "model_debate", "debate": d, "view": v}, ctx, names)  # noqa
        self.assertEqual(get("full").count("REBUTTAL"), 2)
        self.assertNotIn("REBUTTAL", get("openings"))
        self.assertIn("PROSECUTION OPENING", get("prosecution_only"))
        self.assertNotIn("DEFENSE", get("prosecution_only"))
        self.assertNotIn("PROSECUTION", get("defense_only"))

    def test_stratified_subsample(self):
        ids = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(2)]
        key = {i: i[0] for i in ids}
        sub = sc_utils.stratified_subsample(ids, key, 5, 1)
        self.assertEqual(len(sub), 5)
        self.assertEqual(sum(s.startswith("a") for s in sub), 4)
        self.assertEqual(sub, sc_utils.stratified_subsample(ids, key, 5, 1))

    def test_render_marks_unverified(self):
        ctx = {"source_texts": {"ASR_0001": "hello world"}, "frame_ids": {"F0001"}}
        check = scc.citation_checker(ctx)
        text = scc.format_citations([{"source_id": "ASR_0001", "quote": "hello"},
                                     {"source_id": "ASR_0001", "quote": "bye"},
                                     {"source_id": "F0001", "quote": ""}], checker=check)
        self.assertEqual(text.count("[verified]"), 2)
        self.assertEqual(text.count("[UNVERIFIED]"), 1)
        # Without a checker (teacher data) everything is marked verified, as in the export.
        self.assertNotIn("UNVERIFIED", scc.format_citations([{"source_id": "X", "quote": "bye"}]))

    def test_metrics(self):
        preds = [{"label": "Non-scam", "subtype": "N03", "content_role": "news", "has_risk_cues": True,
                  "p_scam": 0.7, "generated": "{}", "pred_subtype": "S03", "pred_role": "News", "parsed": True,
                  "pred_label_generated": "Scam"},
                 {"label": "Scam", "subtype": "S03", "content_role": "promo", "has_risk_cues": True,
                  "p_scam": 0.9, "generated": "{}", "pred_subtype": "S03", "pred_role": "x", "parsed": True,
                  "pred_label_generated": "Scam"}]
        m = sc_utils.judge_metrics(preds, ["S03", "N03"])
        self.assertEqual(m["false_positive_rate"]["nonscam_with_risk_cues"], 1.0)
        self.assertEqual(m["content_role_accuracy"], 0.5)
        self.assertEqual(m["fine_grained"]["accuracy"], 0.5)
        rows = [{"sample_id": "a", "protocol": "single", "p_wrong": 0.8, "strategy": "s1", "gold_label": "Scam",
                 "argument_strength": "normal", "gold_subtype": "S03", "qwen_selected": True,
                 "weak_case_mode": False},
                {"sample_id": "a", "protocol": "single", "p_wrong": 0.2, "strategy": "s2", "gold_label": "Scam",
                 "argument_strength": "weak", "gold_subtype": "S03", "qwen_selected": False,
                 "weak_case_mode": True}]
        a = sc_utils.adversarial_metrics(rows, {"a": 0.1})
        self.assertEqual(a["single"]["all_arguments"]["misled_rate"], 0.5)
        self.assertEqual(a["single"]["qwen_selected_hardest"]["misled_rate"], 1.0)
        self.assertEqual(a["single"]["normal_mode_only"]["n"], 1)
        self.assertEqual(a["single"]["all_arguments"]["mean_shift_p_wrong"], 0.4)


class OomSplitTest(unittest.TestCase):
    def test_batch_halves_on_oom(self):
        import fake_torch
        saved = {k: sys.modules.get(k) for k in fake_torch.FAKE_MODULES}
        saved["sc_model"] = sys.modules.pop("sc_model", None)
        try:
            fake_torch.install()
            import sc_model
            calls = []

            @sc_model._oom_split
            def fn(model, processor, batch, tag):
                calls.append(len(batch))
                if len(batch) > 1:
                    raise sys.modules["torch"].cuda.OutOfMemoryError()
                return [f"{tag}{batch[0]}"]

            self.assertEqual(fn(None, None, [1, 2, 3, 4, 5], "x"), ["x1", "x2", "x3", "x4", "x5"])
            self.assertEqual(calls[0], 5)

            @sc_model._oom_split
            def always(model, processor, batch):
                raise sys.modules["torch"].cuda.OutOfMemoryError()
            with self.assertRaises(sys.modules["torch"].cuda.OutOfMemoryError):
                always(None, None, [1, 2])
        finally:
            sys.modules.pop("sc_model", None)
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        build_project(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_end_to_end(self):
        import fake_torch
        saved = {k: sys.modules.get(k) for k in fake_torch.FAKE_MODULES}
        try:
            fake_torch.install()
            sys.modules.pop("sc_model", None)
            import sc_model
            sc_model.load_model = lambda *a, **k: (fake_torch.Proc(), fake_torch.Model())
            sc_model.model_device = lambda m: "cpu"
            for name in ("train_sft", "run_debate", "build_judge_data", "build_adversarial_testset", "evaluate"):
                sys.modules.pop(name, None)
            import build_adversarial_testset
            import build_judge_data
            import evaluate
            import run_debate
            import train_sft
            import build_scamcourt_coldstart_sft as export
            root, R = self.root, ["--project-root", str(self.root)]
            sft = "work/sft_coldstart"

            # Cold-start export of training videos only (scores skipped in this fixture).
            self.assertEqual(run(export.main, R + ["--no-scores", "--split-file",
                                                   "data/splits/scamcourt_v3_seed42.json",
                                                   "--roles", "false_opening,false_rebuttal,true_opening,"
                                                   "true_rebuttal,judge"]), 0)
            exported = [json.loads(l) for l in (root / sft / "false_opening.jsonl").read_text().splitlines()]
            self.assertTrue(exported and all(e["meta"]["sample_id"] in TRAIN for e in exported))
            # The shared builder reproduces an exported prompt exactly.
            ex = exported[0]
            ds = sc_utils.Dataset(root, sc_utils.DEFAULT_DATASET)
            _, ctx = ds.context(ex["meta"]["sample_id"], 12, 32000)
            rebuilt = scc.role_example(ex["messages"][0]["content"], ctx, scc.debater_extra("opening"), None, {})
            self.assertEqual(rebuilt["messages"][1], ex["messages"][1])
            self.assertEqual(rebuilt["images"], ex["images"])

            # Step 4: debater SFT (the excluded clone nonF is dropped).
            roles = [f"{sft}/{r}.jsonl" for r in ("false_opening", "false_rebuttal", "true_opening", "true_rebuttal")]
            self.assertEqual(run(train_sft.main, R + ["--data", *roles, "--output-dir", "runs/debater",
                                                      "--grad-accum", "2", "--epochs", "1", "--log-every", "1"]), 0)
            self.assertTrue((root / "runs/debater/adapter").is_dir())
            cfg = json.loads((root / "runs/debater/train_config.json").read_text())
            self.assertGreater(cfg["example_counts"]["dropped_excluded"], 0)
            self.assertTrue((root / "runs/debater/train_log.jsonl").exists())

            # Step 5: debates (2 samples per training video, greedy on test).
            for split, extra in (("train", ["--samples-per-video", "2"]), ("test", [])):
                self.assertEqual(run(run_debate.main, R + ["--split", split, "--adapter", "runs/debater/adapter",
                                                           "--output-dir", f"runs/debates_{split}", *extra]), 0)
            train_deb = [json.loads(l) for l in (root / "runs/debates_train/debates.jsonl").read_text().splitlines()]
            self.assertEqual(len(train_deb), 2 * (len(TRAIN) - len(EXCLUDED)))
            self.assertNotIn("youtube_scam:nonF", {r["sample_id"] for r in train_deb})
            r0 = train_deb[0]
            self.assertEqual(r0["openings"]["prosecution"]["claim_label"], "Scam")
            self.assertEqual(r0["citations"]["prosecution"]["opening"], {"n": 2, "verified": 1})
            self.assertFalse(any(v["opening"] for r in train_deb for v in r["format_failed"].values()))
            # The first (malformed) defense answer was regenerated, not replaced by a fallback.
            self.assertTrue(fake_torch.ANSWERS.get("defense_failed_once"))
            self.assertTrue(any(r["format_failed"]["defense"]["opening_attempts"] == 2 for r in train_deb))
            # Resume: nothing new.
            self.assertEqual(run(run_debate.main, R + ["--split", "test", "--adapter", "runs/debater/adapter",
                                                       "--output-dir", "runs/debates_test"]), 0)
            self.assertEqual(len((root / "runs/debates_test/debates.jsonl").read_text().splitlines()), len(TEST))
            with self.assertRaises(SystemExit):  # different settings into the same dir
                run(run_debate.main, R + ["--split", "test", "--adapter", "runs/debater/adapter",
                                          "--output-dir", "runs/debates_test", "--max-frames", "4"])

            # Step 6: judge data (own debates + no-transcript share) and the plain-SFT classifier data.
            self.assertEqual(run(build_judge_data.main, R + ["--debates", "runs/debates_train/debates.jsonl",
                                                             "--no-transcript-fraction", "0.5",
                                                             "--output", "data/judge_rollouts.jsonl"]), 0)
            jr = [json.loads(l) for l in (root / "data/judge_rollouts.jsonl").read_text().splitlines()]
            self.assertTrue(any("[UNVERIFIED]" in e["messages"][1]["content"] for e in jr))
            self.assertTrue(any("No debate transcript" in e["messages"][1]["content"] for e in jr))
            target = json.loads(jr[0]["messages"][2]["content"])
            self.assertEqual(target["label"], jr[0]["meta"]["gold_label"])
            self.assertEqual(run(build_judge_data.main, R + ["--classifier-only", "--output", "data/cls.jsonl"]), 0)
            self.assertEqual(len((root / "data/cls.jsonl").read_text().splitlines()), len(TRAIN) - len(EXCLUDED))
            self.assertEqual(run(train_sft.main, R + ["--data", f"{sft}/judge.jsonl", "data/judge_rollouts.jsonl",
                                                      "--output-dir", "runs/judge", "--grad-accum", "2"]), 0)
            # Examples of test videos never reach training.
            leak = dict(exported[0], meta=dict(exported[0]["meta"], sample_id=TEST[0]))
            (root / "data/leak.jsonl").write_text(json.dumps(leak) + "\n")
            with self.assertRaises(SystemExit):
                run(train_sft.main, R + ["--data", "data/leak.jsonl", "--output-dir", "runs/leak"])
            self.assertEqual(run(build_adversarial_testset.main, R), 0)
            adv = root / "data/adversarial_test.jsonl"
            advrows = [json.loads(l) for l in adv.read_text().splitlines()]
            self.assertEqual({r["sample_id"] for r in advrows}, set(TEST))
            self.assertEqual(len(advrows), 3 * len(TEST))
            self.assertTrue(all(r["true_opening"] for r in advrows))

            # Step 7: evaluation in all three modes.
            self.assertEqual(run(evaluate.main, R + ["--mode", "no_debate", "--adapter", "runs/judge/adapter",
                                                     "--output-dir", "runs/eval_plain"]), 0)
            m = json.loads((root / "runs/eval_plain/metrics.json").read_text())
            self.assertEqual(m["binary"]["n"], len(TEST))
            self.assertEqual(m["fine_grained"]["n"], len(TEST))
            self.assertEqual(m["parse_rate"], 1.0)
            self.assertEqual(run(evaluate.main, R + ["--mode", "debate", "--adapter", "runs/judge/adapter",
                                                     "--debates", "runs/debates_test/debates.jsonl",
                                                     "--output-dir", "runs/eval_debate"]), 0)
            m = json.loads((root / "runs/eval_debate/metrics.json").read_text())
            self.assertEqual(m["debate"]["verified_citation_rate"]["true_opening"], 0.5)
            self.assertEqual(run(evaluate.main, R + ["--mode", "adversarial", "--no-adapter",
                                                     "--output-dir", "runs/eval_adv"]), 0)
            m = json.loads((root / "runs/eval_adv/metrics.json").read_text())
            self.assertEqual(m["adversarial"]["single"]["all_arguments"]["n"], 3 * len(TEST))
            self.assertEqual(m["adversarial"]["full"]["all_arguments"]["n"], 3 * len(TEST))
            self.assertEqual(m["adversarial"]["no_argument"]["n_videos"], len(TEST))
            # Training-free debate baseline: untrained debaters (with role reminder) + untrained judge, all views.
            fake_torch.ANSWERS["defense_failed_once"] = True
            seen = []
            orig = fake_torch.scripted_answer
            fake_torch.scripted_answer = lambda prompt: (seen.append(prompt), orig(prompt))[1]
            self.assertEqual(run(run_debate.main, R + ["--split", "test", "--no-adapter", "--role-reminder",
                                                       "--output-dir", "runs/tf_debates"]), 0)
            fake_torch.scripted_answer = orig
            self.assertTrue(any("Reminder: you are the DEFENSE" in p for p in seen))
            self.assertTrue(json.loads((root / "runs/tf_debates/debate_config.json").read_text())["role_reminder"])
            for view in ("full", "openings", "prosecution_only", "defense_only"):
                self.assertEqual(run(evaluate.main, R + ["--mode", "debate", "--no-adapter", "--no-generate",
                                                         "--debates", "runs/tf_debates/debates.jsonl",
                                                         "--debate-view", view, "--output-dir", f"runs/tf_{view}"]), 0)
            m = json.loads((root / "runs/tf_prosecution_only/metrics.json").read_text())
            self.assertEqual((m["binary"]["n"], m["config"]["debate_view"]), (len(TEST), "prosecution_only"))
            n_pred = len((root / "runs/eval_adv/predictions.jsonl").read_text().splitlines())
            self.assertEqual(run(evaluate.main, R + ["--mode", "adversarial", "--no-adapter",
                                                     "--output-dir", "runs/eval_adv"]), 0)  # resume
            self.assertEqual(len((root / "runs/eval_adv/predictions.jsonl").read_text().splitlines()), n_pred)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
