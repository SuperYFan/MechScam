"""Offline test of api_baselines.py with a fake OpenAI-compatible client."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import api_baselines as api  # noqa: E402
import build_adversarial_testset  # noqa: E402
from test_scamcourt_pipeline import TEST, build_project, run  # noqa: E402


class FakeClient:
    def __init__(self):
        self.lock, self.calls, self.failed_once = threading.Lock(), [], False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        msgs = kw["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        assert any(c["type"] == "image_url" and c["image_url"]["url"].startswith("data:image/jpeg;base64,")
                   for c in user)
        text = "".join(c.get("text", "") for c in user)
        with self.lock:
            self.calls.append(kw)
            first_judge_failure = "You are the JUDGE" in system and not self.failed_once
            if first_judge_failure:
                self.failed_once = True
        if first_judge_failure:
            answer = "I think it is fine."  # unparsable -> retried with feedback
        elif "You are the JUDGE" in system:
            label = "Scam" if ("PROSECUTION OPENING" in text and "DEFENSE OPENING" not in text) else "Non-scam"
            answer = json.dumps({"label": label, "subtype": "S03" if label == "Scam" else "N03",
                                 "content_role": "discussion_or_education", "verdict": "v", "citations": []})
        elif "Stage: REBUTTAL." in text:
            answer = json.dumps({"rebuttal": "No.", "citations": []})
        else:
            pros = "You are the PROSECUTION" in system
            answer = json.dumps({"claim_label": "Scam" if pros else "Non-scam", "claimed_subtype": "S03" if pros
                                 else "N03", "argument": "Arg.",
                                 "citations": [{"source_id": "TITLE_0001", "quote": "Staking video"}]})
        msg = SimpleNamespace(content=answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


class ApiBaselineTest(unittest.TestCase):
    def test_all_protocols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_project(root)
            R = ["--project-root", str(root)]
            self.assertEqual(run(build_adversarial_testset.main, R), 0)
            fake = FakeClient()
            with patch.object(api, "client", return_value=fake):
                argv = R + ["--model", "m", "--output-dir", "runs/api/m", "--workers", "2"]
                # direct first, the other protocols in a later command into the same directory
                self.assertEqual(run(api.main, argv + ["--protocols", "direct"]), 0)
                first = json.loads((root / "runs/api/m/metrics.json").read_text())
                self.assertEqual(first["direct"]["binary"]["n"], len(TEST))
                self.assertEqual(run(api.main, argv), 0)
                n_calls = len(fake.calls)
                self.assertEqual(run(api.main, argv), 0)  # resume: no new calls
                self.assertEqual(len(fake.calls), n_calls)
                with patch.object(sys, "argv", ["x"] + R + ["--model", "m", "--dry-run"]), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(api.main(), 0)
            # direct 2 + debate 2x5 + consultancy 2x2 judge calls (reusing the debates) + attack 12 + 1 retry
            self.assertEqual(n_calls, 2 + 10 + 4 + 12 + 1)
            debater_texts = ["".join(c.get("text", "") for c in k["messages"][1]["content"]) for k in fake.calls
                             if "You are the JUDGE" not in k["messages"][0]["content"]]
            self.assertTrue(debater_texts and all("Reminder: you are the" in t for t in debater_texts))
            self.assertEqual(fake.calls[0]["temperature"], 0.0)
            m = json.loads((root / "runs/api/m/metrics.json").read_text())
            self.assertEqual(m["direct"]["binary"]["n"], len(TEST))
            self.assertEqual(m["self_debate"]["debate"]["verified_citation_rate"]["true_opening"], 1.0)
            self.assertEqual(m["consultancy"]["consultant_argues_true_side"]["n"], len(TEST))
            self.assertEqual(m["consultancy"]["consultant_argues_wrong_side"]["n"], len(TEST))
            self.assertEqual(m["consultancy"]["prosecution_only"]["predicted_scam_rate"], 1.0)
            att = m["attack"]["attack"]
            self.assertEqual(att["single"]["all_arguments"]["n"], 3 * len(TEST))
            self.assertEqual(att["no_argument"]["n_videos"], len(TEST))
            self.assertTrue(all(m[p]["complete"] for p in ("direct", "self_debate", "consultancy", "attack")))


    def test_content_filter_refusal_is_recorded(self):
        class Refusal(Exception):
            status_code = 400
            body = {"code": "data_inspection_failed", "message": "Input data may contain inappropriate content."}

        calls = []

        def create(**kw):
            calls.append(kw)
            raise Refusal("Error code: 400 - data_inspection_failed")

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.assertTrue(api.refusal(Refusal("x")))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_project(root)
            argv = ["--project-root", str(root), "--model", "m", "--protocols", "direct",
                    "--output-dir", "runs/api/r", "--workers", "1"]
            with patch.object(api, "client", return_value=fake):
                self.assertEqual(run(api.main, argv), 0)
                n = len(calls)
                self.assertEqual(n, len(TEST))  # one call per video, no retries on a refusal
                self.assertEqual(run(api.main, argv), 0)  # refusals are final results, not retried
                self.assertEqual(len(calls), n)
            m = json.loads((root / "runs/api/r/metrics.json").read_text())
            self.assertEqual(m["direct"]["refusal_rate"], 1.0)
            self.assertEqual(m["direct"]["binary"]["accuracy"], 0.0)  # strict: refused = wrong



class RefreshGoldTest(unittest.TestCase):
    """Saved rows follow the current dataset labels after a dataset update."""

    def test_refresh(self):
        from sc_utils import refresh_gold

        class DS:
            def sample(self, sid):
                return {"annotation": {"binary_label": "Scam", "primary_subtype": "S03",
                                       "content_role": "promotion_or_instruction"}}, None
        rows = {
            "a": {"id": "a", "sample_id": "x", "label": "Non-scam", "subtype": "N04", "content_role": "mixed",
                  "pred_label": "Scam",
                  "debate": {"citations": {"true": {"n": 1}, "wrong": {"n": 2}}}},
            "b": {"id": "b", "sample_id": "x", "label": "Non-scam", "gold_label": "Non-scam", "pred_label": "Scam"},
            "c": {"id": "c", "sample_id": "x", "label": "Non-scam",
                  "consultant": {"side": "prosecution", "argues_true_side": False}},
            "d": {"id": "d", "sample_id": "x", "label": "Scam", "subtype": "S03",
                  "content_role": "promotion_or_instruction", "gold_label": "Scam"},
        }
        out, stats = refresh_gold(rows, DS())
        self.assertNotIn("b", out)  # adversarial item written for the old gold label
        self.assertEqual(out["a"]["label"], "Scam")
        self.assertEqual(out["a"]["subtype"], "S03")
        self.assertEqual(out["a"]["debate"]["citations"]["true"], {"n": 2})
        self.assertTrue(out["c"]["consultant"]["argues_true_side"])
        self.assertIs(out["d"], rows["d"])
        self.assertEqual(stats["dropped_stale_adversarial"], 1)
        self.assertEqual(rows["a"]["label"], "Non-scam")  # input rows are not mutated


if __name__ == "__main__":
    unittest.main()
