"""Minimal numpy stand-ins for torch / transformers / peft / qwen_vl_utils (plumbing tests only).

The fake tokenizer is character level, so the fake model can read its prompt back and answer
like a debater or judge would (valid JSON, with one deliberately malformed first answer to
exercise the retry path). Nothing here tests learning.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np

VOCAB = 512


class T:
    def __init__(self, a, dtype=None):
        self.a = np.array(a, dtype=dtype) if not isinstance(a, np.ndarray) else a

    @staticmethod
    def _w(o):
        return o.a if isinstance(o, T) else o

    def __getitem__(self, k):
        if isinstance(k, tuple):
            k = tuple(np.array(x) if isinstance(x, list) else x for x in k)
        return T(self.a[k])

    def __setitem__(self, k, v):
        self.a[k] = self._w(v)

    def clone(self): return T(self.a.copy())
    def float(self): return T(self.a.astype(float))
    def detach(self): return self
    def item(self): return float(self.a)
    def __float__(self): return float(self.a)
    def tolist(self): return self.a.tolist()
    def to(self, *_a, **_k): return self
    def backward(self): FakeTorch.backward_calls += 1
    def exp(self): return T(np.exp(self.a))
    @property
    def shape(self): return self.a.shape
    @property
    def dtype(self): return self.a.dtype
    def __ne__(self, o): return T(self.a != self._w(o))
    def sum(self): return T(self.a.sum())
    def __int__(self): return int(self.a)
    def reshape(self, *shape): return T(self.a.reshape(*shape))
    def __truediv__(self, o): return T(self.a / self._w(o))
    def __mul__(self, o): return T(self.a * self._w(o))
    def __add__(self, o): return T(self.a + self._w(o))
    def __sub__(self, o): return T(self.a - self._w(o))


class FakeTorch(types.ModuleType):
    backward_calls = 0


class Batch(dict):
    def to(self, *_a, **_k): return self


def encode(text: str) -> list[int]:
    return [ord(c) % VOCAB for c in text]


def decode(ids) -> str:
    return "".join(chr(int(i)) for i in ids if int(i) != 0)


class Tok:
    padding_side = "right"

    def encode(self, text, add_special_tokens=False): return encode(text)
    def __call__(self, text, add_special_tokens=False): return {"input_ids": encode(text)}
    def decode(self, ids): return decode(ids)


class Proc:
    def __init__(self):
        self.tokenizer = Tok()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            c = m["content"]
            parts.append(c if isinstance(c, str) else "".join(x.get("text", "<img>") for x in c))
        return "<sys>" + "\n<user>".join(parts) + "\n<assistant>\n"

    def __call__(self, text, return_tensors="pt", do_resize=False, images=None, padding=False):
        rows = [encode(t) for t in text]
        width = max(map(len, rows))
        ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, width - len(r):] = r
            mask[i, width - len(r):] = 1
        # Like Qwen3-VL, also return a per-token multimodal type tensor that must stay aligned with input_ids.
        return Batch({"input_ids": T(ids), "attention_mask": T(mask), "mm_token_type_ids": T(np.zeros_like(ids))})

    def batch_decode(self, ids, skip_special_tokens=True):
        return [decode(row) for row in ids.a]


ANSWERS = {"calls": 0}


def scripted_answer(prompt: str) -> str:
    """What a debater / judge would answer, read off the prompt."""
    ANSWERS["calls"] += 1
    if "You are the JUDGE" in prompt:
        return json.dumps({"label": "Scam", "subtype": "S03", "content_role": "promotion_or_instruction",
                           "verdict": "Deposits are linked to returns.", "citations": []})
    side = "PROSECUTION" if "You are the PROSECUTION" in prompt else "DEFENSE"
    if "Stage: REBUTTAL." in prompt:
        return json.dumps({"rebuttal": "The other side overstates the quoted words.", "citations": []})
    if side == "DEFENSE" and ANSWERS.get("defense_failed_once") is None:
        ANSWERS["defense_failed_once"] = True
        return "Sorry, here is my argument without JSON."  # forces one retry
    label, sub = ("Scam", "S03") if side == "PROSECUTION" else ("Non-scam", "N03")
    return json.dumps({"claim_label": label, "claimed_subtype": sub, "argument": f"The {side.lower()} view.",
                       "citations": [{"source_id": "TITLE_0001", "quote": "Staking video"},
                                     {"source_id": "ASR_0001", "quote": "words that are not in the video"}]})


class Model:
    def __init__(self):
        self.p = T(np.zeros(3))
        self.p.requires_grad = True
        self.rng = np.random.default_rng(0)

    def __call__(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=None, **kw):
        b, length = input_ids.shape
        for name, t in [("attention_mask", attention_mask), *kw.items()]:
            if t is not None and t.shape != input_ids.shape:  # the real model fails the same way
                raise IndexError(f"{name} {t.shape} does not match input_ids {input_ids.shape}")
        keep = min(logits_to_keep, length) if logits_to_keep else length
        logits = T(self.rng.normal(size=(b, keep, VOCAB)))
        return types.SimpleNamespace(loss=T(np.array(0.7)) if labels is not None else None, logits=logits)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, do_sample=False, **_):
        outs = [encode(scripted_answer(decode(row))) for row in input_ids.a]
        width = max(map(len, outs))
        gen = np.zeros((len(outs), width), dtype=np.int64)
        for i, o in enumerate(outs):
            gen[i, :len(o)] = o
        return T(np.concatenate([input_ids.a, gen], axis=1))

    def parameters(self): return iter([self.p])
    def train(self): pass
    def eval(self): pass
    def save_pretrained(self, path): Path(path).mkdir(parents=True, exist_ok=True)


def install() -> None:
    torch = FakeTorch("torch")
    torch.Tensor = T
    torch.bfloat16 = "bf16"
    torch.manual_seed = lambda s: None
    torch.inference_mode = contextlib.nullcontext
    torch.tensor = lambda data, dtype=None: T(np.array(data, dtype=dtype))
    torch.cat = lambda ts, dim=0: T(np.concatenate([t.a for t in ts], axis=dim))
    torch.ones_like = lambda t: T(np.ones_like(t.a))

    def softmax(x, dim=-1):
        z = x.a - x.a.max(axis=dim, keepdims=True)
        e = np.exp(z)
        return T(e / e.sum(axis=dim, keepdims=True))
    torch.softmax = softmax

    class OOM(Exception):
        pass
    torch.cuda = types.SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None, is_available=lambda: False,
                                       max_memory_allocated=lambda: 0)

    class AdamW:
        def __init__(self, params, lr, weight_decay=0.0):
            self.param_groups = [{"lr": lr}]
        def step(self): pass
        def zero_grad(self, set_to_none=True): pass
    torch.optim = types.SimpleNamespace(AdamW=AdamW)
    nn = types.ModuleType("torch.nn")
    nn.utils = types.SimpleNamespace(clip_grad_norm_=lambda p, n: None)
    F = types.ModuleType("torch.nn.functional")

    def cross_entropy(logits, target):
        x = logits.a - logits.a.max(axis=-1, keepdims=True)
        logp = x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
        t = target.a.astype(int)
        assert logits.a.shape[0] == t.shape[0], (logits.a.shape, t.shape)
        return T(np.array(-logp[np.arange(len(t)), t].mean()))
    F.cross_entropy = cross_entropy
    nn.functional = F
    torch.nn = nn
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.nn.functional": F})

    transformers = types.ModuleType("transformers")

    class Sched:
        def __init__(self, opt, warm, total): self.opt = opt
        def step(self): pass
        def get_last_lr(self): return [self.opt.param_groups[0]["lr"]]
    transformers.get_cosine_schedule_with_warmup = Sched
    sys.modules["transformers"] = transformers

    qvu = types.ModuleType("qwen_vl_utils")

    def process_vision_info(messages, image_patch_size=16):
        imgs = [c["image"] for m in messages if isinstance(m["content"], list) for c in m["content"]
                if c.get("type") == "image"]
        return (imgs or None), None
    qvu.process_vision_info = process_vision_info
    sys.modules["qwen_vl_utils"] = qvu


FAKE_MODULES = ("torch", "torch.nn", "torch.nn.functional", "transformers", "qwen_vl_utils", "sc_model")
