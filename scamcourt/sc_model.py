"""Qwen3-VL / GLM-4.1V (+ LoRA) helpers for ScamCourt: example -> chat messages, SFT encoding,
batched generation and two-way label probabilities.

The backend is chosen from the model's config.json (model_type "qwen3_vl" or "glm4v"). The Qwen3-VL
path is unchanged from the original version. GLM-4.1V differences:
  * model class Glm4vForConditionalGeneration;
  * frames are resized here (same pixel budget, sides rounded to the processor's patch*merge factor)
    instead of qwen_vl_utils;
  * the end-of-turn string is '<|user|>' (an EOS id; the template writes nothing after an answer);
  * LoRA is restricted to the language model (the GLM vision tower also has gate/up/down_proj);
  * an optional answer prefix (env SC_ANSWER_PREFIX, default "") is appended after the generation
    prompt in training and inference alike, e.g. to skip the thinking block of GLM-4.1V-9B-Thinking;
  * an optional generation-only prefix (env SC_GEN_PREFIX, e.g. '{') starts free generation as JSON for
    untrained models; it is prepended back to the returned text.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

# Long multimodal prompts of varying length fragment the CUDA cache; must be set before the first allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

END_OF_TURN = "<|im_end|>\n"
BACKEND = "qwen3_vl"
ANSWER_PREFIX = ""
GEN_PREFIX = ""
_FACTOR = 32


def model_type(model_path: Path) -> str:
    cfg = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
    return cfg.get("model_type", "")


def _end_of_turn(processor, model_path: Path) -> str:
    """Stop token appended to every SFT target for GLM-4.1V.

    Its chat template writes nothing after an assistant message, so the end of a turn is the start of
    the next user turn. '<|user|>' is one of the model's EOS ids, so a model trained on
    target + '<|user|>' stops there at generation time.
    """
    tok = processor.tokenizer
    gen = json.loads((Path(model_path) / "generation_config.json").read_text(encoding="utf-8"))
    eos = gen.get("eos_token_id") or []
    eos = eos if isinstance(eos, list) else [eos]
    tid = tok.convert_tokens_to_ids("<|user|>")
    if isinstance(tid, int) and tid in eos:
        return "<|user|>"
    return tok.decode([eos[0]]) if eos else (tok.eos_token or "")


def load_model(model_path: Path, adapter: Path | None = None, trainable: bool = False, lora: dict | None = None,
               gradient_checkpointing: bool = False):
    global BACKEND, END_OF_TURN, ANSWER_PREFIX, GEN_PREFIX, _FACTOR
    from transformers import AutoProcessor
    BACKEND = model_type(model_path)
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if BACKEND == "glm4v":
        from transformers import Glm4vForConditionalGeneration as ModelClass
        ip = processor.image_processor
        _FACTOR = int(getattr(ip, "patch_size", 14)) * int(getattr(ip, "merge_size", 2))
        END_OF_TURN = _end_of_turn(processor, model_path)
        ANSWER_PREFIX = os.environ.get("SC_ANSWER_PREFIX", "")
        GEN_PREFIX = os.environ.get("SC_GEN_PREFIX", "")
    elif BACKEND == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    else:
        raise ValueError(f"Unsupported model_type {BACKEND!r} in {model_path}")
    model = ModelClass.from_pretrained(str(model_path), dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
        if not trainable:
            # Inference: fold the LoRA weights into the base model. Same outputs, but no extra
            # LoRA activations over the whole (long) prompt during prefill, and faster decoding.
            model = model.merge_and_unload()
    elif lora is not None:
        from peft import LoraConfig, get_peft_model
        lora = dict(lora)
        if BACKEND == "glm4v":
            lora["target_modules"] = language_model_targets(model, lora["target_modules"])
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **lora))
    return processor, model


def language_model_targets(model, suffixes: list[str]) -> list[str]:
    """Full names of the language model's linear layers whose last name part is a LoRA target.

    GLM-4 fuses the MLP input projection into gate_up_proj; it is added whenever gate_proj/up_proj
    are requested. Vision-tower layers are excluded so that LoRA covers the same part as for Qwen3-VL.
    """
    wanted = set(suffixes) | ({"gate_up_proj"} if {"gate_proj", "up_proj"} & set(suffixes) else set())
    names = [n for n, m in model.named_modules()
             if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in wanted
             and "visual" not in n and "vision" not in n and "language_model" in n]
    if not names:
        raise ValueError("No language-model LoRA targets found; check the module names")
    return names


def model_device(model):
    return next(p for p in model.parameters()).device


def to_messages(example: dict, root: Path, max_pixels: int) -> tuple[list[dict], str | None]:
    """sharegpt example (user text with <image> placeholders + image paths) -> chat messages, target."""
    msgs = example["messages"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    target = msgs[2]["content"] if len(msgs) > 2 else None
    parts = user.split("<image>")
    images = example.get("images", [])
    if len(parts) - 1 != len(images):
        raise ValueError(f"{len(parts) - 1} <image> placeholders but {len(images)} images")
    content = []
    for i, chunk in enumerate(parts):
        if chunk:
            content.append({"type": "text", "text": chunk})
        if i < len(images):
            path = Path(images[i])
            path = path if path.is_absolute() else root / path
            if not path.is_file():
                raise FileNotFoundError(path)
            content.append({"type": "image", "image": path.as_uri(), "max_pixels": max_pixels})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}], target


def _resize(width: int, height: int, max_pixels: int, factor: int) -> tuple[int, int]:
    """Largest size within max_pixels that keeps the aspect ratio, sides multiples of factor."""
    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    w = max(factor, int(width * scale // factor) * factor)
    h = max(factor, int(height * scale // factor) * factor)
    while w * h > max_pixels and (w > factor or h > factor):
        if w >= h:
            w -= factor
        else:
            h -= factor
    return w, h


def _vision(messages_list: list[list[dict]]):
    if BACKEND == "qwen3_vl":
        from qwen_vl_utils import process_vision_info
        images = []
        for messages in messages_list:
            imgs, videos = process_vision_info(messages, image_patch_size=16)
            if videos is not None:
                raise ValueError("Only sampled frames are supported")
            images += imgs or []
        return images or None
    from PIL import Image
    images = []
    for messages in messages_list:
        for msg in messages:
            content = msg["content"] if isinstance(msg["content"], list) else []
            for item in content:
                if item.get("type") != "image":
                    continue
                uri = item["image"]
                path = unquote(urlparse(uri).path) if uri.startswith("file:") else uri
                img = Image.open(path).convert("RGB")
                img = img.resize(_resize(*img.size, int(item.get("max_pixels", 384 * 384)), _FACTOR),
                                 Image.BICUBIC)
                images.append(img)
    return images or None


def _prompt(processor, messages: list[dict]) -> str:
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ANSWER_PREFIX


def _batch_inputs(processor, messages_list: list[list[dict]], suffix: str = ""):
    texts = [_prompt(processor, m) + suffix for m in messages_list]
    kwargs = {"text": texts, "return_tensors": "pt", "do_resize": False, "padding": True}
    images = _vision(messages_list)
    if images:
        kwargs["images"] = images
    return processor(**kwargs)


def encode_train(processor, messages: list[dict], target: str, device):
    """Inputs with labels; loss only on the assistant answer (+ end-of-turn)."""
    prompt = _prompt(processor, messages)
    kwargs = {"text": [prompt + target + END_OF_TURN], "return_tensors": "pt", "do_resize": False}
    images = _vision([messages])
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    target_ids = processor.tokenizer(target + END_OF_TURN, add_special_tokens=False)["input_ids"]
    prompt_len = inputs["input_ids"].shape[1] - len(target_ids)
    if inputs["input_ids"][0, prompt_len:].tolist() != target_ids:
        prompt_len = processor(**dict(kwargs, text=[prompt]))["input_ids"].shape[1]
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    inputs["labels"] = labels
    return inputs.to(device), inputs["input_ids"].shape[1]


def _oom_split(fn):
    """Run fn on the batch; on CUDA OOM free the cache and retry on the two halves (down to batch size 1)."""
    def wrapper(model, processor, messages_list, *args, **kwargs):
        try:
            return fn(model, processor, messages_list, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(messages_list) == 1:
                raise
        torch.cuda.empty_cache()
        half = (len(messages_list) + 1) // 2
        return (wrapper(model, processor, messages_list[:half], *args, **kwargs)
                + wrapper(model, processor, messages_list[half:], *args, **kwargs))
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


@_oom_split
def generate(model, processor, messages_list: list[list[dict]], max_new_tokens: int, temperature: float = 0.0,
             top_p: float = 1.0) -> list[str]:
    inputs = _batch_inputs(processor, messages_list, suffix=GEN_PREFIX).to(model_device(model))
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    del inputs, out
    return [GEN_PREFIX + t for t in texts]


def option_continuation(tokenizer, prefix: str, options: tuple[str, str]) -> tuple[list[int], list[int]]:
    """Shared continuation ids up to where the options diverge, and the diverging token of each option."""
    seqs = [tokenizer.encode(prefix + o, add_special_tokens=False) for o in options]
    n = 0
    while n < min(map(len, seqs)) and seqs[0][n] == seqs[1][n]:
        n += 1
    if n >= min(map(len, seqs)):
        raise ValueError(f"options {options} do not diverge after prefix {prefix!r}")
    return seqs[0][:n], [s[n] for s in seqs]


@_oom_split
def option_probs(model, processor, messages_list: list[list[dict]], prefix: str,
                 options: tuple[str, str]) -> list[float]:
    """p(options[0]) vs options[1] as the next answer after `prefix` (e.g. '{"label": "'), per conversation.

    The shared answer prefix is appended as TEXT, so the processor builds every per-token tensor
    (attention mask, multimodal token types, positions) itself; we then check that the prompt really
    ends with the expected prefix token ids before reading the next-token probabilities.
    """
    shared, ids = option_continuation(processor.tokenizer, prefix, options)
    suffix = processor.tokenizer.decode(shared)
    inputs = _batch_inputs(processor, messages_list, suffix=suffix)
    if shared and inputs["input_ids"][:, -len(shared):].tolist() != [shared] * inputs["input_ids"].shape[0]:
        raise ValueError(f"answer prefix {suffix!r} was tokenized differently inside the prompt")
    inputs = inputs.to(model_device(model))
    with torch.inference_mode():
        try:
            logits = model(**inputs, logits_to_keep=1).logits[:, -1]
        except TypeError:
            logits = model(**inputs).logits[:, -1]
    probs = torch.softmax(logits[:, ids].float(), dim=-1)[:, 0].tolist()
    del inputs, logits
    return [round(p, 6) for p in probs]


def causal_lm_loss(model, inputs):
    """Mean token loss on the answer only, computing logits just for the answer positions.

    Full-sequence logits (about 10k tokens x 152k vocabulary) dominate memory for long multimodal
    prompts; the answer is the last part of the sequence, so only its positions are needed.
    """
    import torch.nn.functional as F
    labels = inputs.pop("labels")
    n = int((labels[0] != -100).sum())
    try:
        logits = model(**inputs, logits_to_keep=n + 1).logits[:, :-1]
    except TypeError:
        logits = model(**inputs).logits[:, -(n + 1):-1]
    target = labels[:, -n:]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
