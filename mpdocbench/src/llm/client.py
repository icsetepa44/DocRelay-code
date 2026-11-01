"""
LLM client — ported from old_pipeline/llm_client.py.

Backends: openrouter (OpenAI-compatible, vision), ollama, hf. v1 uses openrouter.
The OpenRouter path sends the page image as a base64 data-URL `image_url` part and
requests `response_format={"type":"json_object"}`.

Enhancements over the original:
  - per-request timeout,
  - bounded retry with linear backoff on transient errors,
  - one-time drop of response_format if the model rejects it (kept from the original),
  - configurable image mime (png / jpeg).

`chat_completion` returns raw text; parsing/validation is the caller's job
(see structured_output.py).
"""

from __future__ import annotations

import base64
import gc
import os
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.core.exceptions import LLMError
from src.llm import usage

BackendName = Literal["ollama", "hf", "huggingface", "openrouter"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class LLMRequestConfig:
    backend: BackendName
    model: str
    temperature: float = 0.2
    top_p: float = 0.1
    num_ctx: Optional[int] = None
    max_new_tokens: int = 4096
    think: bool = False
    response_format: str = "json"     # "json" -> request json_object where supported
    timeout: float = 120.0
    max_retries: int = 3
    retry_backoff: float = 2.0
    image_format: str = "png"         # "png" | "jpeg"
    provider: Optional[Dict[str, Any]] = None  # OpenRouter extra_body.provider routing


# ============================================================
# IMAGE HELPERS
# ============================================================

def _pil_to_base64(image: Any, image_format: str = "png") -> str:
    fmt = "JPEG" if str(image_format).lower() in ("jpg", "jpeg") else "PNG"
    buffer = BytesIO()
    if fmt == "JPEG":
        image = image.convert("RGB")
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _normalize_pil_images(images: Optional[List[Any]]) -> List[Any]:
    if not images:
        return []
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image input requires Pillow. Run: pip install pillow") from exc

    normalized = []
    for item in images:
        if item is None:
            continue
        if isinstance(item, (str, os.PathLike)):
            normalized.append(Image.open(item).convert("RGB"))
            continue
        if hasattr(item, "convert"):
            normalized.append(item.convert("RGB"))
            continue
        raise TypeError(f"Unsupported image input type: {type(item)}. Use PIL.Image or path.")
    return normalized


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    config: LLMRequestConfig,
    images: Optional[List[Any]] = None,
) -> str:
    """Send system+user (+ optional images) to the selected backend; return text."""
    backend = str(config.backend).strip().lower()

    if backend == "ollama":
        return _chat_ollama(system_prompt=system_prompt, user_prompt=user_prompt,
                            config=config, images=images)
    if backend in {"hf", "huggingface"}:
        return _chat_huggingface(system_prompt=system_prompt, user_prompt=user_prompt,
                                 config=config, images=images)
    if backend == "openrouter":
        return _chat_openrouter(system_prompt=system_prompt, user_prompt=user_prompt,
                                config=config, images=images)

    raise ValueError(
        f"Unsupported LLM backend: {config.backend}. Use 'ollama', 'hf', or 'openrouter'."
    )


# ============================================================
# OPENROUTER BACKEND
# ============================================================

def _looks_like_format_issue(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("response_format" in msg) or ("json_object" in msg) or ("json mode" in msg)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    for needle in ("rate limit", "429", "timeout", "timed out", "temporarily",
                   "overloaded", "502", "503", "504", "connection"):
        if needle in msg:
            return True
    return False


def _retry_delay(exc: Exception, attempt: int, base: float, cap: float = 30.0) -> float:
    """Seconds to wait before retry `attempt` (1-indexed).

    Honours the server's Retry-After header on rate limits when present; otherwise uses
    exponential backoff with jitter (base*2^(attempt-1), capped) so concurrent workers
    don't retry in lockstep and hammer the provider.
    """
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            ra = resp.headers.get("retry-after")
            if ra is not None:
                return min(float(ra), cap)
        except (ValueError, TypeError, AttributeError):
            pass
    return min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, base)


def _chat_openrouter(
    *,
    system_prompt: str,
    user_prompt: str,
    config: LLMRequestConfig,
    images: Optional[List[Any]] = None,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenRouter backend selected, but 'openai' is not installed. Run: pip install openai"
        ) from exc

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter backend selected, but OPENROUTER_API_KEY is not set. "
            "Add it to a .env file in the project root or the environment."
        )

    # max_retries=0 so our loop governs retries (avoid double backoff).
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key,
                    timeout=config.timeout, max_retries=0)

    pil_images = _normalize_pil_images(images)
    if pil_images:
        mime = "jpeg" if str(config.image_format).lower() in ("jpg", "jpeg") else "png"
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for img in pil_images:
            data_url = f"data:image/{mime};base64,{_pil_to_base64(img, config.image_format)}"
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_message: Dict[str, Any] = {"role": "user", "content": user_content}
    else:
        user_message = {"role": "user", "content": user_prompt}

    request_kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "system", "content": system_prompt}, user_message],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_new_tokens,
    }

    use_json = config.response_format == "json"
    if use_json:
        request_kwargs["response_format"] = {"type": "json_object"}

    # Usage accounting: OpenRouter returns the provider-reported cost in `usage` (logged by src/llm/usage.py).
    request_kwargs["extra_body"] = {"usage": {"include": True}}
    # OpenRouter provider routing (e.g. ignore Alibaba's Bangla input-filter).
    if config.provider:
        request_kwargs["extra_body"]["provider"] = config.provider

    dropped_format = False
    attempt = 0
    last_exc: Optional[Exception] = None

    while True:
        t0 = time.perf_counter()
        recorded = False
        try:
            response = client.chat.completions.create(**request_kwargs)
            usage.record(model=config.model, status="ok", latency_s=round(time.perf_counter() - t0, 3),
                         **usage.from_response(response))
            recorded = True
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 (surface as LLMError below)
            if not recorded:
                usage.record(model=config.model, status="error", latency_s=round(time.perf_counter() - t0, 3),
                             error=str(exc)[:300])
            last_exc = exc
            # One-time: some models reject response_format -> drop it and retry immediately.
            if use_json and not dropped_format and request_kwargs.get("response_format") \
                    and _looks_like_format_issue(exc):
                request_kwargs.pop("response_format", None)
                dropped_format = True
                continue
            attempt += 1
            if attempt <= config.max_retries and _is_transient(exc):
                time.sleep(_retry_delay(exc, attempt, config.retry_backoff))
                continue
            raise LLMError(
                f"OpenRouter call failed for model '{config.model}' after {attempt} "
                f"retr{'y' if attempt == 1 else 'ies'}: {exc}"
            ) from last_exc


# ============================================================
# OLLAMA BACKEND (optional; unused in v1)
# ============================================================

def _chat_ollama(*, system_prompt, user_prompt, config, images=None):
    try:
        import ollama
    except ImportError as exc:
        raise ImportError("Ollama backend selected, but 'ollama' is not installed.") from exc

    options: Dict[str, Any] = {"temperature": config.temperature, "top_p": config.top_p}
    if config.num_ctx is not None:
        options["num_ctx"] = config.num_ctx

    if config.response_format in (None, "", "json"):
        format_param: Any = config.response_format
    else:
        try:
            import json
            parsed = json.loads(config.response_format)
            format_param = parsed if isinstance(parsed, dict) else None
        except Exception:
            format_param = None

    pil_images = _normalize_pil_images(images)
    user_message: Dict[str, Any] = {"role": "user", "content": user_prompt}
    if pil_images:
        user_message["images"] = [_pil_to_base64(img, "png") for img in pil_images]

    response = ollama.chat(
        model=config.model,
        messages=[{"role": "system", "content": system_prompt}, user_message],
        format=format_param,
        think=config.think,
        options=options,
    )
    return response["message"]["content"]


# ============================================================
# HUGGINGFACE BACKEND (optional; unused in v1)
# ============================================================

@dataclass
class _HFBundle:
    processor: Any
    model: Any


def _from_pretrained_with_dtype(model_cls: Any, model_name: str, token: Optional[str]):
    try:
        model = model_cls.from_pretrained(model_name, dtype="auto", device_map="auto", token=token)
    except TypeError:
        model = model_cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto", token=token)
    model.eval()
    return model


@lru_cache(maxsize=4)
def _load_hf_bundle(model_name: str, vision: bool = False) -> _HFBundle:
    try:
        import transformers
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "HuggingFace backend selected, but deps are missing. "
            "pip install -U transformers torch accelerate pillow"
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    processor = AutoProcessor.from_pretrained(model_name, token=token)

    if vision:
        last_error = None
        for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                           "AutoModelForMultimodalLM", "AutoModelForCausalLM"):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                model = _from_pretrained_with_dtype(model_cls, model_name, token)
                return _HFBundle(processor=processor, model=model)
            except Exception as e:  # noqa: BLE001
                last_error = e
        raise RuntimeError(f"Could not load HF vision model {model_name}. Last error: {last_error}")

    model = _from_pretrained_with_dtype(AutoModelForCausalLM, model_name, token)
    return _HFBundle(processor=processor, model=model)


def _model_input_device(model: Any):
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if isinstance(inputs, dict):
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs.to(device) if hasattr(inputs, "to") else inputs


def _build_hf_messages(system_prompt, user_prompt, pil_images):
    if pil_images:
        user_content = [{"type": "image", "image": img} for img in pil_images]
        user_content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _chat_huggingface(*, system_prompt, user_prompt, config, images=None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("HuggingFace backend selected, but torch is not installed.") from exc

    pil_images = _normalize_pil_images(images)
    vision = len(pil_images) > 0
    bundle = _load_hf_bundle(config.model, vision=vision)
    processor, model = bundle.processor, bundle.model

    final_user_prompt = user_prompt
    if config.response_format == "json":
        final_user_prompt = (
            user_prompt + "\n\nReturn exactly one valid JSON object only. "
            "No markdown. No explanation. No text before or after JSON."
        )

    messages = _build_hf_messages(system_prompt, final_user_prompt, pil_images)
    device = _model_input_device(model)

    try:
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=config.think)
        except TypeError:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, device)
        input_len = inputs["input_ids"].shape[-1]
    except Exception:
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=config.think)
        except TypeError:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if pil_images:
            inputs = processor(text=text, images=pil_images, return_tensors="pt")
        else:
            inputs = processor(text=text, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, device)
        input_len = inputs["input_ids"].shape[-1]

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
    }
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["pad_token_id"] = eos_token_id
    if config.temperature > 0:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    outputs = generated = None
    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        generated = outputs[0][input_len:]
        raw_response = processor.decode(generated, skip_special_tokens=False)
        parsed = _try_parse_gemma_response(processor, raw_response)
        if parsed:
            return parsed
        clean_response = processor.decode(generated, skip_special_tokens=True)
        return _strip_thinking_blocks(clean_response).strip()
    finally:
        for name in ("inputs", "outputs", "generated"):
            try:
                del name
            except Exception:
                pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _try_parse_gemma_response(processor: Any, raw_response: str) -> str:
    parse_response = getattr(processor, "parse_response", None)
    if not callable(parse_response):
        return ""
    try:
        parsed = parse_response(raw_response)
    except Exception:
        return ""
    if isinstance(parsed, str):
        return _strip_thinking_blocks(parsed).strip()
    if isinstance(parsed, dict):
        for key in ("answer", "content", "response", "text"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return _strip_thinking_blocks(value).strip()
        import json
        return json.dumps(parsed, ensure_ascii=False)
    return ""


def _strip_thinking_blocks(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text
