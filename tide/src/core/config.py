from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = PACKAGE_ROOT.parent

CONFIG_DIR: Path = PACKAGE_ROOT / "config"
PROMPTS_DIR: Path = PACKAGE_ROOT / "prompts"
SCHEMAS_DIR: Path = PACKAGE_ROOT / "schemas"
SKILLS_DIR: Path = PACKAGE_ROOT / "skills"
DATA_DIR: Path = PACKAGE_ROOT / "data"
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
RUNS_DIR: Path = PACKAGE_ROOT / "runs"
EVAL_SUBDIR: str = "_eval"

CORPUS_ROOT: Path = REPO_ROOT / "data" / "tide"
BENCHMARK_GOLD_ROOT: Path = CORPUS_ROOT / "json"
BENCHMARK_PDF_ROOT: Path = CORPUS_ROOT / "pdf"
BENCHMARK_OCR_PARENT: Path = CORPUS_ROOT / "ocr"
MPDOCBENCH_REPO: Path = REPO_ROOT / "data" / "mpdocbench" / "repo" / "MPDocBench"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env)


_load_env()


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


PIPELINE: Dict[str, Any] = _read_yaml(CONFIG_DIR / "pipeline.yaml")
MODELS: Dict[str, Any] = _read_yaml(CONFIG_DIR / "models.yaml")
_BLOCK_TYPES: Dict[str, Any] = _read_json(CONFIG_DIR / "block_types.json")


ALLOWED_BLOCK_TYPES: List[str] = list(_BLOCK_TYPES["allowed_block_types"])

_skill = PIPELINE.get("skill_retrieval", {})
MAIN_TOP_K: int = int(_skill.get("main_top_k", 2))

RETRIEVAL_SCORER: str = str(_skill.get("method", "embedding")).strip().lower()
EMBEDDING_MODEL: str = str(_skill.get("embedding_model", "intfloat/multilingual-e5-small"))
EMBEDDING_QUERY_PREFIX: str = str(_skill.get("embedding_query_prefix", "query: "))
EMBEDDING_DOC_PREFIX: str = str(_skill.get("embedding_doc_prefix", "passage: "))

RETRIEVAL_MODE: str = str(PIPELINE.get("retrieval", {}).get("mode", "region")).strip().lower()

USE_JUDGE_WHEN_UNCERTAIN: bool = bool(
    PIPELINE.get("continuation", {}).get("use_judge_when_uncertain", True)
)
JUDGE_ON_UNCONFIRMED_MAIN_CLOSE: bool = bool(
    PIPELINE.get("continuation", {}).get("judge_on_unconfirmed_main_close", True)
)
CARRY_OPEN_TEXT_MAX_CHARS: int = int(
    PIPELINE.get("continuation", {}).get("carry_open_text_max_chars", 4000)
)
CARRY_COLLISION_GUARD: bool = bool(
    PIPELINE.get("continuation", {}).get("carry_collision_guard", False)
)
UNRESOLVED_POLICY: str = str(
    PIPELINE.get("failure_handling", {}).get("unresolved_policy", "coerce")
)
MIN_CHARS_NONTRIVIAL_PAGE: int = int(
    PIPELINE.get("failure_handling", {}).get("min_chars_nontrivial_page", 200)
)
RETRY_EMPTY_PAGE: bool = bool(
    PIPELINE.get("failure_handling", {}).get("retry_empty_page", True)
)
MAX_EMPTY_PAGE_RATIO: float = float(
    PIPELINE.get("failure_handling", {}).get("max_empty_page_ratio", 0.15)
)

_sched = PIPELINE.get("schedule_over_segmentation", {})
SCHEDULE_COLLAPSE: bool = bool(_sched.get("collapse", False))
SCHEDULE_MIN_RUN: int = int(_sched.get("min_run", 4))
SCHEDULE_MAX_ITEM_CHARS: int = int(_sched.get("max_item_chars", 160))
USE_SUPERVALIDATOR: bool = bool(PIPELINE.get("use_supervalidator", True))

_inputs = PIPELINE.get("inputs", {})
OCR_INPUT_FORMAT: str = str(_inputs.get("ocr_format", "mixed")).strip().lower()
if OCR_INPUT_FORMAT not in ("markdown", "html", "both", "mixed"):
    OCR_INPUT_FORMAT = "both"

_render = PIPELINE.get("rendering", {})
RENDER_DPI: int = int(_render.get("dpi", 300))
MODEL_IMAGE_MAX_LONG_SIDE: Optional[int] = _render.get("model_image_max_long_side", 1024)
POPPLER_PATH: Optional[str] = _render.get("poppler_path") or None

_openrouter = PIPELINE.get("openrouter", {})
USE_QUANTIZATIONS: bool = bool(_openrouter.get("use_quantizations", True))
_provider_base: Dict[str, Any] = dict(_openrouter.get("provider") or {})
_provider_base.pop("quantizations", None)
if not bool(_openrouter.get("use_provider_order", False)):
    _provider_base.pop("order", None)
OPENROUTER_PROVIDER_BASE: Optional[Dict[str, Any]] = _provider_base or None


def provider_for(model: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    prov: Dict[str, Any] = dict(OPENROUTER_PROVIDER_BASE or {})
    if USE_QUANTIZATIONS and model:
        q = model.get("quantizations")
        if q:
            prov["quantizations"] = list(q)
    return prov or None

_req = PIPELINE.get("request", {})
TEMPERATURE: float = float(_req.get("temperature", 0.0))
TOP_P: float = float(_req.get("top_p", 0.1))
MAX_NEW_TOKENS: int = int(_req.get("max_new_tokens", 16384))
TIMEOUT_SECONDS: float = float(_req.get("timeout_seconds", 600))
MAX_RETRIES: int = int(_req.get("max_retries", 3))
RETRY_BACKOFF_SECONDS: float = float(_req.get("retry_backoff_seconds", 2.0))


def _resolve_model(entry: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(entry, str):
        entry = {"model": entry}
    merged = {**default, **(entry or {})}
    if "backend" not in merged or "model" not in merged:
        raise KeyError("Model entry is missing backend/model and no complete default was provided.")
    q = merged.get("quantizations")
    return {
        "backend": str(merged["backend"]),
        "model": str(merged["model"]),
        "vision": bool(merged.get("vision", False)),
        "quantizations": list(q) if q else None,
    }


def model_for(role: str) -> Dict[str, Any]:
    default = dict(MODELS.get("default", {}))
    role_cfg = dict((MODELS.get("roles", {}) or {}).get(role, {}) or {})
    if "models" in role_cfg and role_cfg["models"]:
        return _resolve_model(role_cfg["models"][0], default)
    return _resolve_model(role_cfg, default)


def extractor_models() -> List[Dict[str, Any]]:
    default = dict(MODELS.get("default", {}))
    role_cfg = dict((MODELS.get("roles", {}) or {}).get("extractor", {}) or {})
    if "models" in role_cfg and role_cfg["models"]:
        return [_resolve_model(e, default) for e in role_cfg["models"]]
    return [model_for("extractor")]


def model_label(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def has_together_key() -> bool:
    return bool(os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY"))


def has_llm_key() -> bool:
    return has_openrouter_key() or has_together_key()
