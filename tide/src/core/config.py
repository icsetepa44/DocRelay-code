"""
Central configuration, exposed as module-level GLOBAL variables (no CLI/argparse).

Loads:
  config/pipeline.yaml     -> pipeline knobs (spec §7 + v1 rendering/request blocks)
  config/models.yaml       -> per-role backend + model id
  config/block_types.json  -> the allowed block-type whitelist (spec §2)

and the repo-root `.env` (for OPENROUTER_API_KEY).

Usage:
    from src.core import config
    config.MAIN_TOP_K
    config.ALLOWED_BLOCK_TYPES
    config.model_for("extractor")   # -> {"backend": ..., "model": ...}

To change behaviour, edit the YAML/JSON files or set the globals here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ------------------------------------------------------------------
# Paths (resolved from this file, independent of the current directory)
# ------------------------------------------------------------------

# src/core/config.py -> parents[0]=core, [1]=src, [2]=tide (this package); its parent = the repository root
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = PACKAGE_ROOT.parent

CONFIG_DIR: Path = PACKAGE_ROOT / "config"
PROMPTS_DIR: Path = PACKAGE_ROOT / "prompts"
SCHEMAS_DIR: Path = PACKAGE_ROOT / "schemas"
SKILLS_DIR: Path = PACKAGE_ROOT / "skills"
DATA_DIR: Path = PACKAGE_ROOT / "data"            # generated working files (rendered page images)
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
RUNS_DIR: Path = PACKAGE_ROOT / "runs"
# Eval-output folder name inside a run (runs/<run>/<EVAL_SUBDIR>/metrics.json).
EVAL_SUBDIR: str = "_eval"

# The TIDE corpus (not shipped with the code; see data/README.md), one folder per kind:
#   data/tide/pdf/<name>.pdf          <- source PDFs (page-image rendering + OCR input)
#   data/tide/ocr/<name>/{markdown,html,layout,<name>.json}  <- Chandra OCR (pipeline input)
#   data/tide/json/<name>_output.json <- gold annotations (evaluation only)
# The discovery functions (chandra_adapter._build_ocr_cache/ocr_dir_for/_source_pdf,
# document_store.resolve_pdf_path, evaluate.gold_path_for) all read these globals, so pointing
# them here is all that's needed.
CORPUS_ROOT: Path = REPO_ROOT / "data" / "tide"
BENCHMARK_GOLD_ROOT: Path = CORPUS_ROOT / "json"   # gold_path_for -> <gold_root>/<name>_output.json
BENCHMARK_PDF_ROOT: Path = CORPUS_ROOT / "pdf"     # resolve_pdf_path / _source_pdf
BENCHMARK_OCR_PARENT: Path = CORPUS_ROOT / "ocr"   # _build_ocr_cache / ocr_dir_for direct lookup


# ------------------------------------------------------------------
# .env (OPENROUTER_API_KEY / TOGETHER_API_KEY)
# ------------------------------------------------------------------

def _load_env() -> None:
    """Read the repository-root .env (copy of .env.example). Variables already set in the environment win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root_env = REPO_ROOT / ".env"
    if root_env.exists():
        load_dotenv(root_env)


_load_env()


# ------------------------------------------------------------------
# Load config files
# ------------------------------------------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


PIPELINE: Dict[str, Any] = _read_yaml(CONFIG_DIR / "pipeline.yaml")
MODELS: Dict[str, Any] = _read_yaml(CONFIG_DIR / "models.yaml")
_BLOCK_TYPES: Dict[str, Any] = _read_json(CONFIG_DIR / "block_types.json")


# ------------------------------------------------------------------
# Flattened globals (the values stage code actually reads)
# ------------------------------------------------------------------

ALLOWED_BLOCK_TYPES: List[str] = list(_BLOCK_TYPES["allowed_block_types"])

# The PIPELINE.get(key, DEFAULT) fallbacks below are SAFETY NETS that match config/pipeline.yaml (the
# single source of truth); they apply only if a key is missing from the YAML.
_skill = PIPELINE.get("skill_retrieval", {})
MAIN_TOP_K: int = int(_skill.get("main_top_k", 2))

# Skill-retrieval scorer selector + embedding backend settings.
# Read live by skill_retriever (attribute access) so a runner can override at runtime.
RETRIEVAL_SCORER: str = str(_skill.get("method", "embedding")).strip().lower()
EMBEDDING_MODEL: str = str(_skill.get("embedding_model", "intfloat/multilingual-e5-small"))
EMBEDDING_QUERY_PREFIX: str = str(_skill.get("embedding_query_prefix", "query: "))
EMBEDDING_DOC_PREFIX: str = str(_skill.get("embedding_doc_prefix", "passage: "))

# "page" (page-level query + page router) or "region" (Region Analyzer + per-region RAG).
RETRIEVAL_MODE: str = str(PIPELINE.get("retrieval", {}).get("mode", "region")).strip().lower()

USE_JUDGE_WHEN_UNCERTAIN: bool = bool(
    PIPELINE.get("continuation", {}).get("use_judge_when_uncertain", True)
)
# B2: also invoke the continuation judge when a main point would CLOSE but the next main
# number is not yet visible on the page (unconfirmed close). Set False to isolate B2.
JUDGE_ON_UNCONFIRMED_MAIN_CLOSE: bool = bool(
    PIPELINE.get("continuation", {}).get("judge_on_unconfirmed_main_close", True)
)
# Carry the actual text of a still-open unit across the page boundary, bounded to this many
# chars (head+tail kept on overflow). 0 -> snippet-only (pre-enrichment behavior).
# Read live by carry_builder (attribute access) so a runner can override at runtime.
CARRY_OPEN_TEXT_MAX_CHARS: int = int(
    PIPELINE.get("continuation", {}).get("carry_open_text_max_chars", 4000)
)
# T11 carry collision guard: require a fresh main to be a plausible successor of an open rule's
# expected_next_main_point before it may close that rule (so a stray shredded field-number can't).
# Read live by the orchestrator + carry_rejection (attribute access) so a runner can override.
CARRY_COLLISION_GUARD: bool = bool(
    PIPELINE.get("continuation", {}).get("carry_collision_guard", False)
)
UNRESOLVED_POLICY: str = str(
    PIPELINE.get("failure_handling", {}).get("unresolved_policy", "coerce")
)
# C2 — reliability / coverage gate (stop losing pages silently).
MIN_CHARS_NONTRIVIAL_PAGE: int = int(
    PIPELINE.get("failure_handling", {}).get("min_chars_nontrivial_page", 200)
)
RETRY_EMPTY_PAGE: bool = bool(
    PIPELINE.get("failure_handling", {}).get("retry_empty_page", True)
)
MAX_EMPTY_PAGE_RATIO: float = float(
    PIPELINE.get("failure_handling", {}).get("max_empty_page_ratio", 0.15)
)

# T1/T4 schedule over-segmentation safety net. The detector (over_segmentation.json) always runs;
# SCHEDULE_COLLAPSE re-nests a flagged run of uniform numbered top-level blocks (never fabricates a
# table). Read live by the orchestrator (attribute access) so a runner can override at runtime.
_sched = PIPELINE.get("schedule_over_segmentation", {})
SCHEDULE_COLLAPSE: bool = bool(_sched.get("collapse", False))
SCHEDULE_MIN_RUN: int = int(_sched.get("min_run", 4))
SCHEDULE_MAX_ITEM_CHARS: int = int(_sched.get("max_item_chars", 160))
USE_SUPERVALIDATOR: bool = bool(PIPELINE.get("use_supervalidator", True))

# Which Chandra OCR format(s) are sent to the LLM stages: "markdown" | "html" | "both" | "mixed".
# "mixed" = markdown prose with tables kept inline as HTML (see stages/_common.mixed_ocr_text).
# Read live by every stage (attribute access) so a runner can override at runtime.
_inputs = PIPELINE.get("inputs", {})
OCR_INPUT_FORMAT: str = str(_inputs.get("ocr_format", "mixed")).strip().lower()
if OCR_INPUT_FORMAT not in ("markdown", "html", "both", "mixed"):
    OCR_INPUT_FORMAT = "both"  # guard: a typo must never silently drop all OCR text

_render = PIPELINE.get("rendering", {})
RENDER_DPI: int = int(_render.get("dpi", 300))
MODEL_IMAGE_MAX_LONG_SIDE: Optional[int] = _render.get("model_image_max_long_side", 1024)
POPPLER_PATH: Optional[str] = _render.get("poppler_path") or None

# OpenRouter provider routing sent as extra_body.provider on every call. `use_provider_order`
# gates the shared `order` (provider preference); `use_quantizations` gates the precision pin.
# The precision pin is resolved PER-MODEL in provider_for() (not global), because each model
# exposes different quantizations on OpenRouter (gemma has bf16 hosts, deepseek only fp8/fp4).
# Base keys (allow_fallbacks, data_collection) are always kept. Empty result -> None.
_openrouter = PIPELINE.get("openrouter", {})
USE_QUANTIZATIONS: bool = bool(_openrouter.get("use_quantizations", True))
_provider_base: Dict[str, Any] = dict(_openrouter.get("provider") or {})
_provider_base.pop("quantizations", None)  # per-model now; ignore any legacy global value
if not bool(_openrouter.get("use_provider_order", False)):
    _provider_base.pop("order", None)
OPENROUTER_PROVIDER_BASE: Optional[Dict[str, Any]] = _provider_base or None


def provider_for(model: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """OpenRouter extra_body.provider for a resolved model dict: shared base routing plus,
    when use_quantizations is on, that model's own precision pin (provider.quantizations).
    Per-model pinning lets each model use its best available precision — e.g. gemma bf16/fp16,
    deepseek-v4.1-flash bf16/fp16/fp8 — instead of one global list that would 404 any model
    lacking a bf16/fp16 host."""
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


# ------------------------------------------------------------------
# Per-role model resolution
# ------------------------------------------------------------------

def _resolve_model(entry: Any, default: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one model entry (str id or dict) into {backend, model, vision, quantizations}."""
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
    """
    Resolve {"backend", "model", "vision"} for a pipeline role, inheriting `default`.
    For a role configured with a `models` LIST (e.g. extractor), the first entry is
    returned as the single-model fallback.
    """
    default = dict(MODELS.get("default", {}))
    role_cfg = dict((MODELS.get("roles", {}) or {}).get(role, {}) or {})
    if "models" in role_cfg and role_cfg["models"]:
        return _resolve_model(role_cfg["models"][0], default)
    return _resolve_model(role_cfg, default)


def extractor_models() -> List[Dict[str, Any]]:
    """List of {backend, model, vision} for the extractor 'one run per model' comparison."""
    default = dict(MODELS.get("default", {}))
    role_cfg = dict((MODELS.get("roles", {}) or {}).get("extractor", {}) or {})
    if "models" in role_cfg and role_cfg["models"]:
        return [_resolve_model(e, default) for e in role_cfg["models"]]
    return [model_for("extractor")]


def model_label(model: str) -> str:
    """Filesystem-safe label for a model id (used in run output paths)."""
    return str(model).replace("/", "_").replace(":", "_")


def has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def has_together_key() -> bool:
    return bool(os.getenv("TOGETHER_API_KEY") or os.getenv("TOGETHERAI_API_KEY"))


def has_llm_key() -> bool:
    return has_openrouter_key() or has_together_key()
