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

# src/core/config.py -> parents[0]=core, [1]=src, [2]=mpdocbench (this package); its parent = the repository root
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = PACKAGE_ROOT.parent

CONFIG_DIR: Path = PACKAGE_ROOT / "config"
PROMPTS_DIR: Path = PACKAGE_ROOT / "prompts"
SCHEMAS_DIR: Path = PACKAGE_ROOT / "schemas"
SKILLS_DIR: Path = PACKAGE_ROOT / "skills"
DATA_DIR: Path = PACKAGE_ROOT / "data"            # generated working files (page images per document)
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
RUNS_DIR: Path = PACKAGE_ROOT / "runs"

# MPDocBench (not shipped with the code; see data/README.md):
#   data/mpdocbench/repo/MPDocBench/  <- the official MPDocBench code + gold (MPDocBench.json); its metric code
#                                        (TEDS, HeadTEDS, matching, CDM) is reused unchanged by src/eval/mpdocbench.py
#   data/mpdocbench/images/<doc>/page_N.jpg   <- the benchmark's page images (OCR input)
#   data/mpdocbench/ocr/<doc>/                <- Chandra OCR of those images (pipeline input; scripts/ocr_images.py)
MPDOCBENCH_ROOT: Path = REPO_ROOT / "data" / "mpdocbench"
MPDOCBENCH_REPO: Path = MPDOCBENCH_ROOT / "repo" / "MPDocBench"
MPDOCBENCH_GOLD: Path = MPDOCBENCH_REPO / "MPDocBench.json"
MPDOCBENCH_IMAGES: Path = MPDOCBENCH_ROOT / "images"
# OCR input of the pipeline (and the evaluator's figure layouts). Optional override: pipeline.yaml
# inputs.ocr_root (re-points this global below).
BENCHMARK_OCR_PARENT: Path = MPDOCBENCH_ROOT / "ocr"


# ------------------------------------------------------------------
# .env (OPENROUTER_API_KEY) — the repository-root file
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

# Optional OCR-root override: pipeline.yaml inputs.ocr_root, relative to PACKAGE_ROOT (or absolute);
# unset -> data/mpdocbench/ocr/.
_ocr_root = (PIPELINE.get("inputs", {}) or {}).get("ocr_root")
if _ocr_root:
    BENCHMARK_OCR_PARENT = (PACKAGE_ROOT / str(_ocr_root)).resolve()
MODELS: Dict[str, Any] = _read_yaml(CONFIG_DIR / "models.yaml")
_BLOCK_TYPES: Dict[str, Any] = _read_json(CONFIG_DIR / "block_types.json")
DOMAIN_PROFILES: Dict[str, Any] = _read_yaml(CONFIG_DIR / "domain_profiles.yaml")  # per-domain prompt guidance


# ------------------------------------------------------------------
# Flattened globals (the values stage code actually reads)
# ------------------------------------------------------------------

ALLOWED_BLOCK_TYPES: List[str] = list(_BLOCK_TYPES["allowed_block_types"])

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

# Track 2 — page-drop guard. The detector (page_underproduction.json) ALWAYS runs; PAGE_UNDERPRODUCTION_RETRY
# re-extracts a flagged page with a corrective nudge (default OFF so a baseline run stays a pure model
# comparison). A page is flagged when it looks like an extractor bail-out: regions >= min_regions AND
# blocks <= max(2, round(block_ratio*regions)) AND coverage < min_coverage. Read live (attribute access)
# so a runner can override at runtime.
_underprod = PIPELINE.get("page_underproduction", {})
PAGE_UNDERPRODUCTION_RETRY: bool = bool(_underprod.get("retry", False))
PAGE_UNDERPRODUCTION_MIN_REGIONS: int = int(_underprod.get("min_regions", 5))
PAGE_UNDERPRODUCTION_BLOCK_RATIO: float = float(_underprod.get("block_ratio", 0.25))
PAGE_UNDERPRODUCTION_MIN_COVERAGE: float = float(_underprod.get("min_coverage", 0.3))
PAGE_UNDERPRODUCTION_MAX_RETRIES: int = int(_underprod.get("max_retries", 1))

# T1/T4 schedule over-segmentation safety net. The detector (over_segmentation.json) always runs;
# SCHEDULE_COLLAPSE re-nests a flagged run of uniform numbered top-level blocks (never fabricates a
# table). Read live by the orchestrator (attribute access) so a runner can override at runtime.
_sched = PIPELINE.get("schedule_over_segmentation", {})
SCHEDULE_COLLAPSE: bool = bool(_sched.get("collapse", False))
SCHEDULE_MIN_RUN: int = int(_sched.get("min_run", 4))
SCHEDULE_MAX_ITEM_CHARS: int = int(_sched.get("max_item_chars", 160))

# Critic-corrector: True = draft -> SuperValidator ->
# FinalDecider (on a correction-triggering issue); False = legacy report-only validator + continuation_judge.
USE_SUPERVALIDATOR: bool = bool(PIPELINE.get("use_supervalidator", True))
# Extractor only (an ablation setting; default false): on the critic-corrector path, skip the SuperValidator AND
# FinalDecider calls; the draft goes straight to the deterministic continuation step. Read live (attribute access).
SKIP_VERIFIER_AND_DECIDER: bool = bool(PIPELINE.get("skip_verifier_and_decider", False))

# Eval knobs. EQUATION_METRIC selects the Formula scorer: "cdm" = paper-faithful Character Detection
# Matching (reuses the vendored paper code via src/eval/cdm_formula.py; needs the WSL toolchain — falls
# back to "edit" with an honest method label when unavailable); "edit" = per-equation LaTeX edit distance
# (our toolchain-free diagnostic, also always reported as `equation_diag`). Read live so a runner can
# override at runtime.
_eval = PIPELINE.get("eval", {})
EQUATION_METRIC: str = str(_eval.get("equation_metric", "cdm")).strip().lower()
if EQUATION_METRIC not in ("cdm", "edit"):
    EQUATION_METRIC = "cdm"
EQUATION_CDM: bool = EQUATION_METRIC == "cdm"
EQUATION_CDM_KEEP_ARTIFACTS: bool = bool(_eval.get("equation_cdm_keep_artifacts", False))
# Optional override for the CDM render-scratch dir. The vendored CDM renderer shells out to xelatex /
# ImageMagick with UNQUOTED paths, so a scratch dir under a path containing spaces (e.g. this repo's
# "…/My Documents/…") makes every render silently fail -> F1 0. Point this at a space-free
# location (e.g. "/tmp/mpdocbench_cdm_work" under WSL) to run CDM from such a checkout. None -> default under
# the run dir (`<run>/_mpdocbench/_cdm_work`). Read live so a runner can override at runtime.
_cdm_wd = _eval.get("equation_cdm_work_dir") or None
EQUATION_CDM_WORK_DIR: Optional[str] = str(_cdm_wd) if _cdm_wd else None

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

# OpenRouter provider routing sent as extra_body.provider on every call.
# `use_provider_order` gates the shared `order` (provider preference); `use_quantizations` gates the precision
# pin, which is resolved PER-MODEL in provider_for() because each model exposes different quantizations on
# OpenRouter (gemma has bf16 hosts, deepseek only fp8/fp4). Base keys (allow_fallbacks, data_collection) are
# always kept. Empty result -> None.
_openrouter = PIPELINE.get("openrouter", {})
USE_QUANTIZATIONS: bool = bool(_openrouter.get("use_quantizations", True))
_provider_base: Dict[str, Any] = dict(_openrouter.get("provider") or {})
_provider_base.pop("quantizations", None)  # per-model; ignore any legacy global value
if not bool(_openrouter.get("use_provider_order", False)):
    _provider_base.pop("order", None)
OPENROUTER_PROVIDER_BASE: Optional[Dict[str, Any]] = _provider_base or None


def provider_for(model: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """OpenRouter extra_body.provider for a resolved model dict: shared base routing plus, when
    use_quantizations is on, that model's own precision pin (provider.quantizations)."""
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
    Resolve {"backend", "model", "vision", "quantizations"} for a pipeline role, inheriting `default`.
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


# ------------------------------------------------------------------
# Per-domain resolution
# ------------------------------------------------------------------

def domain_for(doc_id: str) -> str:
    """Document domain for a stem: the stem prefix (MPDocBench stems are ``<domain>_<lang>_<hash>``);
    ``""`` when unknown."""
    if not doc_id:
        return ""
    return doc_id.split("_")[0]


def domain_profile(domain: Optional[str]) -> Dict[str, Any]:
    """Resolve a per-domain profile -> ``{guidance, pin_skills, weight_skills}``, the named
    domain merged over ``defaults``. None/unknown domain -> defaults."""
    base = dict(DOMAIN_PROFILES.get("defaults") or {})
    if domain:
        override = (DOMAIN_PROFILES.get("domains") or {}).get(domain) or {}
        base.update({k: v for k, v in override.items() if v is not None})
    return {
        "guidance": str(base.get("guidance") or ""),
        "pin_skills": list(base.get("pin_skills") or []),
        "weight_skills": dict(base.get("weight_skills") or {}),
    }
