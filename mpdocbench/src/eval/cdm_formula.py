"""Paper-faithful Formula/equation metric — a thin ADAPTER over MPDocBench's OWN code.

The paper's Formula column is **CDM (Character Detection Matching)**: render each LaTeX to an image,
match rendered character bboxes (Hungarian + RANSAC), report F1. It is NOT a standalone scorer — in the
paper's harness formulas are first *aligned* to the gold by a text-edit-distance assignment, then each
matched pair is scored by CDM, and the corpus number is the micro-mean of per-pair F1.

This module reproduces that faithfully by REUSING the paper's real functions (imported from the vendored
`data/mpdocbench/repo/MPDocBench/`), never reimplementing them:

  * pairing  -> ``utils.match.match_gt2pred_simple(..., 'formula', ...)`` — the exact formula-only
                Hungarian LAP over normalized edit distance the paper uses for formulas (pure Python:
                scipy + Levenshtein; no render toolchain). We feed it OUR predicted equation LaTeX + the
                gold equation list, shaped exactly as ``get_gt_pred_lines`` expects.
  * scoring  -> ``metrics.cdm_metric.CDM.evaluate(gt, pred, img_id)`` — the paper's render-and-match F1,
                UNCHANGED. Needs xelatex + ImageMagick 7 + node + the Source Han Sans SC font (available
                under WSL). Missing toolchain -> ``cdm_available()`` is False and the caller falls back to
                the edit-distance diagnostic (``equation_diag``) with an honest ``method`` label.

The only thing this file authors is the adapter glue: build the paper's input dicts from our data, apply
the paper's exact delimiter strip, and micro-aggregate. Deviation from the leaderboard end2end run: the
paper's end2end config aligns formulas via the *mixed reading stream* (``match_gt2pred_quick``); our
structured input maps to the *formula-only* matcher (the md2md variant) — same CDM core + the paper's own
matcher, disclosed as the formula-only alignment variant. And, re-rendered locally, it is the same metric,
not byte-identical leaderboard numbers.
"""

from __future__ import annotations

import importlib
import re
import shutil
import sys
import tempfile
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core import config

# Cached handles to the vendored paper functions (loaded lazily so importing this module — and running the
# pure-Python alignment — never requires the CDM render toolchain).
_MATCHER = None          # (match_gt2pred_simple, normalized_formula)
_CDM_CLASS = None


@lru_cache(maxsize=1)
def _spacefree_repo() -> Path:
    """A path to the vendored MPDocBench repo that contains NO spaces.

    The vendored CDM code shells out (unquoted) to ``node <preprocess_formula.js>`` (and ``xelatex``/``perl``)
    where the script path is derived from the module's own ``__file__`` — i.e. the repo location. This
    checkout may live under a path with spaces (``…/My Documents/…``), so the shell splits the
    path and the tool fails (node: ``Cannot find module '/mnt/c/My'``), which silently degrades the
    paper's KaTeX tokenization to a raw-LaTeX fallback. If the real repo path has no space we use it as-is;
    otherwise we expose it through a space-free symlink (created once) and import the vendored code from
    there, so every ``__file__``-derived shell path is space-free. Pure portability shim: it changes no
    metric behaviour and edits none of the paper's code. Falls back to the real path if symlinks are
    unavailable (e.g. Windows without privilege) — where CDM does not run anyway (``cdm_available()`` is
    False), so the pure-Python matcher still imports fine from the spaces path."""
    repo = Path(config.MPDOCBENCH_REPO)
    if " " not in str(repo):
        return repo
    link = Path(tempfile.gettempdir()) / "mpdocbench_repo_cdm"
    try:
        if link.is_symlink():
            if link.resolve() == repo.resolve():
                return link
            link.unlink()
        elif link.exists():
            return repo  # something real sits there — don't clobber
        link.symlink_to(repo, target_is_directory=True)
        return link
    except OSError:
        return repo


def _load_matcher():
    """Import the paper's real formula matcher from the vendored ``utils`` package (pure Python).

    ``utils`` is a namespace package under the MPDocBench repo; importing ``utils.match`` runs no package
    init and does NOT touch the poisoned ``metrics/__init__``. Deps: scipy, python-Levenshtein, numpy, bs4,
    pylatexenc (all present in the project conda env)."""
    global _MATCHER
    if _MATCHER is None:
        repo_dir = str(_spacefree_repo())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from utils.match import match_gt2pred_simple, normalized_formula  # type: ignore
        _MATCHER = (match_gt2pred_simple, normalized_formula)
    return _MATCHER


def _load_cdm():
    """Import the paper's real ``CDM`` class WITHOUT triggering the poisoned ``metrics/__init__``.

    ``metrics/__init__`` does ``from .cal_metric import *`` which pulls pandas/evaluate/TEDS (and prints).
    ``cdm_metric.py`` uses relative imports (``from .cdm.modules...``), so it can only be imported as a
    submodule of a ``metrics`` package. We pre-register a lightweight stub ``metrics`` package pointing at
    the real dir, so the relative imports resolve but ``cal_metric`` never runs."""
    global _CDM_CLASS
    if _CDM_CLASS is None:
        repo_dir = str(_spacefree_repo())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        metrics_dir = str(_spacefree_repo() / "metrics")
        existing = sys.modules.get("metrics")
        if existing is None or getattr(existing, "__file__", None):
            # Not present, or the REAL package is loaded — install a stub so the poisoned init is skipped.
            stub = types.ModuleType("metrics")
            stub.__path__ = [metrics_dir]  # type: ignore[attr-defined]
            sys.modules["metrics"] = stub
        cdm_metric = importlib.import_module("metrics.cdm_metric")
        _CDM_CLASS = cdm_metric.CDM
    return _CDM_CLASS


@lru_cache(maxsize=1)
def cdm_available() -> bool:
    """True iff the CDM render toolchain is usable: xelatex + ImageMagick 7 (`magick`) + node + a skimage
    whose ``ransac`` still accepts ``random_state`` (the paper calls ``ransac(..., random_state=42)``).

    The skimage check is a deliberate correctness guard, NOT pedantry: ``CDM.evaluate`` swallows every
    exception and returns F1 0, so a too-new skimage (``random_state`` renamed to ``rng`` in 0.22+) would
    make EVERY formula silently score 0 — an all-zeros CDM run masquerading as a real result. Failing the
    gate here instead routes to the honest ``edit`` fallback. Faithful CDM needs Python <=3.11 +
    scikit-image <=0.20 (RandomState(42) seeding); see the CDM notes in the README.

    The Source Han Sans SC font (hard-required by the paper's template) can't be probed cheaply; a missing
    font surfaces as a per-pair render failure -> F1 0, which is the paper's own behavior."""
    if not (shutil.which("xelatex") and shutil.which("magick") and shutil.which("node")):
        return False
    try:
        import inspect
        from skimage.measure import ransac  # noqa: F401
        if "random_state" not in inspect.signature(ransac).parameters:
            return False  # skimage too new: random_state->rng would silently zero every pair
    except Exception:
        return False
    return True


def _strip_delims(s: str, is_pred: bool = False) -> str:
    """The paper's exact pre-CDM delimiter strip (metrics/cal_metric.py:199-211), reproduced verbatim.

    (`str.lstrip("$$")`/`rstrip("$$")` strip a *set* of `$` chars — kept as-is to match the paper byte for
    byte.)"""
    s = str(s)
    if is_pred:
        s = s.split("```latex")[-1].split("```")[0]
    s = s.lstrip("$$").rstrip("$$").strip()
    s = s.lstrip("$").rstrip("$").strip()
    s = re.sub(r'^\\\[|\\\]$', '', s.strip()).strip()
    s = re.sub(r'^\\\(|\\\)$', '', s.strip()).strip()
    return s


def align(gold_equations: List[Dict[str, Any]], pred_latex_list: List[str],
          img_name: str) -> List[Tuple[str, str]]:
    """Align our predicted equations to the gold via the paper's real ``match_gt2pred_simple``.

    Returns the list of ``(gt_latex, pred_latex)`` pairs CDM will score, each already delimiter-stripped.
    Pure Python — no render toolchain needed (unit-tested in the conda env).

    We pre-drop items whose raw OR normalized form is empty so the paper's internal filter can't desync the
    ``gt_lines``/``gt_items`` index pairing — this reproduces exactly which items become samples (the paper
    filters the same empties) while keeping every pred item carrying a ``position`` list as the matcher
    requires. Missing pred -> pair ``(gt, "")`` (CDM scores 0); extra pred (gt_idx==[""]) -> dropped;
    inline equations -> dropped (md2md convention)."""
    match_gt2pred_simple, normalized_formula = _load_matcher()

    gold_items: List[Dict[str, Any]] = []
    for i, e in enumerate(gold_equations):
        latex = str(e.get("latex", ""))
        if latex.strip() and normalized_formula(latex).strip():
            gold_items.append({"category_type": "equation_isolated", "latex": latex,
                               "order": e.get("order", i)})
    pred_items: List[Dict[str, Any]] = []
    for i, p in enumerate(pred_latex_list):
        content = str(p)
        if content.strip() and normalized_formula(content).strip():
            pred_items.append({"category_type": "equation_isolated", "content": content,
                               "position": [i, i]})

    match_list, _ = match_gt2pred_simple(gold_items, pred_items, "formula", img_name)

    pairs: List[Tuple[str, str]] = []
    for m in match_list:
        if m.get("gt_idx") == [""]:                       # extra pred, no gold -> not a formula sample
            continue
        if m.get("gt_category_type") == "equation_inline":  # md2md drops inline equations
            continue
        pairs.append((_strip_delims(m.get("gt", ""), is_pred=False),
                      _strip_delims(m.get("pred", ""), is_pred=True)))
    return pairs


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)) or "doc"


def score(gold_equations: List[Dict[str, Any]], pred_latex_list: List[str],
          img_name: str, work_dir: Path) -> Dict[str, Any]:
    """Paper CDM over the aligned pairs. Returns per-pair F1/precision/recall + n_pairs.

    Each pair is scored by the paper's ``CDM.evaluate`` (unchanged); any render/match failure yields F1 0
    for that pair (the paper's own semantics). Renders go under ``work_dir/<doc>`` (a fresh dir so stale
    caches never leak between re-scores); cleaned unless ``EQUATION_CDM_KEEP_ARTIFACTS``."""
    CDM = _load_cdm()
    pairs = align(gold_equations, pred_latex_list, img_name)

    out_root = Path(work_dir) / _safe(img_name)
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)
    cdm = CDM(output_root=str(out_root))

    f1s: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    for i, (gt, pred) in enumerate(pairs):
        res = cdm.evaluate(gt, pred, f"{i:04d}")
        f1s.append(float(res.get("F1_score", 0.0)))
        precs.append(float(res.get("precision", 0.0)))
        recs.append(float(res.get("recall", 0.0)))

    if not config.EQUATION_CDM_KEEP_ARTIFACTS:
        shutil.rmtree(out_root, ignore_errors=True)

    return {"pair_f1": f1s, "precision": precs, "recall": recs, "n_pairs": len(pairs)}
