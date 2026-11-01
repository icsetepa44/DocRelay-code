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

_MATCHER = None
_CDM_CLASS = None


@lru_cache(maxsize=1)
def _spacefree_repo() -> Path:
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
            return repo
        link.symlink_to(repo, target_is_directory=True)
        return link
    except OSError:
        return repo


def _load_matcher():
    global _MATCHER
    if _MATCHER is None:
        repo_dir = str(_spacefree_repo())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from utils.match import match_gt2pred_simple, normalized_formula
        _MATCHER = (match_gt2pred_simple, normalized_formula)
    return _MATCHER


def _load_cdm():
    global _CDM_CLASS
    if _CDM_CLASS is None:
        repo_dir = str(_spacefree_repo())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        metrics_dir = str(_spacefree_repo() / "metrics")
        existing = sys.modules.get("metrics")
        if existing is None or getattr(existing, "__file__", None):
            stub = types.ModuleType("metrics")
            stub.__path__ = [metrics_dir]
            sys.modules["metrics"] = stub
        cdm_metric = importlib.import_module("metrics.cdm_metric")
        _CDM_CLASS = cdm_metric.CDM
    return _CDM_CLASS


@lru_cache(maxsize=1)
def cdm_available() -> bool:
    if not (shutil.which("xelatex") and shutil.which("magick") and shutil.which("node")):
        return False
    try:
        import inspect
        from skimage.measure import ransac
        if "random_state" not in inspect.signature(ransac).parameters:
            return False
    except Exception:
        return False
    return True


def _strip_delims(s: str, is_pred: bool = False) -> str:
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
        if m.get("gt_idx") == [""]:
            continue
        if m.get("gt_category_type") == "equation_inline":
            continue
        pairs.append((_strip_delims(m.get("gt", ""), is_pred=False),
                      _strip_delims(m.get("pred", ""), is_pred=True)))
    return pairs


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)) or "doc"


def score(gold_equations: List[Dict[str, Any]], pred_latex_list: List[str],
          img_name: str, work_dir: Path) -> Dict[str, Any]:
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
