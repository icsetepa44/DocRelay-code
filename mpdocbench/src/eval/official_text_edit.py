from __future__ import annotations

import sys
import time
import types
from typing import Any, Dict, Optional

from src.core import config

TEXT_CATEGORIES_FROM_FORMULA = ("equation_inline", "equation_isolated", "")

_STATE: Dict[str, Any] = {}


def _load() -> Dict[str, Any]:
    if _STATE:
        return _STATE
    repo = config.MPDOCBENCH_REPO
    if not (repo / "dataset" / "end2end_dataset.py").is_file():
        raise FileNotFoundError(f"MPDocBench repo not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if "dataset" not in sys.modules:
        pkg = types.ModuleType("dataset")
        pkg.__path__ = [str(repo / "dataset")]
        sys.modules["dataset"] = pkg
    import Levenshtein
    from dataset.end2end_dataset import End2EndDataset
    from pylatexenc.latex2text import LatexNodes2Text
    from utils.data_preprocess import clean_string

    matcher = End2EndDataset.__new__(End2EndDataset)
    matcher.match_method = "quick_match"
    matcher.required_metrics = set()
    _STATE.update(matcher=matcher, latex2text=LatexNodes2Text, clean_string=clean_string,
                  distance=Levenshtein.distance)
    return _STATE


def _text_samples(result) -> list:
    S = _load()
    samples = list(result[0])
    for item in result[2]:
        if item.get("pred_category_type", None) not in TEXT_CATEGORIES_FROM_FORMULA:
            item = dict(item)
            item["gt"] = S["latex2text"]().latex_to_text(item.get("gt", None))
            item["norm_gt"] = S["clean_string"](item["gt"])
            samples.append(item)
    return samples


def score(sample: Dict[str, Any], markdown: str, img_name: str) -> Optional[Dict[str, Any]]:
    S = _load()
    result = S["matcher"].process_get_matched_elements(sample, markdown, img_name, time.time())
    samples = _text_samples(result)
    edit_num = upper = 0
    for s in samples:
        gt = s["norm_gt"] if s.get("norm_gt") else s["gt"]
        pred = s["norm_pred"] if s.get("norm_pred") else s["pred"]
        upper += max(len(pred), len(gt))
        if len(pred) > 0 or len(gt) > 0:
            edit_num += S["distance"](pred, gt)
    if not samples or not upper:
        return None
    edit_dist = edit_num / upper
    return {"edit_num": edit_num, "upper": upper, "edit_dist": edit_dist, "score": 1.0 - edit_dist,
            "n_blocks": len(samples)}
