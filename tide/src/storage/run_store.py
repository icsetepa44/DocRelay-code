from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core import config


def new_run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(run_id: Optional[str] = None) -> Path:
    run_id = run_id or new_run_id()
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def doc_dir(run_dir: Path, doc_name: str, model_label: Optional[str] = None) -> Path:
    d = run_dir / doc_name
    if model_label:
        d = d / model_label
    d.mkdir(parents=True, exist_ok=True)
    return d


def page_dir_path(run_dir: Path, doc_name: str, page_number: int,
                  model_label: Optional[str] = None) -> Path:
    d = run_dir / doc_name
    if model_label:
        d = d / model_label
    return d / "pages" / f"{page_number:04d}"


def page_dir(run_dir: Path, doc_name: str, page_number: int, model_label: Optional[str] = None) -> Path:
    d = page_dir_path(run_dir, doc_name, page_number, model_label)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
