"""Read-only OCR preflight for a benchmark run (no LLM, no API, writes only its own report).

Run before a billed pipeline run (run.py RUN_OCR_PREFLIGHT) over the selected documents under
config.BENCHMARK_OCR_PARENT. Two severities:

- HARD (the run must not start): no OCR folder, no completion marker (<doc>.json), no pages, per-page
  markdown / layout / page_images counts that disagree, a missing page image, or no gold entry (the doc
  could never be scored).
- SOFT (reported, the run proceeds): near-empty pages, pages whose LLM payload exceeds the payload clip
  (`_common.MAX_TEXT_CHARS`; the tail is not sent), and suspected inline OCR repetition loops.

Also totals pages / characters / a rough input-token estimate per page payload, and a domain breakdown.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from src.core import config
from src.ocr.chandra_adapter import ocr_dir_for
from src.stages._common import MAX_TEXT_CHARS, mixed_ocr_text

NEAR_EMPTY_CHARS = 20
# A >=8-char chunk repeated back-to-back 20+ times (e.g. ", no, no, no ..."); repeated HTML table rows are
# legitimate and are not matched because each row differs.
_INLINE_LOOP = re.compile(r"(.{8,60}?)\1{20,}", re.S)


def _page_files(folder: Path, sub: str, pattern: str) -> List[Path]:
    d = folder / sub
    return sorted(d.glob(pattern)) if d.is_dir() else []


def check_document(doc: str, gold_stems: Optional[Set[str]] = None,
                   clip_chars: int = MAX_TEXT_CHARS) -> Dict[str, Any]:
    """Preflight one document. Returns {doc, domain, pages, chars, payload_chars, hard[], soft{...}}."""
    folder = ocr_dir_for(doc)  # flat or category-nested OCR root
    rec: Dict[str, Any] = {"doc": doc, "domain": config.domain_for(doc), "pages": 0, "chars": 0,
                           "payload_chars": 0, "hard": [],
                           "soft": {"near_empty": [], "over_clip": [], "repetition": []}}
    if not folder.is_dir():
        rec["hard"].append("no OCR folder")
        return rec
    if not (folder / f"{doc}.json").exists():
        rec["hard"].append("no completion marker")
    md = _page_files(folder, "markdown", "page_*.md")
    layout = _page_files(folder, "layout", "page_*.json")
    images = _page_files(folder, "page_images", "page_*.png")
    rec["pages"] = len(md)
    if not md:
        rec["hard"].append("no markdown pages")
    if not (len(md) == len(layout) == len(images)):
        rec["hard"].append(f"page count mismatch: markdown {len(md)} / layout {len(layout)} / "
                           f"page_images {len(images)}")
    image_names = {p.stem for p in images}
    for path in md:
        if path.stem not in image_names:
            rec["hard"].append(f"missing page image for {path.name}")
        text = path.read_text(encoding="utf-8", errors="replace")
        payload = len(mixed_ocr_text(text))
        rec["chars"] += len(text)
        rec["payload_chars"] += min(payload, clip_chars)
        if len(text.strip()) < NEAR_EMPTY_CHARS:
            rec["soft"]["near_empty"].append(path.stem)
        if payload > clip_chars:
            rec["soft"]["over_clip"].append({"page": path.stem, "payload_chars": payload})
        m = _INLINE_LOOP.search(text)
        if m:
            rec["soft"]["repetition"].append({"page": path.stem, "unit": m.group(1)[:40],
                                              "span_chars": len(m.group(0))})
    if gold_stems is not None and doc not in gold_stems:
        rec["hard"].append("no gold entry")
    if not rec["domain"]:
        rec["soft"].setdefault("other", []).append("domain unresolved")
    return rec


def run_preflight(docs: Iterable[str], gold_stems: Optional[Set[str]] = None,
                  clip_chars: int = MAX_TEXT_CHARS) -> Dict[str, Any]:
    """Preflight every doc; returns {ok, summary, hard, soft, docs}. Pure read-only."""
    recs = [check_document(d, gold_stems, clip_chars) for d in docs]
    hard = [{"doc": r["doc"], "problem": p} for r in recs for p in r["hard"]]
    soft: Dict[str, List[Any]] = {"near_empty": [], "over_clip": [], "repetition": []}
    for r in recs:
        for key in soft:
            soft[key] += [{"doc": r["doc"], **(x if isinstance(x, dict) else {"page": x})}
                          for x in r["soft"][key]]
    payload = sum(r["payload_chars"] for r in recs)
    summary = {
        "ocr_root": str(config.BENCHMARK_OCR_PARENT),
        "n_docs": len(recs),
        "n_pages": sum(r["pages"] for r in recs),
        "ocr_chars": sum(r["chars"] for r in recs),
        # ~4 chars/token for English; the OCR text per LLM call, before images, prompts and JSON.
        "est_page_text_tokens": payload // 4,
        "by_domain": dict(sorted(Counter(r["domain"] or "?" for r in recs).items())),
        "pages_by_domain": dict(sorted(
            {d: sum(r["pages"] for r in recs if (r["domain"] or "?") == d)
             for d in {r["domain"] or "?" for r in recs}}.items())),
        "n_hard": len(hard),
        "n_soft": {k: len(v) for k, v in soft.items()},
    }
    return {"ok": not hard, "summary": summary, "hard": hard, "soft": soft, "docs": recs}


def format_summary(report: Dict[str, Any], print_fn: Callable[[str], None] = print) -> None:
    s = report["summary"]
    print_fn(f"OCR preflight: {s['n_docs']} docs / {s['n_pages']} pages from {s['ocr_root']}")
    print_fn(f"  domains: {s['by_domain']}")
    print_fn(f"  OCR chars {s['ocr_chars']:,} (~{s['est_page_text_tokens']:,} page-text tokens per pass)")
    print_fn(f"  soft: {s['n_soft']}  hard: {s['n_hard']}")
    for h in report["hard"][:20]:
        print_fn(f"  HARD {h['doc']}: {h['problem']}")
