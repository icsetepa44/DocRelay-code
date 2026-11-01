"""
MPDocBench evaluation adapter (fork).

Maps our ``FinalDocument.units`` onto MPDocBench's element/metric shape and scores them against
the MPDocBench gold, **reusing the benchmark's own leaf scorers** (``TEDS``, ``HeadTEDS``) and its
text / formula / table normalizers so results are comparable to their leaderboard. Their aggregation
layer (``metrics/cal_metric.py``) imports pandas/evaluate/CDM at module top and is only trivial
arithmetic, so we reimplement the aggregation to their *exact* formulas and reuse only the
non-trivial leaf algorithms.

Import strategy (verified): ``metrics/`` has a poisoned ``__init__`` (pandas/evaluate/CDM), but
``metrics/table_metric.py`` and ``metrics/head_metric.py`` use only absolute imports, so we file-load
them standalone (bypassing the package init). ``utils/`` has no ``__init__`` (namespace package), so
``utils.data_preprocess`` / ``utils.merge_tables`` / ``utils.relation_utils`` import cleanly. Required
pip deps beyond the fork's base: ``Levenshtein apted pylatexenc``.

Metrics (per doc; averaged per domain / overall):
  - text            whole-document concatenated-string edit distance (1 - normalized Levenshtein): a
                    segmentation-blind CONTENT-FIDELITY number (does the doc contain the right words),
                    NOT the paper's per-block ``edit_whole`` -> treat as our own, not directly comparable
  - text_diag       OUR non-comparable diagnostic: segmentation precision/recall/F1 of the pred<->gold
                    text-block matching (this is where under-segmentation / over-production show up, since
                    the whole-string ``text`` metric is blind to them)
  - equation        PAPER-FAITHFUL CDM (Character Detection Matching): reuses the paper's OWN formula
                    matcher (``utils.match.match_gt2pred_simple``) + CDM scorer (``metrics.cdm_metric.CDM``)
                    via ``src.eval.cdm_formula``, micro-averaged over all pairs. Carries ``method``; falls
                    back to the edit diagnostic (honestly labelled) when the CDM toolchain is unavailable.
  - equation_diag   OUR toolchain-free diagnostic: per-equation latex edit distance (matched by order) —
                    the pre-CDM metric, kept as a sanity anchor / fallback (not paper-comparable)
  - table           TEDS / TEDS-structure (matched by reading order)
  - heads           HeadTEDS on the heading tree
  - figures         per-page IoU@0.5 F1 (pred boxes from Chandra layout)
  - reading_order   paper-faithful matched-block order edit distance (``get_order_paired``): per page,
                    Levenshtein over the sequence of matched gold indices (both sides matched set); blocks
                    matched with the official ``utils.match_quick.match_gt2pred_quick`` (configs/end2end.yaml)
  - merged_text/table   cross-page truncation reconstruction (our headline capability)
  - relation        optional table/text truncation-pair F1 (their relation_utils)

Base-agnostic by design: page *rank* alignment (not raw page_id) for text_diag/reading-order block
matching and for figures, and order-rank matching for tables/equations, so the fork's OCR page numbering
need not match gold ``page_id``. reading_order is micro-averaged at the run level (Sum edits / Sum uppers,
matching the paper's ``edit_whole``); text (whole-string) and all other metrics are macro-averaged over docs.

No CLI: driven by ``run.py`` (RUN_EVAL) or ``python -m src.eval.mpdocbench``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DU = Path(__file__).resolve().parents[2]  # mpdocbench/
if str(_DU) not in sys.path:
    sys.path.insert(0, str(_DU))

from src.core import config  # noqa: E402
from src.ocr.chandra_adapter import ocr_dir_for  # noqa: E402  (same doc-dir resolver as the pipeline)
from src.eval import adapt  # noqa: E402  (reuse _unit_text / _iter_table_contents / _merge_table_contents / _top_ancestor)
from src.eval import cdm_formula  # noqa: E402  (paper-faithful Formula/CDM adapter; loaders lazy — no toolchain at import)
from src.eval.table_structure import to_grid, _assign  # noqa: E402  (_assign: guarded Hungarian/greedy)

# ============================================================
# CATEGORY MAPS (our block types <-> MPDocBench gold category_type)
# ------------------------------------------------------------
# Gold quirks confirmed against real data: equations are ``equation_isolated``;
# captions split into ``figure_caption`` / ``table_caption``; furniture is ``footer`` / ``page_number``
# / headers; ``poly`` is 8 floats; ``order`` is global; ``page_id`` is 1-based.
# ============================================================

BLOCK_TO_CATEGORY: Dict[str, str] = {
    "text_block": "text_block",
    "heading": "title",
    "table": "table",
    "figure": "figure",
    "equation": "equation_isolated",
    "caption": "caption",
    "list": "text_block",
    "code": "text_block",
}
FURNITURE_BLOCK_TYPES = {"header_footer_page_number"}
DROP_BLOCK_TYPES = {"noise_or_separator"}

# Chandra layout chunk labels that denote a figure (confirmed label universe).
FIGURE_LABELS = {"Image", "Figure", "Diagram"}

# Gold categories that carry comparable prose for the text-edit metric (exclude tables/equations/
# figures which have their own metrics, and furniture).
GOLD_TEXT_CATEGORIES = {"text_block", "title", "figure_caption", "table_caption", "caption",
                        "list", "footnote"}
# Our pred block types that feed the text stream (mirror of GOLD_TEXT_CATEGORIES on our side).
PRED_TEXT_BLOCK_TYPES = {"text_block", "heading", "caption", "list", "code"}

# OUR table diagnostic only (NOT the paper's metric): an assigned (gold, pred) table pair counts as a
# "true match" iff its TEDS is at least this — used by ``score_table_diag`` for precision/recall/F1 that
# expose over-prediction and misclassification. The paper-faithful headline ``table``
# metric has NO threshold (``match_tables`` assigns best-match, then scores mean TEDS).
TABLE_DIAG_MATCH_TEDS = 0.5

# OUR text diagnostic only (NOT the paper's metric): a matched (gold, pred) text-block pair counts as a
# "true match" iff its normalised edit distance is at most this — used by ``score_text_diag`` for
# segmentation precision/recall/F1 (over-production lowers precision; under-segmentation lowers recall).
TEXT_DIAG_MATCH_EDIT = 0.5


# ============================================================
# LEAF-SCORER SHIM (reuse the benchmark's own scorers/normalizers)
# ============================================================

_LEAF: Dict[str, Any] = {}


def _load_leaf() -> Dict[str, Any]:
    """Load and cache the benchmark's leaf scorers + normalizers.

    Raises a single clear ``ImportError`` naming the missing pip package, so a machine without the
    metric deps fails loudly here (not at ``import mpdocbench``).
    """
    if _LEAF:
        return _LEAF
    repo = config.MPDOCBENCH_ROOT / "repo" / "MPDocBench"
    if not repo.is_dir():
        raise FileNotFoundError(f"MPDocBench repo not found at {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    def _load_file(mod_name: str, rel: str):
        path = repo / rel
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build import spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except ImportError as exc:  # a leaf dep (Levenshtein/apted/lxml) is missing
            raise ImportError(
                f"MPDocBench metric dependency missing while loading {rel}: {exc}. "
                f"Install with:  pip install Levenshtein apted pylatexenc"
            ) from exc
        return mod

    tm = _load_file("mpdb_table_metric", "metrics/table_metric.py")
    hm = _load_file("mpdb_head_metric", "metrics/head_metric.py")
    try:
        dp = importlib.import_module("utils.data_preprocess")
        mt = importlib.import_module("utils.merge_tables")
        ru = importlib.import_module("utils.relation_utils")
        # text-block matcher: the official evaluation's quick_match (configs/end2end.yaml), which lets one pred
        # block cover several consecutive gold blocks (and the reverse)
        mq = importlib.import_module("utils.match_quick")
        import Levenshtein as lev  # noqa: N813
    except ImportError as exc:
        raise ImportError(
            f"MPDocBench metric dependency missing: {exc}. "
            f"Install with:  pip install Levenshtein apted pylatexenc beautifulsoup4 evaluate"
        ) from exc

    def _match_quick(gt_items, pred_items, line_type, img_name):
        # quick_match returns the record list alone; keep the (records, extra) shape of the simple matcher
        return mq.match_gt2pred_quick(gt_items, pred_items, line_type, img_name), None

    _LEAF.update(
        TEDS=tm.TEDS,
        HeadTEDS=hm.HeadTEDS,
        clean_string=dp.clean_string,
        textblock2unicode=dp.textblock2unicode,
        normalized_formula=dp.normalized_formula,
        normalized_table=dp.normalized_table,
        merge_tables=mt.merge_tables,
        relation_utils=ru,
        match_gt2pred=_match_quick,
        lev_distance=lev.distance,
    )
    return _LEAF


def _norm_text(s: str) -> str:
    L = _load_leaf()
    try:
        return L["clean_string"](L["textblock2unicode"](s or ""))
    except Exception:
        return (s or "").strip()


def _norm_table_html(html: str) -> str:
    L = _load_leaf()
    try:
        return L["normalized_table"](html or "", format="html")
    except Exception:
        return html or ""


def _norm_formula(latex: str) -> str:
    L = _load_leaf()
    try:
        return L["normalized_formula"](latex or "")
    except Exception:
        return (latex or "").strip()


def _edit(a: str, b: str) -> Tuple[int, int]:
    """(levenshtein distance, upper_len=max(len)). (0,0) when both empty."""
    L = _load_leaf()
    upper = max(len(a), len(b))
    if upper == 0:
        return 0, 0
    return L["lev_distance"](a, b), upper


def _edit_score(a: str, b: str) -> float:
    dist, upper = _edit(a, b)
    return 1.0 if upper == 0 else 1.0 - dist / upper


# ============================================================
# HELPERS
# ============================================================

def _parse_page_seq(block_id: str) -> Tuple[int, int]:
    """``doc:pNNNN:bNNNN`` -> (page, seq). (0, 0) on any parse failure."""
    try:
        parts = block_id.split(":")
        return int(parts[-2][1:]), int(parts[-1][1:])
    except (ValueError, IndexError, AttributeError):
        return 0, 0


def _grid_to_html(grid: List[List[str]]) -> str:
    """A rectangular string grid -> minimal HTML table (all <td>; the benchmark's normalizer
    unwraps <thead>/<th> anyway, so header vs body is irrelevant to TEDS)."""
    if not grid:
        return ""
    rows = []
    for row in grid:
        cells = "".join(f"<td>{'' if c is None else str(c)}</td>" for c in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _heading_level_text(unit: Dict[str, Any]) -> Tuple[int, str]:
    """(level>=1, text) for a heading unit (first fragment if stitched)."""
    content = unit.get("content")
    if isinstance(content, list):
        content = (content[0].get("content") if content else {}) or {}
    content = content or {}
    try:
        level = int(content.get("level") or 1)
    except (TypeError, ValueError):
        level = 1
    level = max(1, level)
    text = str(content.get("text") or content.get("description") or "").strip()
    return level, text


def _pred_table_html(unit: Dict[str, Any]) -> str:
    """HTML for a pred table unit: explicit ``html`` fallback wins, else render {headers,rows}
    (or markdown) via the shared ``to_grid``."""
    contents = list(adapt._iter_table_contents(unit))
    if not contents:
        return ""
    entry = adapt._merge_table_contents(contents)
    if not entry:
        return ""
    if entry.get("html"):
        return str(entry["html"])
    grid = to_grid(entry)
    if grid:
        return _grid_to_html(grid)
    return str(entry.get("markdown") or "")


# ============================================================
# PRED SIDE (our FinalDocument -> pred elements)
# ============================================================

@dataclass
class PredElement:
    unit_id: str
    block_type: str
    category: Optional[str]
    page: int
    seq: int
    text: str = ""
    html: str = ""
    latex: str = ""
    table_text: str = ""  # flattened cell text of a table unit, used if it is demoted to text
    furniture: bool = False
    merged: bool = False  # spans >1 source page (our cross-page stitch)


@dataclass
class PredDoc:
    elements: List[PredElement] = field(default_factory=list)
    head_md: List[str] = field(default_factory=list)         # ["# Title", "## Sub", ...]
    figures: List[Dict[str, Any]] = field(default_factory=list)  # [{page_id, page_rank, bbox_norm}]
    n_pages: int = 0
    # Full physical-page universe (sorted), so text-block ranking aligns page-for-page with gold even
    # when a page carries no text on this side (cover / silently text-dropped page). Empty -> the text
    # matcher falls back to the pages that actually bear text (legacy behavior; see _group_text_by_rank).
    page_universe: List[int] = field(default_factory=list)


def build_pred_doc(fd: Dict[str, Any], doc: str) -> PredDoc:
    units: List[Dict[str, Any]] = fd.get("units", []) or []
    elements: List[PredElement] = []
    for u in units:
        bt = u.get("block_type", "")
        if bt in DROP_BLOCK_TYPES:
            continue
        src = u.get("source_block_ids") or []
        page, seq = _parse_page_seq(src[0]) if src else (0, 0)
        pages = {_parse_page_seq(b)[0] for b in src} if src else {page}
        el = PredElement(
            unit_id=u.get("unit_id", ""),
            block_type=bt,
            category=BLOCK_TO_CATEGORY.get(bt),
            page=page,
            seq=seq,
            furniture=bt in FURNITURE_BLOCK_TYPES,
            merged=len(pages) > 1,
        )
        if bt == "table":
            el.html = _pred_table_html(u)
            el.table_text = adapt._unit_text(u)  # kept so apply_table_text_redirect can move it to text
        elif bt == "equation":
            el.latex = adapt._unit_text(u)
        else:
            el.text = adapt._unit_text(u)
        elements.append(el)

    elements.sort(key=lambda e: (e.page, e.seq))

    head_md: List[str] = []
    for e in elements:
        if e.block_type == "heading":
            u = next((x for x in units if x.get("unit_id") == e.unit_id), None)
            level, text = _heading_level_text(u or {})
            head_md.append("#" * level + " " + text)

    figures = _pred_figures(doc)
    # Page universe = every page the units actually touch (furniture included; only DROP_BLOCK_TYPES are
    # skipped). match_text unions this with gold's universe on a shared physical frame, so a whole-page
    # drop is covered by gold's side and an interior page empty on BOTH sides simply never appears.
    page_universe = sorted({e.page for e in elements})
    return PredDoc(elements=elements, head_md=head_md, figures=figures,
                   n_pages=len(page_universe), page_universe=page_universe)


def _pred_figures(doc: str) -> List[Dict[str, Any]]:
    """Figure boxes from the doc's Chandra layout jsons, normalized to 0-1000 per page_box.

    Returns [{page_id, page_rank, bbox_norm:[x1,y1,x2,y2]}]. ``page_rank`` is the 0-based rank of the
    page among the doc's sorted layout page_nums (base-agnostic vs gold page_id)."""
    layout_dir = ocr_dir_for(doc) / "layout"  # flat or category-nested OCR root (pipeline.yaml inputs.ocr_root)
    if not layout_dir.is_dir():
        return []
    per_page: List[Tuple[int, List[Dict[str, Any]]]] = []
    for path in sorted(layout_dir.glob("page_*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        pb = d.get("page_box") or [0, 0, 0, 0]
        pw, ph = (pb[2] - pb[0]), (pb[3] - pb[1])
        if not pw or not ph:
            continue
        page_num = int(d.get("page_num", 0))
        boxes = []
        for c in d.get("chunks", []) or []:
            if c.get("label") not in FIGURE_LABELS:
                continue
            bb = c.get("bbox") or []
            if len(bb) < 4:
                continue
            boxes.append({
                "bbox_norm": [bb[0] / pw * 1000, bb[1] / ph * 1000,
                              bb[2] / pw * 1000, bb[3] / ph * 1000],
            })
        per_page.append((page_num, boxes))
    per_page.sort(key=lambda x: x[0])
    out: List[Dict[str, Any]] = []
    for rank, (page_num, boxes) in enumerate(per_page):
        for b in boxes:
            out.append({"page_id": page_num, "page_rank": rank, "bbox_norm": b["bbox_norm"]})
    return out


# ============================================================
# GOLD SIDE (MPDocBench gold entry -> gold elements)
# ============================================================

@dataclass
class GoldDoc:
    text_elems: List[Dict[str, Any]] = field(default_factory=list)   # {page_id, order, category, text}
    tables: List[Dict[str, Any]] = field(default_factory=list)       # {page_id, order, html, is_merged, cross_pages}
    equations: List[Dict[str, Any]] = field(default_factory=list)    # {page_id, order, latex}
    figures: List[Dict[str, Any]] = field(default_factory=list)      # {page_id, page_rank, bbox_norm}
    head_input: Any = None                                           # [parent_son_list, head_nodes]
    n_titles: int = 0
    merged_text: List[Dict[str, Any]] = field(default_factory=list)
    merged_tables: List[Dict[str, Any]] = field(default_factory=list)
    # Full physical-page universe (sorted distinct page_id over ALL layout dets) — see PredDoc.page_universe.
    page_universe: List[int] = field(default_factory=list)


_GOLD_INDEX: Dict[str, Dict[str, Any]] = {}


def load_gold_index() -> Dict[str, Dict[str, Any]]:
    """{image_path.stem: gold_entry} over the whole MPDocBench gold (cached)."""
    if _GOLD_INDEX:
        return _GOLD_INDEX
    data = json.loads(config.MPDOCBENCH_GOLD.read_text(encoding="utf-8"))
    for entry in data:
        stem = Path((entry.get("page_info") or {}).get("image_path", "")).stem
        if stem:
            _GOLD_INDEX[stem] = entry
    return _GOLD_INDEX


def _remove_consecutive_dupes(arr: List[Any]) -> List[Any]:
    if not arr:
        return []
    out = [arr[0]]
    for x in arr[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _merge_truncated(entry: Dict[str, Any]):
    """Mirror of End2EndDataset.get_page_elements: returns (per_category dict, merged_records).

    Cross-page/truncated fragments are grouped (from extra.relation truncated), sorted by ``order``,
    and merged (tables via merged_html/merge_tables, text via concatenation), tagged with is_merged
    and *_merge_cross_pages. Non-truncated items pass through by category."""
    L = _load_leaf()
    relations = (entry.get("extra") or {}).get("relation") or []
    dets = entry.get("layout_dets") or []

    related: List[List[Any]] = []
    truncated_all: Dict[Any, Any] = {}
    for r in relations:
        if r.get("relation_type") != "truncated":
            continue
        s, t = r.get("source_anno_id"), r.get("target_anno_id")
        truncated_all.setdefault(s, "")
        truncated_all.setdefault(t, "")
        placed = False
        for grp in related:
            if s in grp or t in grp:
                grp.append(s)
                grp.append(t)
                placed = True
        if not placed:
            related.append([s, t])

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for item in dets:
        aid = item.get("anno_id")
        if aid not in truncated_all:
            by_cat.setdefault(item.get("category_type"), []).append(item)
        else:
            truncated_all[aid] = item

    merged_records: List[Dict[str, Any]] = []
    for grp in related:
        grp = _remove_consecutive_dupes(grp)
        blocks = [truncated_all[k] for k in grp if isinstance(truncated_all.get(k), dict)]
        if not blocks:
            continue
        blocks.sort(key=lambda x: x.get("order", 0))
        page_ids = {b.get("page_id") for b in blocks}
        first = blocks[0]
        if first.get("category_type") == "table":
            html = first.get("merged_html")
            if html is None:
                try:
                    html = L["merge_tables"]([b.get("html", "") for b in blocks])
                except Exception:
                    html = "".join(b.get("html", "") for b in blocks)
            rec = {"category_type": "table", "order": first.get("order", 0),
                   "page_id": first.get("page_id"), "html": html, "is_merged": "table",
                   "cross_pages": len(page_ids)}
        else:
            text = "".join(b.get("text", "") for b in blocks)
            rec = {"category_type": first.get("category_type"), "order": first.get("order", 0),
                   "page_id": first.get("page_id"), "text": text,
                   "is_merged": "text_block" if first.get("category_type") == "text_block" else "unknown",
                   "cross_pages": len(page_ids)}
        by_cat.setdefault(rec["category_type"], []).append(rec)
        merged_records.append(rec)
    return by_cat, merged_records


def _document_heads(entry: Dict[str, Any]):
    """Mirror of End2EndDataset.get_document_heads: (parent_son_list, head_nodes) restricted to
    title nodes; head_nodes items already carry 'text' and 'anno_id' for HeadTEDS."""
    dets = entry.get("layout_dets") or []
    head_nodes = {it.get("anno_id"): it for it in dets if it.get("category_type") == "title"}
    parent_son = []
    for r in (entry.get("extra") or {}).get("relation") or []:
        if r.get("relation_type") != "parent_son":
            continue
        s, t = r.get("source_anno_id"), r.get("target_anno_id")
        if (s in head_nodes or s == "root") and t in head_nodes:
            parent_son.append(r)
    return parent_son, head_nodes


def _gold_figures(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gold figure boxes normalized to 0-1000 (poly[0,1,4,5] / per-page width,height); page_rank is
    the 0-based rank among the doc's sorted distinct page_ids (base-agnostic)."""
    pi = entry.get("page_info") or {}
    widths, heights = pi.get("width") or [], pi.get("height") or []
    all_pages = sorted({it.get("page_id") for it in entry.get("layout_dets") or []
                        if it.get("page_id") is not None})
    rank_of = {p: i for i, p in enumerate(all_pages)}
    figs = []
    for det in entry.get("layout_dets") or []:
        if det.get("category_type") != "figure":
            continue
        pid = det.get("page_id")
        poly = det.get("poly") or []
        if pid is None or len(poly) < 6:
            continue
        idx = pid - 1
        if not isinstance(widths, list) or idx < 0 or idx >= len(widths) or idx >= len(heights):
            continue
        pw, ph = widths[idx], heights[idx]
        if not pw or not ph:
            continue
        figs.append({
            "page_id": pid,
            "page_rank": rank_of.get(pid, pid),
            "bbox_norm": [poly[0] / pw * 1000, poly[1] / ph * 1000,
                          poly[4] / pw * 1000, poly[5] / ph * 1000],
        })
    return figs


def build_gold_doc(entry: Dict[str, Any]) -> GoldDoc:
    by_cat, merged = _merge_truncated(entry)
    g = GoldDoc()

    for cat, items in by_cat.items():
        for it in items:
            pid = it.get("page_id", 0) or 0
            order = it.get("order", 0) or 0
            if cat == "table":
                g.tables.append({"page_id": pid, "order": order, "html": it.get("html", ""),
                                 "is_merged": it.get("is_merged"), "cross_pages": it.get("cross_pages", 1)})
            elif cat == "equation_isolated":
                g.equations.append({"page_id": pid, "order": order, "latex": it.get("latex", "")})
            elif cat in GOLD_TEXT_CATEGORIES:
                g.text_elems.append({"page_id": pid, "order": order, "category": cat,
                                     "text": it.get("text", "")})
    g.tables.sort(key=lambda x: (x["page_id"], x["order"]))
    g.equations.sort(key=lambda x: (x["page_id"], x["order"]))
    g.text_elems.sort(key=lambda x: (x["page_id"], x["order"]))

    parent_son, head_nodes = _document_heads(entry)
    g.head_input = [parent_son, head_nodes]
    g.n_titles = len(head_nodes)

    g.figures = _gold_figures(entry)

    g.merged_tables = [m for m in merged if m.get("is_merged") == "table"]
    g.merged_text = [m for m in merged if m.get("is_merged") == "text_block"]
    # Full page universe = every page that carries ANY det (text/table/figure/furniture), so text-block
    # ranking aligns page-for-page with pred even on pages that bear no comparable text.
    g.page_universe = sorted({d.get("page_id") for d in (entry.get("layout_dets") or [])
                              if d.get("page_id") is not None})
    return g


# ============================================================
# METRIC FUNCTIONS
# ============================================================

@dataclass
class TextMatch:
    """Paper-faithful matching of pred text blocks to gold text blocks, per page-rank.

    ``per_page[r]`` is ``match_gt2pred_quick``'s record list for the r-th page (gold text items vs pred
    text items; one record may cover several gold blocks). Consumed by ``score_text`` (edit distance over matched pairs), ``score_reading_order``
    (order of the matched gold indices) and ``score_text_diag`` (segmentation P/R/F1)."""
    per_page: List[List[Dict[str, Any]]]
    n_gold: int
    n_pred: int


def _gold_text_entries(gold: GoldDoc) -> List[Tuple[int, int, str, str]]:
    return [(t.get("page_id", 0) or 0, t.get("order", 0) or 0, t.get("text", ""),
             t.get("category") or "text_block") for t in gold.text_elems]


def _pred_text_entries(pred: PredDoc) -> List[Tuple[int, int, str, str]]:
    return [(e.page, e.seq, e.text, e.category or e.block_type) for e in pred.elements
            if e.block_type in PRED_TEXT_BLOCK_TYPES and not e.furniture and e.text]


def _group_text_by_rank(entries: List[Tuple[int, int, str, str]],
                        rank_of: Dict[int, int]) -> Dict[int, List[Dict[str, Any]]]:
    """Drop items whose *normalised* text is empty (so the matcher's own empty-filter can never desync
    its record indices), place each item on its page's SHARED rank (``rank_of``, built by ``match_text``
    so rank *i* denotes the same physical page on both sides), and within each rank sort by reading order
    and emit the ``text_all`` dicts the benchmark's matcher expects — with a 1-based within-page
    ``order``/``position`` (1-based so ``get_order_paired`` never drops element #0). Entries whose page is
    absent from ``rank_of`` are skipped defensively (``match_text`` seeds ``rank_of`` from the entries, so
    this cannot drop real content)."""
    from collections import defaultdict
    kept = [(p, o, txt, cat) for (p, o, txt, cat) in entries if _norm_text(txt) and p in rank_of]
    groups: Dict[int, List[Tuple[int, str, str]]] = defaultdict(list)
    for (p, o, txt, cat) in kept:
        groups[rank_of[p]].append((o, txt, cat))
    out: Dict[int, List[Dict[str, Any]]] = {}
    for r, items in groups.items():
        items.sort(key=lambda x: x[0])
        out[r] = [{"category_type": "text_all", "content": txt, "order": i + 1,
                   "position": [i + 1], "fine_category_type": cat}
                  for i, (_o, txt, cat) in enumerate(items)]
    return out


def match_text(pred: PredDoc, gold: GoldDoc) -> TextMatch:
    """Match pred text blocks to gold text blocks the official way (``match_gt2pred_quick``, the matcher
    of MPDocBench's configs/end2end.yaml: one pred block may cover several gold blocks), per page-rank.
    Call AFTER ``apply_table_text_redirect`` so pred tables that won no gold table (really prose) are
    counted in the text stream."""
    matcher = _load_leaf()["match_gt2pred"]
    gold_entries = _gold_text_entries(gold)   # page = gold page_id (1-based physical)
    pred_entries = _pred_text_entries(pred)   # page = OCR page (0-based)
    # Align both sides on ONE shared physical-page frame. gold page_id and pred OCR page index the SAME
    # physical page up to a constant base offset (OCR is 0-based, gold 1-based -> offset 1; verified
    # on the development documents). Derive the offset from each side's MINIMUM page so it self-adjusts
    # (real data -> 1; a same-frame hand-built fixture -> 0) instead of hardcoding it. Then rank over the
    # UNION of both sides' full page universes (all dets / all units, each unioned with its text pages),
    # so a cover, an interior page empty on both sides, an over-produced page, or a whole-page drop keeps
    # the SHARED pages aligned and costs only its own rank — never shifting the pages after it.
    gold_pages = set(gold.page_universe) | {p for (p, _, _, _) in gold_entries}
    pred_pages = set(pred.page_universe) | {p for (p, _, _, _) in pred_entries}
    offset = (min(gold_pages) - min(pred_pages)) if (gold_pages and pred_pages) else 0
    phys = sorted(gold_pages | {p + offset for p in pred_pages})
    rank = {ph: i for i, ph in enumerate(phys)}
    gold_rank_of = {p: rank[p] for p in gold_pages}
    pred_rank_of = {p: rank[p + offset] for p in pred_pages}
    gold_groups = _group_text_by_rank(gold_entries, gold_rank_of)
    pred_groups = _group_text_by_rank(pred_entries, pred_rank_of)
    per_page: List[List[Dict[str, Any]]] = []
    for r in sorted(set(gold_groups) | set(pred_groups)):
        records, _extra = matcher(gold_groups.get(r, []), pred_groups.get(r, []), "text", f"rank{r}")
        per_page.append(records)
    n_gold = sum(len(v) for v in gold_groups.values())
    n_pred = sum(len(v) for v in pred_groups.values())
    return TextMatch(per_page=per_page, n_gold=n_gold, n_pred=n_pred)


def score_text(pred: PredDoc, gold: GoldDoc) -> Dict[str, Any]:
    """Whole-document text edit distance (the "smoothie" content-fidelity metric): concatenate ALL gold
    text and ALL pred text into one string each, normalise, and take a single Levenshtein.

    This is deliberately **segmentation-blind** — it measures whether the *words* were captured, not how
    they were chopped into blocks. So it neither penalises under-segmentation (merged blocks are invisible
    once concatenated) nor credits finer splitting; the segmentation signal lives in ``score_text_diag``
    (and, for ordering, ``score_reading_order``). NB: because it is whole-string, it is NOT directly
    comparable to the paper's per-block ``edit_whole`` — treat it as our own content-fidelity number.

    Call AFTER ``apply_table_text_redirect`` so pred tables that won no gold table (really prose) are
    included in the pred string."""
    gold_txt = _norm_text(" ".join(t["text"] for t in gold.text_elems if t.get("text")))
    pred_txt = _norm_text(" ".join(
        e.text for e in pred.elements
        if e.block_type in PRED_TEXT_BLOCK_TYPES and not e.furniture and e.text))
    dist, upper = _edit(pred_txt, gold_txt)
    return {"edit_dist": (dist / upper) if upper else 0.0,
            "score": 1.0 - (dist / upper) if upper else 1.0,
            "gold_chars": len(gold_txt), "pred_chars": len(pred_txt)}


def score_text_diag(match: TextMatch) -> Optional[Dict[str, Any]]:
    """OUR diagnostic (NOT paper-comparable): segmentation precision/recall/F1 of the text matching. A
    matched pair is a "true match" iff its normalised edit distance <= ``TEXT_DIAG_MATCH_EDIT``; a record
    that covers several gold blocks counts once.
    recall = true/n_gold (under-segmentation lowers it: merged blocks leave gold blocks unmatched);
    precision = true/n_pred (over-production lowers it: extra pred blocks win no gold). This is the real
    signal the headline text edit deliberately hides. None when there is no text on either side."""
    if match.n_gold == 0 and match.n_pred == 0:
        return None
    L = _load_leaf()
    lev = L["lev_distance"]
    true_matches = 0
    for records in match.per_page:
        for rec in records:
            if rec["gt_idx"] == [""] or not (rec.get("norm_pred") or ""):
                continue
            ng, npr = rec.get("norm_gt", "") or "", rec.get("norm_pred", "") or ""
            u = max(len(ng), len(npr))
            if u and (lev(npr, ng) / u) <= TEXT_DIAG_MATCH_EDIT:
                true_matches += 1
    precision = true_matches / match.n_pred if match.n_pred else 0.0
    recall = true_matches / match.n_gold if match.n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_gold": match.n_gold, "n_pred": match.n_pred, "n_true_matches": true_matches}


def score_equations_diag(pred: PredDoc, gold: GoldDoc) -> Optional[Dict[str, Any]]:
    """Toolchain-free diagnostic (ours, NOT the paper): per-equation normalized-Levenshtein edit
    similarity, paired by reading order (gold[i] vs pred[i]). Kept alongside the paper-faithful CDM
    `equation` as `equation_diag`, and reused verbatim as the honest fallback when CDM is unavailable."""
    if not gold.equations:
        return None
    pred_eqs = [e.latex for e in pred.elements if e.block_type == "equation" and e.latex]
    scores = []
    for i, ge in enumerate(gold.equations):
        g = _norm_formula(ge.get("latex", ""))
        p = _norm_formula(pred_eqs[i]) if i < len(pred_eqs) else ""
        scores.append(_edit_score(p, g))
    return {"score": sum(scores) / len(scores) if scores else 0.0,
            "n_gold": len(gold.equations), "n_pred": len(pred_eqs)}


def score_equations(pred: PredDoc, gold: GoldDoc, doc: str = "",
                    work_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Paper-faithful Formula metric = CDM (Character Detection Matching).

    Reuses the paper's OWN code via ``src.eval.cdm_formula`` (unchanged): align our predicted equation
    LaTeX to the gold with ``utils.match.match_gt2pred_simple``, score each matched pair with the paper's
    ``metrics.cdm_metric.CDM.evaluate``, and report the per-doc micro-mean of pair F1 (`score`). The corpus
    number is micro-aggregated over ALL pairs in ``_aggregate`` (see ``_equation_micro``), matching the
    paper. Falls back to the edit-distance diagnostic (``method='edit'``) — never crashing the eval — when
    ``EQUATION_METRIC != 'cdm'`` or the CDM toolchain (xelatex + ImageMagick + node) is unavailable; the
    ``method`` field keeps a fallback from ever being presented as CDM."""
    if not gold.equations:
        return None
    if config.EQUATION_METRIC == "cdm" and cdm_formula.cdm_available():
        try:
            pred_eqs = [e.latex for e in pred.elements if e.block_type == "equation" and e.latex]
            wd = Path(work_dir) if work_dir is not None else (config.RUNS_DIR / "_cdm_work")
            r = cdm_formula.score(gold.equations, pred_eqs, doc or "doc", wd)
            f1s = r["pair_f1"]
            return {"score": (sum(f1s) / len(f1s)) if f1s else 0.0,
                    "n_gold": len(gold.equations), "n_pred": len(pred_eqs),
                    "n_pairs": r["n_pairs"], "sum_f1": float(sum(f1s)),
                    "precision": (sum(r["precision"]) / len(r["precision"])) if r["precision"] else 0.0,
                    "recall": (sum(r["recall"]) / len(r["recall"])) if r["recall"] else 0.0,
                    "method": "cdm"}
        except Exception as exc:  # noqa: BLE001 — a metric must never break the eval; degrade to edit
            print(f"[cdm_formula] CDM scoring failed for {doc!r} ({exc}); falling back to edit distance")
    diag = score_equations_diag(pred, gold)
    return {**diag, "method": "edit"} if diag is not None else None


@dataclass
class TableMatch:
    """Result of matching predicted table elements to gold tables the paper's way (best-match).

    ``gold_scores`` has one entry per gold table (TEDS vs its assigned pred, or 0.0 if that gold got no
    pred — MPDocBench counts a missed gold as 0). ``unmatched_pred`` are the pred table *elements* that
    won no gold (extra predictions); their text is redirected into the text metric by
    ``apply_table_text_redirect``. ``pair_teds`` (assigned pairs only) feeds ``score_table_diag``.
    """
    gold_scores: List[float]
    gold_scores_struct: List[float]
    pair_teds: List[float]
    matched_pred_idxs: set
    unmatched_pred: List["PredElement"]
    n_gold: int
    n_pred: int


def match_tables(pred: PredDoc, gold: GoldDoc) -> TableMatch:
    """Assign predicted tables to gold tables by MPDocBench's method: globally-optimal (Hungarian, or
    greedy fallback via ``_assign``) matching that maximises total table similarity (1 - normalised HTML
    edit distance), then score each gold against its assigned pred with TEDS. No assignment threshold
    (as in the paper: even a poor pair is assigned and its low TEDS counts). Runs even when the gold has
    no tables, so the unmatched pred tables can still be redirected to the text metric."""
    pred_tbl = [e for e in pred.elements if e.block_type == "table" and e.html]
    gold_htmls = [_norm_table_html(t.get("html", "")) for t in gold.tables]
    pred_htmls = [_norm_table_html(e.html) for e in pred_tbl]
    n_gold, n_pred = len(gold_htmls), len(pred_tbl)
    if n_gold == 0 or n_pred == 0:
        return TableMatch([0.0] * n_gold, [0.0] * n_gold, [], set(), list(pred_tbl), n_gold, n_pred)
    sim = [[_edit_score(g, p) for p in pred_htmls] for g in gold_htmls]  # 1 - normalised edit distance
    gold_to_pred = {gi: pi for gi, pi in _assign(sim)}  # (gold_idx, pred_idx) optimal pairs
    L = _load_leaf()
    teds = L["TEDS"](structure_only=False)
    teds_s = L["TEDS"](structure_only=True)
    gold_scores: List[float] = []
    gold_scores_struct: List[float] = []
    pair_teds: List[float] = []
    matched: set = set()
    for gi in range(n_gold):
        pi = gold_to_pred.get(gi)
        if pi is None:  # this gold table got no prediction -> 0 (recall penalty, as in the paper)
            gold_scores.append(0.0)
            gold_scores_struct.append(0.0)
            continue
        g_html, p_html = gold_htmls[gi], pred_htmls[pi]
        try:
            fv = teds.evaluate(p_html, g_html)
        except Exception:
            fv = 0.0
        try:
            sv = teds_s.evaluate(p_html, g_html)
        except Exception:
            sv = 0.0
        gold_scores.append(fv)
        gold_scores_struct.append(sv)
        pair_teds.append(fv)
        matched.add(pi)
    unmatched = [e for idx, e in enumerate(pred_tbl) if idx not in matched]
    return TableMatch(gold_scores, gold_scores_struct, pair_teds, matched, unmatched, n_gold, n_pred)


def score_tables(pred: PredDoc, gold: GoldDoc, match: TableMatch) -> Optional[Dict[str, Any]]:
    """Paper-faithful Table TEDS: mean TEDS over gold tables under the best-match assignment (a missed
    gold scores 0). Over-predicted tables are NOT penalised here — their text is redirected to the text
    metric by ``apply_table_text_redirect`` — mirroring MPDocBench's own table metric (Hungarian match,
    mean TEDS, no precision term). ``score_table_diag`` is our separate, non-comparable count metric."""
    if not gold.tables:
        return None
    gs, gss = match.gold_scores, match.gold_scores_struct
    return {"teds": sum(gs) / len(gs) if gs else 0.0,
            "teds_struct": sum(gss) / len(gss) if gss else 0.0,
            "n_gold": match.n_gold, "n_pred": match.n_pred, "n_matched": len(match.matched_pred_idxs)}


def score_heads(pred: PredDoc, gold: GoldDoc) -> Optional[Dict[str, Any]]:
    if gold.n_titles == 0:
        return None
    L = _load_leaf()
    try:
        score, depth = L["HeadTEDS"]().evaluate(list(pred.head_md), gold.head_input)
    except Exception:
        return {"score": 0.0, "gt_depth": 0, "n_gold_titles": gold.n_titles,
                "n_pred_heads": len(pred.head_md)}
    return {"score": float(score), "gt_depth": int(depth),
            "n_gold_titles": gold.n_titles, "n_pred_heads": len(pred.head_md)}


def score_figures(pred: PredDoc, gold: GoldDoc, iou_threshold: float = 0.5) -> Optional[Dict[str, Any]]:
    if not gold.figures:
        return None
    from collections import defaultdict
    gt_by_rank: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    for i, f in enumerate(gold.figures):
        gt_by_rank[f["page_rank"]].append((i, f["bbox_norm"]))
    pred_by_rank: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    for j, f in enumerate(pred.figures):
        pred_by_rank[f["page_rank"]].append((j, f["bbox_norm"]))

    pairs = []
    for rank, gts in gt_by_rank.items():
        for gi, gbox in gts:
            for pj, pbox in pred_by_rank.get(rank, []):
                iou = _compute_iou(gbox, pbox)
                if iou >= iou_threshold:
                    pairs.append((gi, pj, iou))
    pairs.sort(key=lambda x: x[2], reverse=True)
    m_gt, m_pred = set(), set()
    for gi, pj, _ in pairs:
        if gi not in m_gt and pj not in m_pred:
            m_gt.add(gi)
            m_pred.add(pj)
    gt_count, pred_count, matched = len(gold.figures), len(pred.figures), len(m_gt)
    precision = matched / pred_count if pred_count else 0.0
    recall = matched / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": f1, "precision": precision, "recall": recall,
            "gt_count": gt_count, "pred_count": pred_count, "matched": matched}


def _compute_iou(a: List[float], b: List[float]) -> float:
    xl, yt = max(a[0], b[0]), max(a[1], b[1])
    xr, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xr - xl) * max(0.0, yb - yt)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_reading_order(match: TextMatch) -> Dict[str, Any]:
    """Paper-faithful reading order (MPDocBench ``get_order_paired``), per page-rank: build ``gt`` = the
    order-positions of every gold text block, and ``pred`` = those same positions reordered by where the
    matched pred put them, then Levenshtein over the two index sequences, micro-averaged
    Sum(edit)/Sum(upper) across pages. Because both sides draw from the same matched gold-index universe,
    segmentation *count* alone no longer inflates the score (as the old category-symbol proxy did) — a
    genuine order permutation does, and unmatched gold blocks still cost deletions. A gold block covered
    by a pred block that also covers its neighbours (quick_match) counts as matched, in that block's place.
    Positions are 1-based so the ``if x`` guards (mirrored from the benchmark) never drop element #0."""
    L = _load_leaf()
    lev = L["lev_distance"]
    edit_num = upper = 0
    n_gold = n_matched = 0
    for records in match.per_page:
        matched = [(rec["gt_position"], rec["pred_position"]) for rec in records
                   if rec["gt_position"] != [""] and rec["pred_position"] != ""]
        gt_idx_all = [rec["gt_position"] for rec in records if rec["gt_position"] != [""]]
        read_order_pred = [gp for gp, _pp in sorted(matched, key=lambda x: x[1])]
        gt = sorted([x for grp in gt_idx_all for x in grp if x])
        prd = [x for grp in read_order_pred for x in grp if x]
        if not gt and not prd:
            continue
        # tokens are small 1-based ints; map to distinct chars so any Levenshtein backend accepts them
        # (edit distance is invariant under this bijection, so results match the paper's int-list form).
        gt_s = "".join(chr(x) for x in gt)
        prd_s = "".join(chr(x) for x in prd)
        edit_num += lev(prd_s, gt_s)
        upper += max(len(gt_s), len(prd_s))
        n_gold += len(gt)
        n_matched += len(prd)
    return {"edit_num": edit_num, "upper": upper,
            "score": (1.0 - edit_num / upper) if upper else 1.0,
            "n_gold": n_gold, "n_pred": n_matched, "n_matched": n_matched}


def score_merged_text(pred: PredDoc, gold: GoldDoc) -> Optional[Dict[str, Any]]:
    """Reconstruction fidelity of each gold merged (truncated) text element.

    MPDocBench's *Truncated Text* metric scores how faithfully the **reconstructed merged element**
    is reproduced — not whether the system flagged a multi-page span. Crucially, MPDocBench's own
    truncated relations are dominated by **within-page** (column->column) continuations, which our
    extractor reconstructs *inline* as ONE ``text_block``. So we score each gold merged element
    against the **best-matching single pred text unit** (any page span): a correct inline merge ->
    the block holding the rejoined paragraph matches; a genuine column-split -> no single unit
    matches. (The earlier cross-page-span-only proxy scored a correct within-page merge as 0.)
    ``n_pred_merges`` — units whose ``source_block_ids`` span >1 page — is kept only as a diagnostic.
    """
    if not gold.merged_text:
        return None
    cands = [c for c in (_norm_text(e.text) for e in pred.elements
                         if e.block_type in PRED_TEXT_BLOCK_TYPES and not e.furniture and e.text) if c]
    scores: List[float] = []
    n_reconstructed = 0
    for m in sorted(gold.merged_text, key=lambda x: x.get("order", 0)):
        g = _norm_text(m.get("text", ""))
        if not g:
            continue
        # A far-longer candidate can only score low (edit_score <= 1 - |Δlen|/max_len); skip it for
        # speed, keep every candidate of comparable length so the true reconstruction is never pruned.
        cap = 4 * len(g) + 200
        best = max((_edit_score(c, g) for c in cands if len(c) <= cap), default=0.0)
        scores.append(best)
        if best >= 0.8:
            n_reconstructed += 1
    return {"score": sum(scores) / len(scores) if scores else 0.0,
            "n_gold_merges": len(gold.merged_text),
            "n_reconstructed": n_reconstructed,
            "n_pred_merges": sum(1 for e in pred.elements
                                 if e.merged and e.block_type in PRED_TEXT_BLOCK_TYPES)}


def score_merged_table(pred: PredDoc, gold: GoldDoc) -> Optional[Dict[str, Any]]:
    """Reconstruction fidelity of each gold merged table (best TEDS vs any single pred table).

    Same intent as ``score_merged_text``: a same-page or cross-page merged table that we emit as
    ONE table is credited; a split is not. Matching by best TEDS (not by reading-order rank) so a
    correct same-page column-split table — which our pipeline cannot flag as cross-page — is still
    credited. ``n_pred_merges`` (cross-page-flagged tables) kept as a diagnostic.
    """
    if not gold.merged_tables:
        return None
    L = _load_leaf()
    teds = L["TEDS"](structure_only=False)
    cands = [c for c in (_norm_table_html(e.html) for e in pred.elements
                         if e.block_type == "table" and e.html) if c]
    scores: List[float] = []
    n_reconstructed = 0
    for gm in sorted(gold.merged_tables, key=lambda x: x.get("order", 0)):
        g_html = _norm_table_html(gm.get("html", ""))
        if not g_html:
            continue
        cap = 4 * len(g_html) + 200
        best = 0.0
        for c in cands:
            if len(c) > cap:
                continue
            try:
                v = teds.evaluate(c, g_html)
            except Exception:
                v = 0.0
            if v > best:
                best = v
        scores.append(best)
        if best >= 0.8:
            n_reconstructed += 1
    return {"teds": sum(scores) / len(scores) if scores else 0.0,
            "n_gold_merges": len(gold.merged_tables),
            "n_reconstructed": n_reconstructed,
            "n_pred_merges": sum(1 for e in pred.elements
                                 if e.merged and e.block_type == "table")}


def score_relation(pred: PredDoc, gold_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Optional table-truncation relation F1 via the benchmark's relation_utils. Best-effort:
    returns None on any plumbing/shape issue (this metric is disabled in their default config)."""
    try:
        L = _load_leaf()
        ru = L["relation_utils"]
        gt_pairs, _groups, _anno = ru.extract_table_truncated_info(gold_entry)
        all_gt_tables = sorted(
            [it for it in gold_entry.get("layout_dets", []) if it.get("category_type") == "table"],
            key=lambda x: (x.get("page_id", 0), x.get("order", float("inf"))))
        pred_tables = [{"content": e.html} for e in pred.elements
                       if e.block_type == "table" and e.html]
        if not all_gt_tables or (not gt_pairs and not pred_tables):
            return None
        pred_norm = [ru._norm_table(pt["content"]) for pt in pred_tables]
        pred_pairs, _detail = ru.detect_pred_merges(all_gt_tables, pred_norm, match_threshold=0.15)
        if not gt_pairs and not pred_pairs:
            return None
        m = ru.compute_relation_f1(gt_pairs, pred_pairs)
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"f1": f1, "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "gt_pairs": len(gt_pairs), "pred_pairs": len(pred_pairs)}
    except Exception:
        return None


# ============================================================
# ORCHESTRATION
# ============================================================

def _domain_map(stems) -> Dict[str, str]:
    """stem -> domain (the stem prefix, config.domain_for), for the per-domain breakdown of the report."""
    return {s: (config.domain_for(s) or "unknown") for s in stems}


def apply_table_text_redirect(pred: PredDoc, match: TableMatch) -> None:
    """Redirect every predicted table that won no gold table into the text stream, in place — the
    paper's ``unmatch_table_pred -> text`` behaviour.

    Chandra OCR routinely serializes borderless, column-aligned prose (tables of contents, index pages,
    aligned lists — anything "structured like a table" without rules/borders) as a Markdown/HTML table,
    and our extractor faithfully carries it through as a ``table`` block. When the gold has no matching
    table, that block is really text: excluded from the text metric (tables route to ``html``) and
    unscored by the table metric — double-penalising a *correct* read (worst case a prospectus table of
    contents). Flipping each unmatched pred table to a ``text_block`` (carrying its flattened
    ``table_text``) is made once, here, so it propagates coherently to every metric. Matched tables (they
    won a gold in ``match_tables``) are untouched.
    """
    for e in match.unmatched_pred:
        e.block_type = "text_block"
        e.category = BLOCK_TO_CATEGORY["text_block"]
        e.text = e.table_text
        e.html = ""


def score_table_diag(match: TableMatch) -> Optional[Dict[str, Any]]:
    """OUR diagnostic (NOT paper-comparable): best-match precision/recall/F1 on table COUNT, plus the
    matched-pair TEDS quality. A pair counts as a true match iff its TEDS >= ``TABLE_DIAG_MATCH_TEDS``.
    Unlike the headline metric this surfaces over-prediction (precision<1), misses (recall<1) AND
    misclassification (an assigned pair below the threshold — e.g. a TOC forced onto a real gold table on
    an equal-count doc, which the headline TEDS hides). None when the doc has no tables on either side."""
    if match.n_gold == 0 and match.n_pred == 0:
        return None
    true_matches = sum(1 for t in match.pair_teds if t >= TABLE_DIAG_MATCH_TEDS)
    precision = true_matches / match.n_pred if match.n_pred else 0.0
    recall = true_matches / match.n_gold if match.n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    matched_teds = sum(match.pair_teds) / len(match.pair_teds) if match.pair_teds else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "matched_teds": matched_teds,
            "n_gold": match.n_gold, "n_pred": match.n_pred,
            "n_true_matches": true_matches, "n_assigned": len(match.pair_teds)}


def evaluate_doc(doc: str, fd: Dict[str, Any], gold_entry: Dict[str, Any],
                 work_dir: Optional[Path] = None) -> Dict[str, Any]:
    pred = build_pred_doc(fd, doc)
    gold = build_gold_doc(gold_entry)
    match = match_tables(pred, gold)  # paper-faithful best-match of pred tables to gold tables
    apply_table_text_redirect(pred, match)  # unmatched pred tables' text -> the text metric (paper)
    text_match = match_text(pred, gold)  # per-page text-block matching (AFTER redirect); feeds diag+order
    return {
        "doc": doc,
        "text": score_text(pred, gold),
        "text_diag": score_text_diag(text_match),
        "equation": score_equations(pred, gold, doc, work_dir),        # paper CDM (or honest edit fallback)
        "equation_diag": score_equations_diag(pred, gold),             # ours (edit distance), always present
        "table": score_tables(pred, gold, match),
        "table_diag": score_table_diag(match),
        "heads": score_heads(pred, gold),
        "figures": score_figures(pred, gold),
        "reading_order": score_reading_order(text_match),
        "merged_text": score_merged_text(pred, gold),
        "merged_table": score_merged_table(pred, gold),
        "relation": score_relation(pred, gold_entry),
        "counts": {
            "gold": {"text": len(gold.text_elems), "tables": len(gold.tables),
                     "equations": len(gold.equations), "figures": len(gold.figures),
                     "titles": gold.n_titles, "merged_text": len(gold.merged_text),
                     "merged_tables": len(gold.merged_tables)},
            "pred": {"elements": len(pred.elements), "figures": len(pred.figures),
                     "heads": len(pred.head_md)},
        },
    }


# metric -> (path into the per-doc dict, "higher is better" primary key)
_PRIMARY = [
    ("text", ["text", "score"]),
    ("text_diag_f1", ["text_diag", "f1"]),
    ("equation", ["equation", "score"]),
    ("equation_diag", ["equation_diag", "score"]),
    ("table", ["table", "teds"]),
    ("table_struct", ["table", "teds_struct"]),
    ("table_diag_f1", ["table_diag", "f1"]),
    ("heads", ["heads", "score"]),
    ("figure_f1", ["figures", "f1"]),
    ("reading_order", ["reading_order", "score"]),
    ("merged_text", ["merged_text", "score"]),
    ("merged_table", ["merged_table", "teds"]),
    ("relation_f1", ["relation", "f1"]),
]


def _dig(d: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or cur.get(k) is None:
            return None
        cur = cur[k]
    return float(cur) if isinstance(cur, (int, float)) else None


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


# reading_order follows the paper's corpus micro-average (Sum edits / Sum uppers), NOT the macro
# mean-over-docs used for every other metric. ``_micro`` overrides its ``overall``/``by_domain`` value
# (the per-doc ``score`` in the _PRIMARY table is the doc-level micro, kept for the per-document view).
# ``text`` is the whole-string smoothie -> plain macro mean-over-docs like the rest (NOT micro here).
_MICRO_METRICS = ("reading_order",)


def _micro(docs: List[Dict[str, Any]], metric: str) -> Optional[float]:
    num = sum(d[metric]["edit_num"] for d in docs if d.get(metric) and "edit_num" in d[metric])
    den = sum(d[metric]["upper"] for d in docs if d.get(metric) and "upper" in d[metric])
    return (1.0 - num / den) if den else None


def _equation_micro(docs: List[Dict[str, Any]]) -> Optional[float]:
    """Paper CDM aggregation: micro-mean of pair F1 over ALL formula pairs (Sum sum_f1 / Sum n_pairs),
    matching metrics/cal_metric.call_CDM (a sample-level, not per-doc, average). Returns None unless EVERY
    doc with equations was scored by CDM — so an edit-fallback (or mixed) run keeps the macro mean-over-docs
    from ``_PRIMARY`` and the report labels it honestly rather than presenting it as CDM."""
    eqs = [d["equation"] for d in docs if d.get("equation")]
    if not eqs or not all(e.get("method") == "cdm" and "n_pairs" in e for e in eqs):
        return None
    num = sum(float(e.get("sum_f1", 0.0)) for e in eqs)
    den = sum(int(e.get("n_pairs", 0)) for e in eqs)
    return (num / den) if den else None


def _aggregate(per_doc: List[Dict[str, Any]], domain_of: Dict[str, str]) -> Dict[str, Any]:
    overall: Dict[str, Optional[float]] = {}
    for name, path in _PRIMARY:
        overall[name] = _mean([v for v in (_dig(d, path) for d in per_doc) if v is not None])
    for m in _MICRO_METRICS:
        overall[m] = _micro(per_doc, m)
    _eqm = _equation_micro(per_doc)                     # CDM: micro over all pairs (else keep the macro mean)
    if _eqm is not None:
        overall["equation"] = _eqm
    by_domain: Dict[str, Dict[str, Optional[float]]] = {}
    for d in per_doc:
        by_domain.setdefault(domain_of.get(d["doc"], "unknown"), {})
    for dom in by_domain:
        docs = [d for d in per_doc if domain_of.get(d["doc"], "unknown") == dom]
        for name, path in _PRIMARY:
            by_domain[dom][name] = _mean([v for v in (_dig(x, path) for x in docs) if v is not None])
        for m in _MICRO_METRICS:
            by_domain[dom][m] = _micro(docs, m)
        _eqm_d = _equation_micro(docs)
        if _eqm_d is not None:
            by_domain[dom]["equation"] = _eqm_d
    return {"overall": overall, "by_domain": by_domain, "n_docs": len(per_doc)}


def _resolve_cdm_work_dir(run_dir: Path) -> Path:
    """Where CDM renders its scratch tex/pdf/png. An explicit ``config.EQUATION_CDM_WORK_DIR`` (space-free)
    wins; else the default under the run dir. The override exists because the vendored CDM renderer shells
    out to xelatex/ImageMagick with UNQUOTED paths and this repo lives under a path containing spaces — a
    scratch dir with spaces makes every render silently fail -> F1 0."""
    override = getattr(config, "EQUATION_CDM_WORK_DIR", None)
    return Path(override) if override else run_dir / "_mpdocbench" / "_cdm_work"


def evaluate_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    gold_index = load_gold_index()
    domain_of = _domain_map(gold_index)

    fd_paths = sorted(run_dir.glob("*/*/final_document.json"))
    single = False
    if not fd_paths:
        fd_paths = sorted(run_dir.glob("*/final_document.json"))
        single = True

    cdm_work = _resolve_cdm_work_dir(run_dir)  # CDM render scratch (per-doc subdirs; cleaned unless kept)
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[str] = []
    for fd_path in fd_paths:
        doc = fd_path.parent.name if single else fd_path.parent.parent.name
        model = "default" if single else fd_path.parent.name
        gold_entry = gold_index.get(doc)
        if gold_entry is None:
            skipped.append(doc)
            continue
        try:
            fd = json.loads(fd_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            skipped.append(doc)
            continue
        by_model.setdefault(model, []).append(evaluate_doc(doc, fd, gold_entry, cdm_work))

    results = {"run": run_dir.name, "skipped_no_gold": sorted(set(skipped)), "models": {}}
    for model, per_doc in by_model.items():
        per_doc.sort(key=lambda d: d["doc"])
        results["models"][model] = {"per_doc": per_doc,
                                    "aggregate": _aggregate(per_doc, domain_of)}

    out_dir = run_dir / "_mpdocbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "weak_spots.md").write_text(
        render_weak_spots(results, domain_of), encoding="utf-8")
    return results


# ============================================================
# WEAK-SPOT REPORT (plain tone)
# ============================================================

_METRIC_LABELS = [
    ("text", "Text"), ("text_diag_f1", "TxtDiagF1"), ("table", "Table"), ("heads", "Heading"),
    ("figure_f1", "Figure"), ("equation", "Equation"), ("equation_diag", "EqDiag"),
    ("reading_order", "ReadOrder"), ("merged_text", "MrgText"), ("merged_table", "MrgTable"),
    ("table_diag_f1", "TblDiagF1"),
]


def _equation_method(per_doc: List[Dict[str, Any]]) -> Optional[str]:
    """The scoring method behind the `equation` column for this model: 'cdm', 'edit', 'mixed', or None
    (no equations). Drives the honest column label so an edit-fallback number is never shown as CDM."""
    methods = {d["equation"]["method"] for d in per_doc
               if d.get("equation") and d["equation"].get("method")}
    if not methods:
        return None
    if methods == {"cdm"}:
        return "cdm"
    if methods == {"edit"}:
        return "edit"
    return "mixed"


def _fmt(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def render_weak_spots(results: Dict[str, Any], domain_of: Dict[str, str]) -> str:
    lines: List[str] = ["# MPDocBench weak-spot report", ""]
    if results.get("skipped_no_gold"):
        lines += [f"> Skipped (no gold match): {', '.join(results['skipped_no_gold'])}", ""]
    if not results.get("models"):
        lines += ["No scored documents found in this run.", ""]
        return "\n".join(lines)

    for model, mres in results["models"].items():
        agg = mres["aggregate"]
        lines += [f"## Model: {model}  ({agg['n_docs']} docs)", ""]
        # Honest `equation` column label: paper CDM only when EVERY scored doc used CDM; otherwise mark the
        # fallback so an edit-distance number is never presented under the CDM name.
        eq_method = _equation_method(mres["per_doc"])
        eq_label = {"cdm": "Equation(CDM)", "edit": "Equation(edit!)",
                    "mixed": "Equation(mixed!)"}.get(eq_method, "Equation")
        labels = [(k, eq_label if k == "equation" else lbl) for k, lbl in _METRIC_LABELS]
        if eq_method in ("edit", "mixed"):
            fb = [d["doc"] for d in mres["per_doc"]
                  if d.get("equation") and d["equation"].get("method") == "edit"]
            lines += [f"> ⚠ Equation = edit-distance fallback, NOT paper CDM "
                      f"(CDM toolchain unavailable){' for: ' + ', '.join(sorted(fb)) if fb else ''}. "
                      f"`EqDiag` is always the edit-distance diagnostic.", ""]
        # headline: metric x domain
        header = "| Domain | " + " | ".join(lbl for _, lbl in labels) + " |"
        sep = "|" + "---|" * (len(labels) + 1)
        lines += [header, sep]
        row_all = "| **all** | " + " | ".join(
            _fmt(agg["overall"].get(k)) for k, _ in labels) + " |"
        lines.append(row_all)
        for dom in sorted(agg["by_domain"]):
            dm = agg["by_domain"][dom]
            lines.append("| " + dom + " | " + " | ".join(
                _fmt(dm.get(k)) for k, _ in labels) + " |")
        lines.append("")

        # where it struggles
        lines += ["### Where it struggles", ""]
        struggles = _find_struggles(mres["per_doc"], domain_of)
        if not struggles:
            lines += ["Nothing flagged above threshold.", ""]
        else:
            for s in struggles:
                lines.append(f"- {s}")
            lines.append("")

        # per-doc table
        lines += ["<details><summary>Per-document scores</summary>", "",
                  "| Doc | Domain | " + " | ".join(lbl for _, lbl in labels) + " |",
                  "|" + "---|" * (len(labels) + 2)]
        for d in mres["per_doc"]:
            vals = " | ".join(_fmt(_dig(d, dict(_PRIMARY)[k])) for k, _ in labels)
            lines.append(f"| {d['doc']} | {domain_of.get(d['doc'], '?')} | {vals} |")
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def _find_struggles(per_doc: List[Dict[str, Any]], domain_of: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for d in per_doc:
        doc = d["doc"]
        dom = domain_of.get(doc, "?")
        fig = d.get("figures")
        if fig and fig["gt_count"] >= 3 and fig["recall"] < 0.5:
            out.append(f"**{doc}** ({dom}): figure recall {fig['recall']:.2f} "
                       f"({fig['matched']}/{fig['gt_count']} gold figures matched) — layout figure detection gap.")
        tbl = d.get("table")
        if tbl and tbl["n_gold"] >= 2 and tbl["teds"] < 0.5:
            out.append(f"**{doc}** ({dom}): table TEDS {tbl['teds']:.2f} over {tbl['n_gold']} gold "
                       f"tables (pred {tbl['n_pred']}) — table structure/reading.")
        mt = d.get("merged_text")
        if mt and mt["n_gold_merges"] >= 1 and mt["score"] < 0.5:
            out.append(f"**{doc}** ({dom}): cross-page text merge {mt['score']:.2f} "
                       f"({mt['n_pred_merges']} pred vs {mt['n_gold_merges']} gold merges) — continuation miss.")
        mtb = d.get("merged_table")
        if mtb and mtb["n_gold_merges"] >= 1 and mtb["teds"] < 0.5:
            out.append(f"**{doc}** ({dom}): cross-page table merge {mtb['teds']:.2f} "
                       f"({mtb['n_pred_merges']} pred vs {mtb['n_gold_merges']} gold) — table continuation miss.")
        txt = d.get("text")
        if txt and txt["gold_chars"] > 200 and txt["score"] < 0.5:
            out.append(f"**{doc}** ({dom}): text edit score {txt['score']:.2f} "
                       f"({txt['pred_chars']} pred vs {txt['gold_chars']} gold chars) — text extraction gap.")
        tdg = d.get("text_diag")
        if tdg and tdg["n_gold"] >= 5 and tdg["recall"] < 0.6:
            out.append(f"**{doc}** ({dom}): text segmentation recall {tdg['recall']:.2f} "
                       f"({tdg['n_true_matches']} of {tdg['n_gold']} gold text blocks matched, "
                       f"{tdg['n_pred']} pred) — under-segmentation (blocks merged together).")
        if tdg and tdg["n_gold"] >= 5 and tdg["precision"] < 0.6 and tdg["n_pred"] > tdg["n_gold"]:
            out.append(f"**{doc}** ({dom}): text segmentation precision {tdg['precision']:.2f} "
                       f"({tdg['n_pred']} pred vs {tdg['n_gold']} gold text blocks) — over-production.")
        ro = d.get("reading_order")
        if ro and ro["n_gold"] >= 5 and ro["score"] < 0.5:
            out.append(f"**{doc}** ({dom}): reading-order {ro['score']:.2f} "
                       f"({ro['n_matched']} of {ro['n_gold']} gold text blocks matched & in order) — order/coverage.")
        hd = d.get("heads")
        if hd and hd["n_gold_titles"] >= 3 and hd["score"] < 0.35:
            out.append(f"**{doc}** ({dom}): heading-tree {hd['score']:.2f} "
                       f"({hd['n_pred_heads']} pred vs {hd['n_gold_titles']} gold titles) — heading level/nesting.")
        td = d.get("table_diag")
        if td and td["n_pred"] > td["n_gold"] and td["precision"] < 0.5:
            out.append(f"**{doc}** ({dom}): over-predicts tables — {td['n_pred']} pred vs {td['n_gold']} gold, "
                       f"table_diag precision {td['precision']:.2f} — blocks called tables that aren't (over-tabling).")
    return out


def main() -> int:
    from src.eval.adapt import latest_run_dir
    run_dir = latest_run_dir(config.RUNS_DIR)
    print(f"MPDocBench eval on {run_dir}")
    res = evaluate_run(run_dir)
    for model, mres in res["models"].items():
        print(f"  {model}: {json.dumps(mres['aggregate']['overall'])}")
    print(f"  -> {run_dir / '_mpdocbench'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
