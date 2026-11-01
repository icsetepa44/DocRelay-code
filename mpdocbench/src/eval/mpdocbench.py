from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DU = Path(__file__).resolve().parents[2]
if str(_DU) not in sys.path:
    sys.path.insert(0, str(_DU))

from src.core import config
from src.eval import adapt, official_text_edit
from src.eval.table_structure import to_grid, _assign


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

GOLD_TEXT_CATEGORIES = {"text_block", "title", "figure_caption", "table_caption", "caption",
                        "list", "footnote"}
PRED_TEXT_BLOCK_TYPES = {"text_block", "heading", "caption", "list", "code"}

TABLE_DIAG_MATCH_TEDS = 0.5

TEXT_DIAG_MATCH_EDIT = 0.5


_LEAF: Dict[str, Any] = {}


def _load_leaf() -> Dict[str, Any]:
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
            spec.loader.exec_module(mod)
        except ImportError as exc:
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
        mq = importlib.import_module("utils.match_quick")
        import Levenshtein as lev
    except ImportError as exc:
        raise ImportError(
            f"MPDocBench metric dependency missing: {exc}. "
            f"Install with:  pip install Levenshtein apted pylatexenc beautifulsoup4 evaluate"
        ) from exc

    def _match_quick(gt_items, pred_items, line_type, img_name):
        return mq.match_gt2pred_quick(gt_items, pred_items, line_type, img_name), None

    _LEAF.update(
        TEDS=tm.TEDS,
        HeadTEDS=hm.HeadTEDS,
        clean_string=dp.clean_string,
        textblock2unicode=dp.textblock2unicode,
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


def _edit(a: str, b: str) -> Tuple[int, int]:
    L = _load_leaf()
    upper = max(len(a), len(b))
    if upper == 0:
        return 0, 0
    return L["lev_distance"](a, b), upper


def _edit_score(a: str, b: str) -> float:
    dist, upper = _edit(a, b)
    return 1.0 if upper == 0 else 1.0 - dist / upper


def _parse_page_seq(block_id: str) -> Tuple[int, int]:
    try:
        parts = block_id.split(":")
        return int(parts[-2][1:]), int(parts[-1][1:])
    except (ValueError, IndexError, AttributeError):
        return 0, 0


def _grid_to_html(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    rows = []
    for row in grid:
        cells = "".join(f"<td>{'' if c is None else str(c)}</td>" for c in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _heading_level_text(unit: Dict[str, Any]) -> Tuple[int, str]:
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


@dataclass
class PredElement:
    unit_id: str
    block_type: str
    category: Optional[str]
    page: int
    seq: int
    text: str = ""
    html: str = ""
    table_text: str = ""
    furniture: bool = False
    merged: bool = False


@dataclass
class PredDoc:
    elements: List[PredElement] = field(default_factory=list)
    head_md: List[str] = field(default_factory=list)
    n_pages: int = 0
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
            el.table_text = adapt._unit_text(u)
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

    page_universe = sorted({e.page for e in elements})
    return PredDoc(elements=elements, head_md=head_md,
                   n_pages=len(page_universe), page_universe=page_universe)


def pred_markdown(fd: Dict[str, Any]) -> str:
    items: List[Tuple[int, int, str]] = []
    for u in fd.get("units", []) or []:
        bt = u.get("block_type", "")
        if bt in DROP_BLOCK_TYPES or bt in FURNITURE_BLOCK_TYPES or bt == "figure":
            continue
        ids = u.get("source_block_ids") or []
        page, seq = _parse_page_seq(ids[0]) if ids else (0, 0)
        if bt == "heading":
            level, text = _heading_level_text(u)
            md = ("#" * max(1, int(level or 1)) + " " + text.strip()) if (text or "").strip() else ""
        elif bt == "table":
            md = _pred_table_html(u)
        elif bt == "equation":
            latex = adapt._unit_text(u).strip()
            md = latex if latex.startswith(("$$", "\\[")) else (f"$$\n{latex}\n$$" if latex else "")
        else:
            md = adapt._unit_text(u)
        if md and md.strip():
            items.append((page, seq, md.strip()))
    items.sort(key=lambda x: (x[0], x[1]))
    return "\n\n".join(md for _, _, md in items)


@dataclass
class GoldDoc:
    text_elems: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    head_input: Any = None
    n_titles: int = 0
    merged_text: List[Dict[str, Any]] = field(default_factory=list)
    merged_tables: List[Dict[str, Any]] = field(default_factory=list)
    page_universe: List[int] = field(default_factory=list)


_GOLD_INDEX: Dict[str, Dict[str, Any]] = {}


def load_gold_index() -> Dict[str, Dict[str, Any]]:
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
            elif cat in GOLD_TEXT_CATEGORIES:
                g.text_elems.append({"page_id": pid, "order": order, "category": cat,
                                     "text": it.get("text", "")})
    g.tables.sort(key=lambda x: (x["page_id"], x["order"]))
    g.text_elems.sort(key=lambda x: (x["page_id"], x["order"]))

    parent_son, head_nodes = _document_heads(entry)
    g.head_input = [parent_son, head_nodes]
    g.n_titles = len(head_nodes)

    g.merged_tables = [m for m in merged if m.get("is_merged") == "table"]
    g.merged_text = [m for m in merged if m.get("is_merged") == "text_block"]
    g.page_universe = sorted({d.get("page_id") for d in (entry.get("layout_dets") or [])
                              if d.get("page_id") is not None})
    return g


@dataclass
class TextMatch:
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
    matcher = _load_leaf()["match_gt2pred"]
    gold_entries = _gold_text_entries(gold)
    pred_entries = _pred_text_entries(pred)
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


def score_text_diag(match: TextMatch) -> Optional[Dict[str, Any]]:
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


@dataclass
class TableMatch:
    gold_scores: List[float]
    gold_scores_struct: List[float]
    pair_teds: List[float]
    matched_pred_idxs: set
    unmatched_pred: List["PredElement"]
    n_gold: int
    n_pred: int


def match_tables(pred: PredDoc, gold: GoldDoc) -> TableMatch:
    pred_tbl = [e for e in pred.elements if e.block_type == "table" and e.html]
    gold_htmls = [_norm_table_html(t.get("html", "")) for t in gold.tables]
    pred_htmls = [_norm_table_html(e.html) for e in pred_tbl]
    n_gold, n_pred = len(gold_htmls), len(pred_tbl)
    if n_gold == 0 or n_pred == 0:
        return TableMatch([0.0] * n_gold, [0.0] * n_gold, [], set(), list(pred_tbl), n_gold, n_pred)
    sim = [[_edit_score(g, p) for p in pred_htmls] for g in gold_htmls]
    gold_to_pred = {gi: pi for gi, pi in _assign(sim)}
    L = _load_leaf()
    teds = L["TEDS"](structure_only=False)
    teds_s = L["TEDS"](structure_only=True)
    gold_scores: List[float] = []
    gold_scores_struct: List[float] = []
    pair_teds: List[float] = []
    matched: set = set()
    for gi in range(n_gold):
        pi = gold_to_pred.get(gi)
        if pi is None:
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


def score_reading_order(match: TextMatch) -> Dict[str, Any]:
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


def _domain_map(stems) -> Dict[str, str]:
    return {s: (config.domain_for(s) or "unknown") for s in stems}


def apply_table_text_redirect(pred: PredDoc, match: TableMatch) -> None:
    for e in match.unmatched_pred:
        e.block_type = "text_block"
        e.category = BLOCK_TO_CATEGORY["text_block"]
        e.text = e.table_text
        e.html = ""


def score_table_diag(match: TableMatch) -> Optional[Dict[str, Any]]:
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


def evaluate_doc(doc: str, fd: Dict[str, Any], gold_entry: Dict[str, Any]) -> Dict[str, Any]:
    pred = build_pred_doc(fd, doc)
    gold = build_gold_doc(gold_entry)
    match = match_tables(pred, gold)
    apply_table_text_redirect(pred, match)
    text_match = match_text(pred, gold)
    return {
        "doc": doc,
        "text": official_text_edit.score(gold_entry, pred_markdown(fd),
                                         os.path.basename((gold_entry.get("page_info") or {}).get("image_path", ""))),
        "text_diag": score_text_diag(text_match),
        "table": score_tables(pred, gold, match),
        "table_diag": score_table_diag(match),
        "heads": score_heads(pred, gold),
        "reading_order": score_reading_order(text_match),
        "merged_text": score_merged_text(pred, gold),
        "merged_table": score_merged_table(pred, gold),
        "relation": score_relation(pred, gold_entry),
        "counts": {
            "gold": {"text": len(gold.text_elems), "tables": len(gold.tables),
                     "titles": gold.n_titles, "merged_text": len(gold.merged_text),
                     "merged_tables": len(gold.merged_tables)},
            "pred": {"elements": len(pred.elements), "heads": len(pred.head_md)},
        },
    }


_PRIMARY = [
    ("text", ["text", "score"]),
    ("text_diag_f1", ["text_diag", "f1"]),
    ("table", ["table", "teds"]),
    ("table_struct", ["table", "teds_struct"]),
    ("table_diag_f1", ["table_diag", "f1"]),
    ("heads", ["heads", "score"]),
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


_MICRO_METRICS = ("reading_order",)


def _micro(docs: List[Dict[str, Any]], metric: str) -> Optional[float]:
    num = sum(d[metric]["edit_num"] for d in docs if d.get(metric) and "edit_num" in d[metric])
    den = sum(d[metric]["upper"] for d in docs if d.get(metric) and "upper" in d[metric])
    return (1.0 - num / den) if den else None


def _aggregate(per_doc: List[Dict[str, Any]], domain_of: Dict[str, str]) -> Dict[str, Any]:
    overall: Dict[str, Optional[float]] = {}
    for name, path in _PRIMARY:
        overall[name] = _mean([v for v in (_dig(d, path) for d in per_doc) if v is not None])
    for m in _MICRO_METRICS:
        overall[m] = _micro(per_doc, m)
    by_domain: Dict[str, Dict[str, Optional[float]]] = {}
    for d in per_doc:
        by_domain.setdefault(domain_of.get(d["doc"], "unknown"), {})
    for dom in by_domain:
        docs = [d for d in per_doc if domain_of.get(d["doc"], "unknown") == dom]
        for name, path in _PRIMARY:
            by_domain[dom][name] = _mean([v for v in (_dig(x, path) for x in docs) if v is not None])
        for m in _MICRO_METRICS:
            by_domain[dom][m] = _micro(docs, m)
    return {"overall": overall, "by_domain": by_domain, "n_docs": len(per_doc)}


def evaluate_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    gold_index = load_gold_index()
    domain_of = _domain_map(gold_index)

    fd_paths = sorted(run_dir.glob("*/*/final_document.json"))
    single = False
    if not fd_paths:
        fd_paths = sorted(run_dir.glob("*/final_document.json"))
        single = True

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
        by_model.setdefault(model, []).append(evaluate_doc(doc, fd, gold_entry))

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


_METRIC_LABELS = [
    ("text", "Text"), ("text_diag_f1", "TxtDiagF1"), ("table", "Table"), ("heads", "Heading"),
    ("reading_order", "ReadOrder"), ("merged_text", "MrgText"), ("merged_table", "MrgTable"),
    ("table_diag_f1", "TblDiagF1"),
]


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
        labels = list(_METRIC_LABELS)
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

        lines += ["### Where it struggles", ""]
        struggles = _find_struggles(mres["per_doc"], domain_of)
        if not struggles:
            lines += ["Nothing flagged above threshold.", ""]
        else:
            for s in struggles:
                lines.append(f"- {s}")
            lines.append("")

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
        if txt and txt["upper"] > 200 and txt["score"] < 0.5:
            out.append(f"**{doc}** ({dom}): text edit {txt['edit_dist']:.2f} "
                       f"over {txt['n_blocks']} matched text blocks — text extraction gap.")
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
