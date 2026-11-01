from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.core import config
from src.eval import adapt, official_text_edit
from src.eval.table_structure import score_tables

try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV
except ImportError:
    _RF_LEV = None


RUN_DIR: Optional[Path] = None
GOLD_ROOT = config.BENCHMARK_GOLD_ROOT
WORKERS = 6
VERIFY_INPUTS_UNCHANGED = True

EVALUATOR_VERSION = "evaluate_v2"
LEGACY_MODEL_LABEL = "document_understanding_v1"


EXCLUDE_GOLD_METADATA_CATEGORIES = {"page"}


_TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_RE = re.compile(r"[#*_>`|\-–—]+")
_WS_RE = re.compile(r"\s+")
PLACEHOLDER_RE = re.compile(r"\[\s*table[\s_-]*(\d+)\s*\]", re.IGNORECASE)


def normalize_prose(text: str, drop_tables: bool = True) -> str:
    text = str(text or "")
    if drop_tables:
        text = _TABLE_RE.sub(" ", text)
    text = _IMG_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _levenshtein_dp(a, b) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _levenshtein(a, b) -> int:
    if _RF_LEV is not None:
        return _RF_LEV.distance(a, b)
    return _levenshtein_dp(a, b)


def _table_key(raw: Any) -> Optional[str]:
    m = re.search(r"(\d+)", str(raw or ""))
    return f"table_{int(m.group(1))}" if m else None


def flatten_table(t: Dict[str, Any]) -> str:
    parts = [str(h) for h in t.get("headers", [])]
    for row in t.get("rows", []):
        parts.extend(str(c) for c in row)
    return " ".join(parts)


def load_gold(path: Path) -> Dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    tables = d.get("tables", []) or []
    table_text: Dict[str, str] = {}
    unkeyed: List[str] = []
    for t in tables:
        key = _table_key(t.get("table_id"))
        if key and key not in table_text:
            table_text[key] = flatten_table(t)
        else:
            unkeyed.append(flatten_table(t))

    items, full_parts = [], []
    referenced: set = set()
    stats = Counter()
    for it in d.get("parsed_text", []):
        raw = str(it.get("text", ""))
        if not raw.strip():
            continue
        if str(it.get("metadata_category") or "").strip().lower() in EXCLUDE_GOLD_METADATA_CATEGORIES:
            stats["excluded_page_items"] += 1
            continue
        phs = list(PLACEHOLDER_RE.finditer(raw))
        prose = PLACEHOLDER_RE.sub(" ", raw).strip()

        def _expand(m: re.Match) -> str:
            key = f"table_{int(m.group(1))}"
            if key in table_text:
                referenced.add(key)
                return f" {table_text[key]} "
            stats["unresolved_placeholders"] += 1
            return " "

        full_parts.append(PLACEHOLDER_RE.sub(_expand, raw))
        stats["placeholders"] += len(phs)
        if len(raw) - sum(len(m.group(0)) for m in phs) != len(PLACEHOLDER_RE.sub("", raw)):
            stats["strip_mismatch"] += 1
        if not prose:
            stats["table_only_items"] += 1
            continue
        items.append({"type": str(it.get("type", "")).strip().lower(), "text": prose})
    full_parts.extend(table_text[k] for k in table_text if k not in referenced)
    full_parts.extend(unkeyed)
    return {"items": items, "tables": tables, "full_text": " ".join(full_parts), "stats": dict(stats), "raw": d}


def gold_path_for(gold_root: Path, document_name: str) -> Optional[Path]:
    clean = document_name.rstrip(".")
    for name in (document_name, clean):
        for c in (Path(gold_root) / name / f"{name}_output.json",
                  Path(gold_root) / f"{name}_output.json",
                  Path(gold_root) / f"{name}.json"):
            if c.exists():
                return c
    return None


def _grid(table: Dict[str, Any]) -> List[List[str]]:
    rows = [[str(h) for h in table.get("headers") or []]] if table.get("headers") else []
    rows += [[str(c) for c in row] for row in table.get("rows") or []]
    return [row for row in rows if row]


def _grid_html(grid: List[List[str]]) -> str:
    if not grid:
        return ""
    rows = []
    for row in grid:
        cells = "".join(f"<td>{'' if c is None else str(c)}</td>" for c in row)
        rows.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(rows) + "</table>"


def _one_block(text: Optional[str]) -> str:
    return re.sub(r"\n\s*\n+", "\n", (text or "").strip())


def official_sample(doc: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    dets: List[Dict[str, Any]] = []
    order_of: Dict[Any, int] = {}
    for i, it in enumerate(raw.get("parsed_text", []) or []):
        page_item = str(it.get("metadata_category") or "").strip().lower() in EXCLUDE_GOLD_METADATA_CATEGORIES
        text = _one_block(PLACEHOLDER_RE.sub(" ", str(it.get("text") or "")))
        order_of[it.get("text_id")] = i + 1
        if text:
            dets.append({"anno_id": f"t{i}", "category_type": "page_number" if page_item else "text_block",
                         "order": i + 1, "page_id": it.get("page_start") or 0, "text": text})
    for k, table in enumerate(raw.get("tables", []) or []):
        grid = _grid(table)
        if grid:
            dets.append({"anno_id": f"tb{k}", "category_type": "table",
                         "order": order_of.get(table.get("parent_text_id"), 10 ** 6) + 0.5,
                         "page_id": table.get("page_start") or 0, "html": _grid_html(grid)})
    return {"layout_dets": dets, "extra": {"relation": []}, "page_info": {"image_path": f"{doc}.pdf"}}


def pred_markdown(pred: Dict[str, Any]) -> str:
    by_parent: Dict[Any, List[Dict[str, Any]]] = {}
    for table in pred.get("tables", []) or []:
        by_parent.setdefault(table.get("parent_index"), []).append(table)
    blocks: List[str] = []
    for i, item in enumerate(pred.get("contents", []) or []):
        text = _one_block(item.get("text"))
        if text:
            blocks.append(text)
        blocks += [_grid_html(_grid(t)) for t in by_parent.get(i, []) if _grid(t)]
    blocks += [_grid_html(_grid(t)) for t in by_parent.get(None, []) if _grid(t)]
    return "\n\n".join(blocks)


def evaluate_pair(gold: Dict[str, Any], pred: Dict[str, Any], doc: str = "doc") -> Dict[str, Any]:
    gold_items = gold["items"]
    tables = gold["tables"]
    table_shape = score_tables(tables, pred["tables"]) or {}
    return {
        "gold_clauses": sum(1 for x in gold_items if x["type"] == "clause"),
        "gold_metadata": sum(1 for x in gold_items if x["type"] == "metadata"),
        "gold_tables": len(tables),
        "text_edit": official_text_edit.score(official_sample(doc, gold["raw"]), pred_markdown(pred), f"{doc}.pdf"),
        "table_teds": table_shape.get("table_teds"),
    }


def process_pair(doc: str, model: str, fd_path: str, gold_path: str, out_root: str) -> Dict[str, Any]:
    pred = adapt.adapt_file(Path(fd_path))
    out_dir = Path(out_root) / "final_documents" / doc / model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_stitched_document.json").write_text(
        json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    gold = load_gold(Path(gold_path))
    metrics = evaluate_pair(gold, pred, doc)
    integrity = {
        "unrouted_fragments": pred["adapter_stats"].get("unrouted_fragments", 0),
        "gold_strip_mismatch": gold["stats"].get("strip_mismatch", 0),
        "gold_unresolved_placeholders": gold["stats"].get("unresolved_placeholders", 0),
    }
    return {"doc": doc, "model": model, "metrics": metrics, "integrity": integrity}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_inputs(paths: List[Path], desc: str) -> Dict[str, str]:
    return {str(p): _sha256(p) for p in tqdm(paths, desc=desc, unit="file", dynamic_ncols=True)}


def input_files(pairs: List[Tuple[str, str, Path, Path]]) -> List[Path]:
    files = set()
    for _, _, fd, gold in pairs:
        files.add(fd)
        files.add(gold)
        files.update((fd.parent / "pages").glob("*/extraction.json"))
    return sorted(files)


def render_summary(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    lines = ["# TIDE evaluation\n",
             "Per document: Text Edit = the official MPDocBench text edit distance (matched text blocks, summed "
             "edits / summed lengths; lower is better); Table TEDS = tree-edit similarity of the tables x 100, for "
             "documents with gold tables (higher is better). The run-level metrics are in `paper_metrics.json`.\n"]
    for model, docs in results.items():
        lines += [f"## {model}\n", "| Document | Text Edit | Table TEDS |", "|---|---|---|"]
        for doc, m in docs.items():
            text_edit, teds = m.get("text_edit"), m.get("table_teds")
            text_cell = "-" if text_edit is None else f"{text_edit['edit_dist']:.4f}"
            teds_cell = "-" if teds is None else f"{100 * teds:.2f}"
            lines.append(f"| {doc} | {text_cell} | {teds_cell} |")
        lines.append("")
    return "\n".join(lines)


def discover_pairs(run_dir: Path, gold_root: Path) -> Tuple[List[Tuple[str, str, Path, Path]], List[Dict]]:
    found = [(fd.parent.parent.name, fd.parent.name, fd) for fd in sorted(run_dir.glob("*/*/final_document.json"))]
    if not found:
        found = [(fd.parent.name, LEGACY_MODEL_LABEL, fd) for fd in sorted(run_dir.glob("*/final_document.json"))]
    pairs, skipped = [], []
    for doc, model, fd in found:
        if doc.startswith("_"):
            continue
        gold = gold_path_for(gold_root, doc)
        if gold is None:
            skipped.append({"document": doc, "model": model, "reason": "no gold found"})
            continue
        pairs.append((doc, model, fd, gold))
    return pairs, skipped


def evaluate_run(run_dir: Path, gold_root: Path = GOLD_ROOT, out_dir: Optional[Path] = None,
                 workers: int = WORKERS, verify_inputs: bool = VERIFY_INPUTS_UNCHANGED) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    out_dir = Path(out_dir) if out_dir else run_dir / config.EVAL_SUBDIR
    print(f"Run:  {run_dir}\nGold: {gold_root}\nOut:  {out_dir}", flush=True)

    pairs, skipped = discover_pairs(run_dir, Path(gold_root))
    print(f"{len(pairs)} (doc, model) pairs; {len(skipped)} skipped (no gold).", flush=True)
    inputs = input_files(pairs)
    before = hash_inputs(inputs, "hashing inputs") if verify_inputs else {}

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(doc, model, str(fd), str(gold), str(out_dir)) for doc, model, fd, gold in pairs]
    outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    bar = tqdm(total=len(jobs), desc="adapt+evaluate", unit="doc", dynamic_ncols=True)

    def _collect(res: Dict[str, Any]) -> None:
        outputs.append(res)
        bar.set_postfix_str(res["doc"][:28])
        bar.update(1)

    if workers <= 1:
        for job in jobs:
            try:
                _collect(process_pair(*job))
            except Exception as e:
                errors.append({"document": job[0], "model": job[1], "error": repr(e)})
                bar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pair, *job): job for job in jobs}
            for fut in as_completed(futs):
                try:
                    _collect(fut.result())
                except Exception as e:
                    job = futs[fut]
                    errors.append({"document": job[0], "model": job[1], "error": repr(e)})
                    bar.update(1)
    bar.close()

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    integrity_per_doc: Dict[str, Dict[str, Any]] = {}
    for o in sorted(outputs, key=lambda o: (o["model"], o["doc"])):
        results.setdefault(o["model"], {})[o["doc"]] = o["metrics"]
        integrity_per_doc[f"{o['model']}/{o['doc']}"] = o["integrity"]

    after = hash_inputs(inputs, "re-hashing inputs") if verify_inputs else {}
    changed = sorted(p for p in before if before[p] != after.get(p))

    (out_dir / "metrics.json").write_text(json.dumps(
        {"evaluator": EVALUATOR_VERSION, "results": results,
         "skipped": skipped, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(results), encoding="utf-8")

    checks = {
        "inputs_unchanged": (not changed) if verify_inputs else None,
        "inputs_hashed": len(before),
        "changed_inputs": changed,
        "eval_errors": len(errors),
        "docs_with_unrouted_fragments": [k for k, v in integrity_per_doc.items() if v["unrouted_fragments"]],
        "docs_with_gold_strip_mismatch": [k for k, v in integrity_per_doc.items() if v["gold_strip_mismatch"]],
        "docs_with_unresolved_placeholders": {k: v["gold_unresolved_placeholders"]
                                              for k, v in integrity_per_doc.items()
                                              if v["gold_unresolved_placeholders"]},
    }
    passed = (checks["inputs_unchanged"] is not False and not errors
              and not checks["docs_with_unrouted_fragments"] and not checks["docs_with_gold_strip_mismatch"])
    checks["PASS"] = passed
    (out_dir / "integrity.json").write_text(json.dumps(
        {"checks": checks, "per_doc": integrity_per_doc}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEvaluated {len(outputs)} pairs ({len(errors)} errors). Wrote {out_dir}")
    print(f"Integrity: {'PASS' if passed else 'FAIL'} — see {out_dir / 'integrity.json'}", flush=True)
    if changed:
        raise RuntimeError(f"INPUT FILES CHANGED during evaluation ({len(changed)}): {changed[:5]}")
    return {"results": results, "integrity": checks}


def latest_run_dir(runs_root: Path) -> Path:
    runs = [p for p in Path(runs_root).glob("run_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {runs_root}")
    return max(runs, key=lambda p: p.name)


def main() -> int:
    evaluate_run(RUN_DIR or latest_run_dir(config.RUNS_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
