from __future__ import annotations

import difflib
import hashlib
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from src.core import config
from src.eval import adapt
from src.eval.table_structure import score_tables

try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV
except ImportError:
    _RF_LEV = None


RUN_DIR: Optional[Path] = None
GOLD_ROOT = config.BENCHMARK_GOLD_ROOT
THRESHOLD = 0.5
WORKERS = 6
VERIFY_INPUTS_UNCHANGED = True

EVALUATOR_VERSION = "evaluate_v2"
LEGACY_MODEL_LABEL = "document_understanding_v1"


EXCLUDE_GOLD_METADATA_CATEGORIES = {"page"}

GOLD_TEXT_UNRELIABLE_DOCS: Dict[str, str] = {}

STRUCTURAL_COLS = {"over_segmentation_factor", "table_teds", "table_teds_struct",
                   "table_grits_con", "table_grits_top"}


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


def token_bag_similarity(a: str, b: str) -> float:
    ca, cb = Counter(tokenize(a)), Counter(tokenize(b))
    if not ca and not cb:
        return 1.0
    total = sum(ca.values()) + sum(cb.values())
    if total == 0:
        return 0.0
    return round(2 * sum((ca & cb).values()) / total, 4)


def token_similarity(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    if len(ta) > 400 or len(tb) > 400:
        return token_bag_similarity(a, b)
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


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


def error_rate(ref_seq, hyp_seq) -> float:
    if not ref_seq:
        return 0.0 if not hyp_seq else 1.0
    return round(_levenshtein(ref_seq, hyp_seq) / len(ref_seq), 4)


def edit_similarity(a: str, b: str, level: str = "char") -> float:
    seq_a = list(a) if level == "char" else tokenize(a)
    seq_b = list(b) if level == "char" else tokenize(b)
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    return round(1.0 - _levenshtein(seq_a, seq_b) / max(len(seq_a), len(seq_b)), 4)


def soft_clause_scores(gold_norm: List[str], pred_norm: List[str]) -> Tuple[float, float]:
    if not gold_norm or not pred_norm:
        return 0.0, 0.0

    def _sim(g: str, p: str) -> float:
        lg, lp = len(g), len(p)
        if lg == 0 and lp == 0:
            return 1.0
        if lg == 0 or lp == 0:
            return 0.0
        if min(lg, lp) / max(lg, lp) < 0.35:
            return 0.0
        return edit_similarity(g, p)

    sim = [[_sim(g, p) for p in pred_norm] for g in gold_norm]
    recall = statistics.fmean(max(row) for row in sim)
    precision = statistics.fmean(max(sim[gi][pi] for gi in range(len(gold_norm)))
                                 for pi in range(len(pred_norm)))
    return round(precision, 4), round(recall, 4)


def align_items(g_norm: List[str], p_norm: List[str], threshold: float = 0.5):
    used_pred: set = set()
    matches = []
    for gi in sorted(range(len(g_norm)), key=lambda i: -len(g_norm[i])):
        best_pi, best_s = -1, 0.0
        for pi in range(len(p_norm)):
            if pi in used_pred:
                continue
            s = token_similarity(g_norm[gi], p_norm[pi])
            if s > best_s:
                best_s, best_pi = s, pi
        if best_pi >= 0 and best_s >= threshold:
            used_pred.add(best_pi)
            matches.append((gi, best_pi, round(best_s, 4)))
    return matches, used_pred


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else round(2 * p * r / (p + r), 4)


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
    return {"items": items, "tables": tables, "full_text": " ".join(full_parts), "stats": dict(stats)}


def gold_path_for(gold_root: Path, document_name: str) -> Optional[Path]:
    clean = document_name.rstrip(".")
    for name in (document_name, clean):
        for c in (Path(gold_root) / name / f"{name}_output.json",
                  Path(gold_root) / f"{name}_output.json",
                  Path(gold_root) / f"{name}.json"):
            if c.exists():
                return c
    return None


def evaluate_pair(gold: Dict[str, Any], pred: Dict[str, Any], threshold: float = THRESHOLD) -> Dict[str, Any]:
    gold_items = gold["items"]
    tables = gold["tables"]
    pred_items = [x for x in pred["contents"] if x["type"] in ("clause", "metadata") and x["text"].strip()]

    g_norm = [normalize_prose(x["text"]) for x in gold_items]
    p_norm = [normalize_prose(x["text"]) for x in pred_items]
    matches, used_pred = align_items(g_norm, p_norm, threshold)

    gold_clause = [i for i, x in enumerate(gold_items) if x["type"] == "clause"]
    pred_clause = [i for i, x in enumerate(pred_items) if x["type"] == "clause"]
    gold_meta = [i for i, x in enumerate(gold_items) if x["type"] == "metadata"]
    pred_meta = [i for i, x in enumerate(pred_items) if x["type"] == "metadata"]

    clause_pairs = [(gi, pi, s) for gi, pi, s in matches
                    if gold_items[gi]["type"] == "clause" and pred_items[pi]["type"] == "clause"]
    clause_recall = round(len({gi for gi, _, _ in clause_pairs}) / len(gold_clause), 4) if gold_clause else 0.0
    clause_precision = round(len({pi for _, pi, _ in clause_pairs}) / len(pred_clause), 4) if pred_clause else 0.0
    mean_clause_sim = round(statistics.fmean([s for _, _, s in clause_pairs]), 4) if clause_pairs else 0.0
    mean_clause_bag = round(statistics.fmean(
        [token_bag_similarity(g_norm[gi], p_norm[pi]) for gi, pi, _ in clause_pairs]), 4) if clause_pairs else 0.0

    soft_p, soft_r = soft_clause_scores([g_norm[i] for i in gold_clause], [p_norm[i] for i in pred_clause])
    mean_clause_edit = round(statistics.fmean(
        [edit_similarity(g_norm[gi], p_norm[pi]) for gi, pi, _ in clause_pairs]), 4) if clause_pairs else 0.0

    confusion = Counter((gold_items[gi]["type"], pred_items[pi]["type"]) for gi, pi, _ in matches)
    class_acc = round(sum(v for (gt, pt), v in confusion.items() if gt == pt) / len(matches), 4) if matches else 0.0
    matched_g = {m[0] for m in matches}
    missed = [gi for gi in range(len(gold_items)) if gi not in matched_g]
    spurious = [pi for pi in range(len(pred_items)) if pi not in used_pred]

    g_doc = normalize_prose(gold["full_text"], drop_tables=False)
    p_doc = normalize_prose(" ".join(x["full_text"] for x in pred["contents"]), drop_tables=False)
    wer_document = error_rate(tokenize(g_doc), tokenize(p_doc))
    doc_edit_similarity = edit_similarity(g_doc, p_doc, level="char")
    clause_wers = [error_rate(tokenize(g_norm[gi]), tokenize(p_norm[pi])) for gi, pi, _ in clause_pairs]
    clause_cers = [error_rate(list(g_norm[gi]), list(p_norm[pi])) for gi, pi, _ in clause_pairs]

    gc, pc = Counter(tokenize(g_doc)), Counter(tokenize(p_doc))
    gold_total = sum(gc.values())
    inter = sum(min(v, pc.get(t, 0)) for t, v in gc.items())

    pred_tokens = set(pc)
    cell_hits = cell_total = 0
    for t in tables:
        for cell in [c for row in t.get("rows", []) for c in row] + list(t.get("headers", [])):
            ct = tokenize(normalize_prose(str(cell)))
            if not ct:
                continue
            cell_total += 1
            if sum(1 for tok in ct if tok in pred_tokens) / len(ct) >= 0.6:
                cell_hits += 1
    table_shape = score_tables(tables, pred["tables"]) or {}

    return {
        "gold_clauses": len(gold_clause),
        "pred_clauses": len(pred_clause),
        "gold_metadata": len(gold_meta),
        "pred_metadata": len(pred_meta),
        "clause_precision": clause_precision,
        "clause_recall": clause_recall,
        "clause_f1": f1(clause_precision, clause_recall),
        "soft_clause_precision": soft_p,
        "soft_clause_recall": soft_r,
        "soft_clause_f1": f1(soft_p, soft_r),
        "over_segmentation_factor": round(len(pred_clause) / len(gold_clause), 4) if gold_clause else None,
        "wer_document": wer_document,
        "wer_clause_mean": round(statistics.fmean(clause_wers), 4) if clause_wers else None,
        "cer_clause_mean": round(statistics.fmean(clause_cers), 4) if clause_cers else None,
        "mean_clause_text_similarity": mean_clause_sim,
        "mean_clause_bag_similarity": mean_clause_bag,
        "mean_clause_edit_similarity": mean_clause_edit,
        "doc_edit_similarity": doc_edit_similarity,
        "classification_accuracy": class_acc,
        "confusion": {f"{gt}->{pt}": v for (gt, pt), v in sorted(confusion.items())},
        "missed_gold_items": len(missed),
        "spurious_pred_items": len(spurious),
        "gold_excluded_page_items": gold["stats"].get("excluded_page_items", 0),
        "token_coverage": round(inter / gold_total, 4) if gold_total else 0.0,
        "duplication_ratio": round(sum(pc.values()) / gold_total, 4) if gold_total else 0.0,
        "gold_tables": len(tables),
        "table_cell_recall": round(cell_hits / cell_total, 4) if cell_total else None,
        "table_teds": table_shape.get("table_teds"),
        "table_teds_struct": table_shape.get("table_teds_struct"),
        "table_grits_con": table_shape.get("table_grits_con"),
        "table_grits_top": table_shape.get("table_grits_top"),
        "pred_tables": len(pred["tables"]),
        "pred_table_fragments": pred.get("adapter_stats", {}).get("table_fragments", 0),
        "pred_recovered_markdown_fragments": pred.get("adapter_stats", {}).get("recovered_fragments", 0),
        "gold_table_placeholders": gold["stats"].get("placeholders", 0),
        "gold_table_only_items": gold["stats"].get("table_only_items", 0),
        "gold_unresolved_placeholders": gold["stats"].get("unresolved_placeholders", 0),
    }


def process_pair(doc: str, model: str, fd_path: str, gold_path: str, out_root: str,
                 threshold: float) -> Dict[str, Any]:
    pred = adapt.adapt_file(Path(fd_path))
    out_dir = Path(out_root) / "final_documents" / doc / model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "final_stitched_document.json").write_text(
        json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")

    gold = load_gold(Path(gold_path))
    metrics = evaluate_pair(gold, pred, threshold)
    metrics["gold_text_reliable"] = doc not in GOLD_TEXT_UNRELIABLE_DOCS
    metrics["gold_quality_note"] = GOLD_TEXT_UNRELIABLE_DOCS.get(doc)
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


_SUMMARY_COLS = [
    ("clause_f1", "Clause F1"), ("soft_clause_f1", "SoftF1"), ("clause_precision", "Cl.Prec"),
    ("clause_recall", "Cl.Rec"), ("over_segmentation_factor", "OverSeg"), ("wer_document", "WER"),
    ("cer_clause_mean", "CER"), ("mean_clause_text_similarity", "TextSim"),
    ("mean_clause_bag_similarity", "BagSim"), ("mean_clause_edit_similarity", "EditSim"),
    ("classification_accuracy", "ClsAcc"), ("token_coverage", "Coverage"), ("duplication_ratio", "Dup"),
    ("table_cell_recall", "TblRecall"), ("table_teds", "TEDS"), ("table_teds_struct", "TEDS-S"),
    ("table_grits_con", "GriTS-C"), ("table_grits_top", "GriTS-T"),
]


def _avg(values) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(statistics.fmean(vals), 4) if vals else None


def render_summary(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    lines = ["# Benchmark evaluation\n",
             f"_Evaluator `{EVALUATOR_VERSION}`: FIX-1 prose-only clauses (tables scored "
             "by table metrics), FIX-2 exact edit distance (no difflib autojunk), FIX-3 markdown/html-only "
             "text recovered, FIX-4 SoftF1 length gate on normalized text._\n",
             "## Headline — averaged across documents (per model)\n",
             "_Text-similarity columns average over gold-reliable docs only; "
             f"{sorted(STRUCTURAL_COLS)} average over all docs. 'Docs' shows all (reliable)._\n"]
    lines += ["| Model | Docs | " + " | ".join(h for _, h in _SUMMARY_COLS) + " |",
              "|" + "---|" * (2 + len(_SUMMARY_COLS))]
    for model, docs in results.items():
        reliable = [m for m in docs.values() if m.get("gold_text_reliable", True)]
        row = [model, f"{len(docs)} ({len(reliable)})"]
        for key, _ in _SUMMARY_COLS:
            pool = docs.values() if key in STRUCTURAL_COLS else reliable
            row.append(str(_avg([m.get(key) for m in pool])))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Per-document detail\n")
    for model, docs in results.items():
        lines.append(f"### {model}\n")
        lines.append("| Document | gCl/pCl | " + " | ".join(h for _, h in _SUMMARY_COLS) + " | miss/spur |")
        lines.append("|" + "---|" * (3 + len(_SUMMARY_COLS)))
        for doc, m in docs.items():
            label = doc if m.get("gold_text_reliable", True) else f"⚠ {doc}"
            row = [label, f"{m['gold_clauses']}/{m['pred_clauses']}"] + [str(m.get(k)) for k, _ in _SUMMARY_COLS]
            row.append(f"{m['missed_gold_items']}/{m['spurious_pred_items']}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    flagged = {doc: m.get("gold_quality_note") or "gold text flagged unreliable."
               for docs in results.values() for doc, m in docs.items() if not m.get("gold_text_reliable", True)}
    if flagged:
        lines.append("## Gold-quality flags (E2c)\n")
        lines += [f"- ⚠ **{doc}**: {note} Its text-similarity metrics are EXCLUDED from the headline "
                  f"average (structural columns {sorted(STRUCTURAL_COLS)} still include it)."
                  for doc, note in flagged.items()]
        lines.append("")

    lines += ["## Measurement notes\n",
              "- **SoftF1** is the graded clause F1: each gold clause is matched to its most similar predicted "
              "clause by character edit similarity, and each predicted clause to its most similar gold clause "
              "(no 0.5 cutoff; a pair whose lengths differ by more than 0.35x scores 0). **Clause F1** is the "
              "strict version (token similarity ≥ 0.5, one-to-one). Clauses are scored on prose only; tables "
              "are scored by TblRecall / TEDS / GriTS and count in the document-level WER and Coverage.",
              "- **EditSim** is the typo-tolerant fidelity of the strictly matched clauses (not robust to word "
              "reordering); **BagSim** is the order-insensitive companion to **TextSim**.",
              "- **E2a** excludes gold `metadata_category=\"page\"` items (running headers/footers stripped from "
              "the pipeline's input); `gold_excluded_page_items` in metrics.json counts them per pair.",
              ""]
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
                 workers: int = WORKERS, threshold: float = THRESHOLD,
                 verify_inputs: bool = VERIFY_INPUTS_UNCHANGED) -> Dict[str, Any]:
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
    jobs = [(doc, model, str(fd), str(gold), str(out_dir), threshold) for doc, model, fd, gold in pairs]
    outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    bar = tqdm(total=len(jobs), desc="adapt+evaluate", unit="doc", dynamic_ncols=True)

    def _collect(res: Dict[str, Any]) -> None:
        outputs.append(res)
        vals = [o["metrics"]["soft_clause_f1"] for o in outputs]
        bar.set_postfix_str(f"{res['doc'][:28]} | mean SoftF1 {statistics.fmean(vals):.4f}")
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
        {"threshold": threshold, "evaluator": EVALUATOR_VERSION, "results": results,
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
    for model, docs in results.items():
        print(f"  {model}: mean SoftF1 {_avg([m['soft_clause_f1'] for m in docs.values()])} over {len(docs)} docs")
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
