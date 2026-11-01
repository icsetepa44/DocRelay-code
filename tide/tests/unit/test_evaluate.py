"""End-to-end tests for the evaluator (src/eval/evaluate.py): gold placeholders, regression identity with
the original evaluator, read-only inputs, and the legacy single-model run layout."""

import hashlib
import json
from pathlib import Path

from src.eval import adapt, evaluate


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _unit(uid, bt, content, parent=None):
    return {"unit_id": uid, "block_type": bt, "content": content, "parent_unit_id": parent,
            "source_block_ids": []}


# ---------------- FIX-1a / FIX-1c: gold placeholders ----------------

def test_gold_placeholder_strip_and_expand(tmp_path):
    gold = {
        "parsed_text": [
            {"type": "clause", "text": "1. Goods listed below are exempt. [table_1] Subject to conditions."},
            {"type": "clause", "text": "[TABLE_2]"},  # table-only clause
            {"type": "clause", "text": "2. See [Table Header: Serial | Name] for format."},  # annotator prose
            {"type": "metadata", "text": "Page 3", "metadata_category": "page"},
            {"type": "clause", "text": "3. Missing table [table_9] here."},
        ],
        "tables": [
            {"table_id": "table_1", "headers": ["HS", "Desc"], "rows": [["01.01", "Horses"]]},
            {"table_id": "table_2", "headers": ["A"], "rows": [["x"]]},
            {"table_id": "table_3", "headers": ["Z"], "rows": [["unref"]]},
        ],
    }
    g = evaluate.load_gold(_write(tmp_path / "g.json", gold))
    texts = [it["text"] for it in g["items"]]
    assert texts[0] == "1. Goods listed below are exempt.   Subject to conditions."
    assert "[Table Header: Serial | Name]" in texts[1]  # untouched
    assert len(g["items"]) == 3  # table-only clause dropped, page item excluded
    assert g["stats"]["table_only_items"] == 1
    assert g["stats"]["excluded_page_items"] == 1
    assert g["stats"]["unresolved_placeholders"] == 1
    assert g["stats"].get("strip_mismatch", 0) == 0
    full = g["full_text"]
    assert full.index("Horses") < full.index("Subject to conditions")  # expanded in place
    assert "unref" in full and "table_1" not in full.lower().replace(" ", "")


# ---------------- regression identity with the ORIGINAL evaluator ----------------

def _identity_case(tmp_path):
    """No tables, short text: none of the four fixes applies, so the scores must equal the original's."""
    gold = {"parsed_text": [
        {"type": "metadata", "text": "Government of the People's Republic of Bangladesh National Board of Revenue"},
        {"type": "clause", "text": "1. In exercise of the powers conferred by section 19 the Government is pleased to exempt the goods."},
        {"type": "clause", "text": "2. This notification shall come into force at once and remain in force until further orders."},
        {"type": "metadata", "text": "Page 1", "metadata_category": "page"},
        {"type": "metadata", "text": "By order of the President, Secretary"},
    ], "tables": []}
    fd = {"units": [
        _unit("m1", "heading", {"text": "Government of the Peoples Republic of Bangladesh National Board of Revenue"}),
        _unit("c1", "text_block", {"text": "1. In exercise of the powers conferred by section 19 the Government is pleased"}),
        _unit("c1a", "text_block", {"text": "to exempt the goods."}, parent="c1"),
        _unit("c2", "text_block", {"text": "2. This notification shall come into force at once and remain in force."}),
        _unit("c3", "text_block", {"text": "3. A spurious extra clause the gold does not have at all."}),
        _unit("m2", "header_footer_page_number", {"text": "By order of the President Secretary"}),
    ]}
    return _write(tmp_path / "gold.json", gold), fd


# Metrics of the earlier evaluator version on _identity_case (frozen regression fixture).
ORIGINAL_EVALUATOR_METRICS = {
    "cer_clause_mean": 0.1142, "classification_accuracy": 1.0, "clause_f1": 0.8, "clause_precision": 0.6667,
    "clause_recall": 1.0, "confusion": {"clause->clause": 2, "metadata->metadata": 2},
    "doc_edit_similarity": 0.8559, "duplication_ratio": 1.1538, "gold_clauses": 2, "gold_excluded_page_items": 1,
    "gold_metadata": 2, "gold_tables": 0, "mean_clause_bag_similarity": 0.9483,
    "mean_clause_edit_similarity": 0.8859, "mean_clause_text_similarity": 0.9483, "missed_gold_items": 0,
    "over_segmentation_factor": 1.5, "pred_clauses": 3, "pred_metadata": 2, "soft_clause_f1": 0.7682,
    "soft_clause_precision": 0.6781, "soft_clause_recall": 0.8859, "spurious_pred_items": 1,
    "table_cell_recall": None, "table_grits_con": None, "table_grits_top": None, "table_teds": None,
    "table_teds_struct": None, "token_coverage": 0.9038, "wer_clause_mean": 0.0938, "wer_document": 0.2692,
}


def test_regression_identity_with_the_original_evaluator(tmp_path):
    gold_path, fd = _identity_case(tmp_path)
    m = evaluate.evaluate_pair(evaluate.load_gold(gold_path), adapt.adapt_final_document(fd))
    for k, v in ORIGINAL_EVALUATOR_METRICS.items():
        assert m[k] == v, f"{k}: original {v} != now {m[k]}"


# ---------------- end-to-end on a tiny run ----------------

def _tree_hash(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _tiny_run(tmp_path, legacy=False):
    gold_src, fd = _identity_case(tmp_path)
    gold_root = tmp_path / "gold"
    _write(gold_root / "docA_output.json", json.loads(gold_src.read_text(encoding="utf-8")))
    run = tmp_path / "runs" / "r1"
    pair = run / "docA" if legacy else run / "docA" / "m"
    _write(pair / "final_document.json", fd)
    _write(pair / "pages" / "0001" / "extraction.json", {"blocks": []})
    return run, gold_root


def test_run_is_read_only_on_inputs_and_writes_only_eval(tmp_path):
    run, gold_root = _tiny_run(tmp_path)
    before_run, before_gold = _tree_hash(run), _tree_hash(gold_root)

    out = evaluate.evaluate_run(run, gold_root=gold_root, workers=1)

    assert out["integrity"]["inputs_unchanged"] is True and out["integrity"]["PASS"] is True
    assert _tree_hash(gold_root) == before_gold
    assert {k: v for k, v in _tree_hash(run).items() if not k.startswith("_eval/")} == before_run
    metrics = json.loads((run / "_eval" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluator"] == evaluate.EVALUATOR_VERSION
    assert metrics["results"]["m"]["docA"]["soft_clause_f1"] == ORIGINAL_EVALUATOR_METRICS["soft_clause_f1"]
    assert (run / "_eval" / "final_documents" / "docA" / "m" / "final_stitched_document.json").exists()
    assert sorted(p.name for p in (run / "_eval").iterdir()) == [
        "final_documents", "integrity.json", "metrics.json", "summary.md"]


def test_legacy_single_model_layout(tmp_path):
    run, gold_root = _tiny_run(tmp_path, legacy=True)
    out = evaluate.evaluate_run(run, gold_root=gold_root, workers=1)
    assert list(out["results"]) == [evaluate.LEGACY_MODEL_LABEL]
    assert out["results"][evaluate.LEGACY_MODEL_LABEL]["docA"]["gold_clauses"] == 2
