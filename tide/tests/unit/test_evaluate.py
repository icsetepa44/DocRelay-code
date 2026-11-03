import hashlib
import json
from pathlib import Path

import pytest

from src.core import config
from src.eval import adapt, evaluate

needs_mpdocbench_repo = pytest.mark.skipif(not (config.MPDOCBENCH_REPO / "dataset" / "end2end_dataset.py").is_file(),
                                           reason="MPDocBench repo not found (data/mpdocbench/repo/MPDocBench)")


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _unit(uid, bt, content, parent=None):
    return {"unit_id": uid, "block_type": bt, "content": content, "parent_unit_id": parent,
            "source_block_ids": []}


def test_gold_placeholder_strip_and_expand(tmp_path):
    gold = {
        "parsed_text": [
            {"type": "clause", "text": "1. Goods listed below are exempt. [table_1] Subject to conditions."},
            {"type": "clause", "text": "[TABLE_2]"},
            {"type": "clause", "text": "2. See [Table Header: Serial | Name] for format."},
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
    assert "[Table Header: Serial | Name]" in texts[1]
    assert len(g["items"]) == 3
    assert g["stats"]["table_only_items"] == 1
    assert g["stats"]["excluded_page_items"] == 1
    assert g["stats"]["unresolved_placeholders"] == 1
    assert g["stats"].get("strip_mismatch", 0) == 0
    full = g["full_text"]
    assert full.index("Horses") < full.index("Subject to conditions")
    assert "unref" in full and "table_1" not in full.lower().replace(" ", "")


def _identity_case(tmp_path):
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


ORIGINAL_EVALUATOR_METRICS = {"gold_clauses": 2, "gold_metadata": 2, "gold_tables": 0, "table_teds": None}


def _without_text_edit(m):
    return {k: v for k, v in m.items() if k != "text_edit"}


@needs_mpdocbench_repo
def test_regression_identity_with_the_original_evaluator(tmp_path):
    gold_path, fd = _identity_case(tmp_path)
    m = evaluate.evaluate_pair(evaluate.load_gold(gold_path), adapt.adapt_final_document(fd), "docA")
    assert _without_text_edit(m) == ORIGINAL_EVALUATOR_METRICS
    assert 0.0 < m["text_edit"]["edit_dist"] < 1.0 and m["text_edit"]["n_blocks"] >= 3


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


@needs_mpdocbench_repo
def test_run_is_read_only_on_inputs_and_writes_only_eval(tmp_path):
    run, gold_root = _tiny_run(tmp_path)
    before_run, before_gold = _tree_hash(run), _tree_hash(gold_root)

    out = evaluate.evaluate_run(run, gold_root=gold_root, workers=1)

    assert out["integrity"]["inputs_unchanged"] is True and out["integrity"]["PASS"] is True
    assert _tree_hash(gold_root) == before_gold
    assert {k: v for k, v in _tree_hash(run).items() if not k.startswith("_eval/")} == before_run
    metrics = json.loads((run / "_eval" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["evaluator"] == evaluate.EVALUATOR_VERSION
    assert _without_text_edit(metrics["results"]["m"]["docA"]) == ORIGINAL_EVALUATOR_METRICS
    assert metrics["results"]["m"]["docA"]["text_edit"]["n_blocks"] >= 3
    assert (run / "_eval" / "final_documents" / "docA" / "m" / "final_stitched_document.json").exists()
    assert sorted(p.name for p in (run / "_eval").iterdir()) == [
        "final_documents", "integrity.json", "metrics.json"]


@needs_mpdocbench_repo
def test_legacy_single_model_layout(tmp_path):
    run, gold_root = _tiny_run(tmp_path, legacy=True)
    out = evaluate.evaluate_run(run, gold_root=gold_root, workers=1)
    assert list(out["results"]) == [evaluate.LEGACY_MODEL_LABEL]
    assert out["results"][evaluate.LEGACY_MODEL_LABEL]["docA"]["gold_clauses"] == 2
