import json
from pathlib import Path

from src.core import config
from src.llm import structured_output as so
from src.registry import skill_retriever
from src.registry.skill_registry import SkillRegistry
from src.storage import skill_store

GENERAL_SKILLS = {"text_block", "heading", "table", "figure", "equation",
                  "caption", "list", "code", "header_footer_page_number"}
LEGAL_ONLY = {"numbered_point", "form_structure", "quoted_text", "figure_or_seal", "document_metadata"}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_block_types_are_the_general_set():
    assert set(config.ALLOWED_BLOCK_TYPES) == GENERAL_SKILLS | {"noise_or_separator"}
    assert "image_or_seal" not in config.ALLOWED_BLOCK_TYPES


def test_registry_status_comes_from_registry_hermetic(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    _write(skills / "registry.json", {"skills": [
        {"skill_id": "table", "latest_version": 1, "status": "main", "record": "records/table/v0001.json"},
        {"skill_id": "old", "latest_version": 1, "status": "deprecated", "record": "records/old/v0001.json"},
    ]})
    for sid in ("table", "old"):
        _write(skills / "records" / sid / "v0001.json",
               {"skill_id": sid, "version": 1, "status": "main", "description": "d",
                "applicability": ["a"], "instructions": ["i"], "conflicts_with": [], "provenance": {}})
    monkeypatch.setattr(config, "SKILLS_DIR", skills)
    reg = SkillRegistry()
    assert {r.skill_id for r in reg.by_status("main")} == {"table"}
    assert {r.skill_id for r in reg.by_status("deprecated")} == {"old"}
    assert reg.get("table").skill_id == "table"


def test_real_registry_is_the_general_set():
    reg = SkillRegistry()
    main_ids = {r.skill_id for r in reg.by_status("main")}
    assert main_ids == GENERAL_SKILLS, main_ids
    assert not (main_ids & LEGAL_ONLY)
    assert reg.get("table").version == 2
    for r in reg.by_status("main"):
        assert r.extractor_view()["instructions"], f"{r.skill_id} has no instructions"


def test_every_registered_record_satisfies_schema():
    for entry in skill_store.list_registry_entries():
        rec = skill_store.read_record_by_relpath(entry["record"])
        errs = so.schema_errors(rec, "skill_record")
        assert errs == [], f"{entry['skill_id']}: {errs}"


def test_retrieval_surfaces_expected_skills():
    reg = SkillRegistry()

    def top(query):
        return {r.skill_id for r in skill_retriever.retrieve_skills(query, ["main"], 3, registry=reg)}

    assert "table" in top("a long uniform parallel-structured serial appendix schedule "
                          "catalogue list of numbered items")
    assert "equation" in top("a standalone display equation formula in LaTeX")
    assert "figure" in top("an embedded figure image chart plot diagram illustration")
