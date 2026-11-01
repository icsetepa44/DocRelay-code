"""Critic-corrector flow (SuperValidator -> FinalDecider), as in the TIDE pipeline.

Hermetic — NO API. The LLM is faked at `structured_output.chat_completion` (so request_json's real JSON
parsing + schema validation runs on every canned answer), and the real HTTP client is replaced by a function
that fails the test if anything ever reaches it. A 2-page synthetic document runs through the real
orchestrator (page mode, keyword retrieval, vision off), then the produced final_document.json is scored
by the UNCHANGED MPDocBench evaluator.

Proves: clean critique -> no corrector; each correction trigger (incl. the MPDocBench `wrong_reading_order`)
-> corrector adopted; review applied only when the draft is kept (block-id fix); continuation review opens
a cross-page unit that the assembler stitches; critic / corrector failure keeps the draft; last page ends
everything; prompts render fully; legacy path still runs; evaluator integration."""

import json
import re

import pytest

from src.contracts.page import PageData
from src.core import config
from src.core import orchestrator
from src.eval import mpdocbench as mpdb
from src.llm import client as llm_client
from src.llm import structured_output as so
from src.llm.prompt_loader import render_prompt
from src.stages import validator as legacy_validator

DOC = "testdoc"
LABEL = "fake_gemma"
EXTRACTOR = {"backend": "openrouter", "model": "fake/gemma", "vision": False, "quantizations": None}

PAGE0 = "# Results\n\nThe first paragraph is complete.\n\nThe second paragraph runs off the page and"
PAGE1 = "continues on the next page until it ends.\n\nA closing paragraph."


def _payload(user_prompt: str) -> dict:
    obj, _ = json.JSONDecoder().raw_decode(user_prompt[user_prompt.index("{"):])
    return obj


def _draft_blocks(markdown: str, carry: dict) -> list:
    """Fake extractor: one block per markdown paragraph; the first block continues any open carry."""
    open_units = (carry or {}).get("open_units") or []
    blocks = []
    for i, para in enumerate(p.strip() for p in markdown.split("\n\n") if p.strip()):
        m = re.match(r"^(#+)\s+(.*)$", para)
        btype, content = (("heading", {"text": m.group(2), "level": len(m.group(1)) + 1}) if m
                          else ("text_block", {"text": para}))
        stitch = open_units[0]["stitch_id"] if (i == 0 and open_units) else None
        blocks.append({"block_id": f"b{i + 1}", "block_type": btype, "parent_block_id": None,
                       "stitch_id": stitch, "content": content, "skills_used": []})
    return blocks


class FakeLLM:
    """Role-dispatching fake. `sv[page]` / `fd[page]` are canned answers (dict), callables, or exceptions."""

    def __init__(self, sv=None, fd=None, extractor_decision="uncertain"):
        self.sv, self.fd = sv or {}, fd or {}
        self.extractor_decision = extractor_decision
        self.calls = []

    def __call__(self, system_prompt, user_prompt, config=None, images=None):
        payload = _payload(user_prompt)
        page = payload.get("page_number")
        if system_prompt.startswith("You are the SuperValidator"):
            self.calls.append(("sv", page))
            ans = self.sv.get(page, {"page_number": page, "status": "clean", "issues": [],
                                     "continuation_review": []})
        elif system_prompt.startswith("You are the FinalDecider"):
            self.calls.append(("fd", page))
            ans = self.fd.get(page)
            if ans is None:  # default corrector: return the draft unchanged
                ans = payload["draft_extraction"]
        elif system_prompt.startswith("You are the main page extractor"):
            self.calls.append(("extractor", page))
            blocks = _draft_blocks(payload["markdown"], payload.get("carry_state"))
            ans = {"page_number": page, "blocks": blocks, "continuation": {
                "page_number": page,
                "decisions": [{"block_id": blocks[-1]["block_id"], "stitch_id": None,
                               "decision": self.extractor_decision, "evidence": ["cut"]}]}}
        elif system_prompt.startswith("You are the Skill Router"):
            self.calls.append(("router", page))
            ans = {"page_number": page, "skills": [{"skill_id": "text_block", "version": 1}],
                   "coverage": "sufficient"}
        elif system_prompt.startswith("You validate the final assembled document"):
            self.calls.append(("final_validator", None))
            ans = {"document_id": payload.get("document_id", DOC), "status": "pass", "issues": []}
        elif system_prompt.startswith("You are an independent validator"):
            self.calls.append(("legacy_validator", page))
            ans = {"page_number": page, "status": "pass", "issues": []}
        else:  # pragma: no cover - a new role would need a canned answer
            raise AssertionError(f"unexpected role prompt: {system_prompt[:60]!r}")
        if callable(ans):
            ans = ans(payload)
        if isinstance(ans, Exception):
            raise ans
        return json.dumps(ans)

    def count(self, role):
        return sum(1 for r, _ in self.calls if r == role)


@pytest.fixture
def env(monkeypatch, tmp_path):
    def no_network(*a, **k):
        raise AssertionError("real LLM client reached — the smoke must not call any API")

    monkeypatch.setattr(llm_client, "chat_completion", no_network)
    monkeypatch.setattr(config, "RETRIEVAL_SCORER", "keyword")
    monkeypatch.setattr(config, "RETRIEVAL_MODE", "page")
    monkeypatch.setattr(config, "USE_SUPERVALIDATOR", True)
    orig_model_for = config.model_for
    monkeypatch.setattr(config, "model_for", lambda role: {**orig_model_for(role), "vision": False})
    pages = [PageData(DOC, 0, "", PAGE0, ""), PageData(DOC, 1, "", PAGE1, "")]
    monkeypatch.setattr(orchestrator.ds, "load_materialized_pages", lambda doc: list(pages))

    def run(fake, **kw):
        monkeypatch.setattr(so, "chat_completion", fake)
        monkeypatch.setattr(legacy_validator, "chat_completion", fake)
        summary = orchestrator.run_document(DOC, tmp_path, extractor_model=EXTRACTOR, model_label=LABEL,
                                            **kw)
        return summary, tmp_path / DOC / LABEL

    return run


def _page(ddir, page, name):
    return json.loads((ddir / "pages" / f"{page:04d}" / f"{name}.json").read_text(encoding="utf-8"))


def _issue(page, itype, target=None):
    return {"page_number": page, "status": "has_issues", "continuation_review": [],
            "issues": [{"issue_type": itype, "target_block_id": target, "description": "d",
                        "suggested_fix": "f"}]}


# --------------------------------------------------------------------------- critic-corrector gate
def test_clean_critique_skips_corrector(env):
    fake = FakeLLM()
    summary, ddir = env(fake)
    assert fake.count("sv") == 2 and fake.count("fd") == 0 and fake.count("legacy_validator") == 0
    assert _page(ddir, 0, "validation")["status"] == "pass"
    assert _page(ddir, 0, "super_validation")["status"] == "clean"
    assert _page(ddir, 0, "continuation_judge")["judge_invoked"] is False
    assert summary["use_supervalidator"] is True and summary["judge_pages"] == 0
    # extractor's "uncertain" with no review -> ends: nothing stays open
    assert _page(ddir, 0, "carry_state")["open_units"] == []


@pytest.mark.parametrize("itype", ["missing_text", "wrong_hierarchy", "split_clause", "merged_clause",
                                   "wrong_block_type", "wrong_reading_order"])
def test_each_trigger_invokes_corrector(env, itype):
    fake = FakeLLM(sv={0: _issue(0, itype, f"{DOC}:p0000:b0002")})
    summary, ddir = env(fake)
    assert fake.calls.count(("fd", 0)) == 1 and fake.calls.count(("fd", 1)) == 0
    assert _page(ddir, 0, "continuation_judge")["judge_invoked"] is True
    assert _page(ddir, 0, "validation")["status"] == "fail"
    assert summary["judge_pages"] == 1 and summary["corrector_adopted_pages"] == 1


@pytest.mark.parametrize("itype", ["continuation_ambiguity", "noise_as_content"])
def test_report_only_issues_do_not_invoke_corrector(env, itype):
    fake = FakeLLM(sv={0: _issue(0, itype)})
    env(fake)
    assert fake.count("fd") == 0


def test_corrector_output_adopted_and_review_not_reapplied(env):
    # Corrector splits the heading off differently AND drops a block, so ids are renumbered; the critic's
    # review ("continues" for the DRAFT's b0003) must NOT be applied to the corrected page (block-id fix).
    sv = _issue(0, "merged_clause", f"{DOC}:p0000:b0002")
    sv["continuation_review"] = [{"block_id": f"{DOC}:p0000:b0003", "stitch_id": None,
                                  "decision": "continues", "evidence": ["x"]}]
    corrected = {"page_number": 0, "blocks": [
        {"block_id": "n1", "block_type": "heading", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "Results", "level": 2}, "skills_used": []},
        {"block_id": "n2", "block_type": "text_block", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "The first paragraph is complete."}, "skills_used": []},
        {"block_id": "n3", "block_type": "text_block", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "The second paragraph runs off the page and"}, "skills_used": []},
        {"block_id": "n4", "block_type": "text_block", "parent_block_id": None, "stitch_id": None,
         "content": {"text": "extra"}, "skills_used": []}],
        "continuation": {"page_number": 0, "decisions": [
            {"block_id": "n4", "stitch_id": None, "decision": "ends", "evidence": []}]}}
    fake = FakeLLM(sv={0: sv}, fd={0: corrected})
    _, ddir = env(fake)
    ext = _page(ddir, 0, "extraction")
    assert [b["content"].get("text") for b in ext["blocks"]][-1] == "extra"  # corrected page persisted
    assert [d["decision"] for d in ext["continuation"]["decisions"]] == ["ends"]
    jr = _page(ddir, 0, "continuation_judge")
    assert jr["corrector_adopted"] is True and jr["review_applied"] is False
    assert _page(ddir, 0, "carry_state")["open_units"] == []


def test_review_continues_opens_cross_page_unit(env):
    review = {"page_number": 0, "status": "has_issues",
              "issues": [{"issue_type": "continuation_ambiguity", "target_block_id": None,
                          "description": "cut mid-sentence", "suggested_fix": None}],
              "continuation_review": [{"block_id": f"{DOC}:p0000:b0003", "stitch_id": None,
                                       "decision": "continues", "evidence": ["runs off the page and"]}]}
    fake = FakeLLM(sv={0: review})
    _, ddir = env(fake)
    assert fake.count("fd") == 0
    assert _page(ddir, 0, "continuation_judge")["review_applied"] is True
    carry = _page(ddir, 0, "carry_state")["open_units"]
    assert len(carry) == 1
    fd = json.loads((ddir / "final_document.json").read_text(encoding="utf-8"))
    spanning = [u for u in fd["units"] if len({s.split(":")[1] for s in u["source_block_ids"]}) == 2]
    assert len(spanning) == 1 and spanning[0]["unit_id"] == carry[0]["stitch_id"]


def test_last_page_ends_everything(env):
    review1 = {"page_number": 1, "status": "clean", "issues": [], "continuation_review": [
        {"block_id": f"{DOC}:p0001:b0002", "stitch_id": None, "decision": "continues", "evidence": []}]}
    fake = FakeLLM(sv={1: review1}, extractor_decision="continues")
    _, ddir = env(fake)
    assert {d["decision"] for d in _page(ddir, 1, "extraction")["continuation"]["decisions"]} == {"ends"}
    assert _page(ddir, 1, "carry_state")["open_units"] == []


# --------------------------------------------------------------------------- failure handling
def test_critic_failure_keeps_draft(env):
    fake = FakeLLM(sv={0: RuntimeError("critic down")})
    summary, ddir = env(fake)
    assert fake.count("fd") == 0
    assert len(_page(ddir, 0, "extraction")["blocks"]) == 3  # the draft survived
    assert _page(ddir, 0, "validation")["status"] == "error"
    assert any(e["stage"] == "super_validator" for e in summary["page_errors"])


@pytest.mark.parametrize("fd_answer", [RuntimeError("corrector down"),
                                       {"page_number": 0, "blocks": []}])
def test_corrector_failure_or_empty_keeps_draft(env, fd_answer):
    fake = FakeLLM(sv={0: _issue(0, "missing_text")}, fd={0: fd_answer})
    summary, ddir = env(fake)
    assert fake.calls.count(("fd", 0)) == 1
    assert len(_page(ddir, 0, "extraction")["blocks"]) == 3
    assert _page(ddir, 0, "continuation_judge")["corrector_adopted"] is False
    assert summary["corrector_adopted_pages"] == 0


def test_legacy_path_still_runs(env, monkeypatch):
    monkeypatch.setattr(config, "USE_SUPERVALIDATOR", False)
    fake = FakeLLM()
    summary, ddir = env(fake)
    assert fake.count("sv") == 0 and fake.count("fd") == 0 and fake.count("legacy_validator") == 2
    assert summary["use_supervalidator"] is False
    assert not (ddir / "pages" / "0000" / "super_validation.json").exists()


# --------------------------------------------------------------------------- prompts
@pytest.mark.parametrize("name", ["super_validator", "final_decider"])
@pytest.mark.parametrize("domain", [None, "newspaper", "examPaper"])
def test_prompts_render_without_leftover_tokens(name, domain):
    text = render_prompt(name, domain=domain, allowed_block_types=", ".join(config.ALLOWED_BLOCK_TYPES),
                         ocr_format_note="Chandra Markdown prose with any tables kept inline as HTML.")
    assert not re.search(r"\{\{\s*[a-zA-Z0-9_]+\s*\}\}", text)  # every {{token}} filled
    assert "header_footer_page_number" in text
    assert "Bangla" not in text and "যথা" not in text  # no legal-corpus rules


# --------------------------------------------------------------------------- evaluator integration
@pytest.mark.skipif(not (config.MPDOCBENCH_REPO / "metrics").is_dir(),
                    reason="official MPDocBench code not downloaded (data/mpdocbench/repo/MPDocBench)")
def test_output_scores_through_unchanged_mpdocbench_evaluator(env, monkeypatch, tmp_path):
    review = {"page_number": 0, "status": "clean", "issues": [], "continuation_review": [
        {"block_id": f"{DOC}:p0000:b0003", "stitch_id": None, "decision": "continues", "evidence": []}]}
    env(FakeLLM(sv={0: review}))
    gold = {
        "page_info": {"image_path": f"{DOC}.pdf", "width": [1000, 1000], "height": [1000, 1000],
                      "page_attribute": {"data_source": "book", "language": "english"}},
        "layout_dets": [
            {"anno_id": 1, "category_type": "title", "page_id": 1, "order": 1, "text": "Results"},
            {"anno_id": 2, "category_type": "text_block", "page_id": 1, "order": 2,
             "text": "The first paragraph is complete."},
            {"anno_id": 3, "category_type": "text_block", "page_id": 1, "order": 3,
             "text": "The second paragraph runs off the page and"},
            {"anno_id": 4, "category_type": "text_block", "page_id": 2, "order": 4,
             "text": "continues on the next page until it ends."},
            {"anno_id": 5, "category_type": "text_block", "page_id": 2, "order": 5,
             "text": "A closing paragraph."},
        ],
        "extra": {"relation": [
            {"source_anno_id": "root", "target_anno_id": 1, "relation_type": "parent_son"},
            {"source_anno_id": 3, "target_anno_id": 4, "relation_type": "truncated"}],
            "pdf_name": "hash"},
    }
    monkeypatch.setattr(mpdb, "load_gold_index", lambda: {DOC: gold})
    monkeypatch.setattr(mpdb, "_domain_map", lambda stems: {DOC: "book"})
    res = mpdb.evaluate_run(tmp_path)
    overall = res["models"][LABEL]["aggregate"]["overall"]
    assert overall["text"] == pytest.approx(1.0, abs=0.02)
    assert overall["merged_text"] == pytest.approx(1.0, abs=0.02)  # the stitched cross-page paragraph
    assert (tmp_path / "_mpdocbench" / "metrics.json").exists()


# --------------------------------------------------------------------------- progress callback
def test_on_page_done_fires_once_per_page_including_fallback(env, monkeypatch):
    """run.py's page-level progress bar relies on this: one tick per processed page, also when the page
    went through the extraction-failure fallback (which skips the rest of the page loop)."""
    real_extract = orchestrator.extractor.extract

    def flaky_extract(page, *a, **k):
        if page.page_number == 0:
            raise RuntimeError("extractor down")  # first attempt AND the markdown-only retry fail
        return real_extract(page, *a, **k)

    monkeypatch.setattr(orchestrator.extractor, "extract", flaky_extract)
    ticks = []
    summary, ddir = env(FakeLLM(), on_page_done=ticks.append)
    assert ticks == [0, 1]
    assert any(e["stage"] == "extract" for e in summary["page_errors"])  # page 0 took the fallback path


def test_failing_progress_callback_never_breaks_a_run(env):
    def boom(_):
        raise RuntimeError("bar broke")

    summary, _ = env(FakeLLM(), on_page_done=boom)
    assert summary["pages"] == 2


# --------------------------------------------------------------------------- ablation switch: extractor only
def test_skip_verifier_and_decider_makes_no_check_calls(env, monkeypatch):
    """skip_verifier_and_decider: the draft goes straight to the deterministic continuation step — no SuperValidator,
    no FinalDecider, no legacy validator/judge call; uncertain -> ends; the doc still assembles."""
    monkeypatch.setattr(config, "SKIP_VERIFIER_AND_DECIDER", True)
    fake = FakeLLM(sv={0: _issue(0, "missing_text")})  # would trigger the corrector if the critic ran
    summary, ddir = env(fake)
    assert fake.count("sv") == 0 and fake.count("fd") == 0 and fake.count("legacy_validator") == 0
    assert fake.count("extractor") == 2
    assert summary["use_supervalidator"] is True and summary["skip_verifier_and_decider"] is True
    assert summary["judge_pages"] == 0 and summary["page_errors"] == []
    assert _page(ddir, 0, "validation")["status"] == "skipped"
    assert {d["decision"] for d in _page(ddir, 0, "extraction")["continuation"]["decisions"]} == {"ends"}
    assert len(_page(ddir, 0, "extraction")["blocks"]) == 3  # the draft, untouched
    assert (ddir / "final_document.json").exists()


def test_full_system_reports_the_flag_off(env):
    summary, _ = env(FakeLLM())
    assert summary["skip_verifier_and_decider"] is False


def test_ledger_tags_every_call_with_role_and_page(env, tmp_path):
    """Each LLM call carries its role (structured_output) and page (orchestrator) into the cost/time ledger."""
    from src.llm import usage

    inner = FakeLLM()

    def recording(system_prompt, user_prompt, config=None, images=None):
        usage.record(model="fake", status="ok", cost_usd=0.001)
        return inner(system_prompt, user_prompt, config, images)

    with usage.ledger(tmp_path, "s1"), usage.tags(doc=DOC, label=LABEL):
        env(recording)
    rows = [json.loads(x) for x in (tmp_path / "_cost_time" / "llm_calls.jsonl").read_text().splitlines()]
    assert len(rows) == len(inner.calls)
    assert {(r["role"], r["page"]) for r in rows} == {
        ("skill_router", 0), ("extractor", 0), ("super_validator", 0),
        ("skill_router", 1), ("extractor", 1), ("super_validator", 1), ("final_validator", None)}
    assert all(r["doc"] == DOC and r["label"] == LABEL and r["session"] == "s1" and r["repair"] == 0 for r in rows)
