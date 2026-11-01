"""
Orchestrator — the v1 stable-core per-page loop (spec §20 minus the §21 failure branch)
followed by deterministic assembly + final validation (spec §29).

Per page:  retrieve main skills -> route -> extract -> validate (report-only)
           -> resolve continuation (judge only when uncertain) -> build carry state -> persist
Then:      assemble -> final validate -> persist FinalDocument + FinalValidation.

Runs for ONE (document, extractor_model). The extractor model is the compared variable;
all other agents use their configured (fixed) models. Each stage sends the page image only
if its resolved model is vision-capable (text-only models run on OCR markdown+html).
Validator is report-only: results are persisted and logged, never branched into repair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.block import Block
from src.contracts.continuation import CarryState, ContinuationDecision
from src.contracts.page import PageData, PageExtraction
from src.contracts.validation import PageValidation
from src.core import config
from src.core import resume as resume_mod
from src.core.exceptions import PipelineError, UnresolvedContinuationError
from src.core.ids import StitchIdAllocator, block_id
from src.registry.skill_registry import get_registry
from src.registry.skill_retriever import retrieve_skills
from src.stages import (
    assembler,
    carry_builder,
    carry_rejection,
    continuation_judge,
    extractor,
    final_decider,
    final_validator,
    region_analyzer,
    schedule_guard,
    skill_router,
    super_validator,
    validator,
)
from src.stages._common import block_text
from src.storage import document_store as ds
from src.storage import run_store


def _build_query(page: PageData, carry: CarryState) -> str:
    parts = [page.markdown[:800]]
    for u in carry.open_units:
        parts.append(f"[open unit {u.block_type}] {u.state}")
    return "\n".join(p for p in parts if p)


def _open_table_pins(carry: CarryState) -> List[str]:
    """Skills to force into retrieval this page: pin `table` when a table is still open, so a
    continuing table fragment keeps the anti-shred / cross-page instructions even if its page
    text no longer looks table-shaped and drops the skill from top-k (T7)."""
    if any(u.block_type == "table" for u in carry.open_units):
        return ["table"]
    return []


def _regions_guidance(region_analysis, region_selection) -> List[Dict[str, Any]]:
    """Compact per-region guidance for the extractor: description + chosen skill_ids."""
    skills_by_region: Dict[str, List[str]] = {
        r.region_id: [s.skill_id for s in r.skills] for r in region_selection.regions
    }
    out = []
    for r in region_analysis.regions:
        out.append({
            "region_id": r.region_id,
            "description": r.description,
            "continuation_context": r.continuation_context,
            "skills": skills_by_region.get(r.region_id, []),
        })
    return out


def _resolve_continuation(
    extraction: PageExtraction,
    page: PageData,
    next_page: Optional[PageData],
    carry: CarryState,
    image: Any,
    unresolved_policy: str,
    judge_vision: bool = True,
) -> tuple[ContinuationDecision, Dict[str, Any]]:
    """Resolve every 'uncertain' continuation decision and return (cont, judge_record).

    `judge_record` makes the (otherwise transient) judge activity observable: it captures
    what the extractor / the B2 rule flagged as uncertain, whether the judge was actually
    invoked, and its verdicts — because 'uncertain' is always resolved before the page is
    persisted, so it can never be counted from extraction.json afterward.
    """
    cont = extraction.continuation or ContinuationDecision(page.page_number, [])
    extractor_uncertain = [d.block_id for d in cont.decisions if d.decision == "uncertain"]
    judge_record: Dict[str, Any] = {
        "page_number": page.page_number,
        "extractor_uncertain": list(extractor_uncertain),
        "b2_flagged": [],
        "judge_invoked": False,
        "judged_block_ids": [],
        "verdicts": [],
    }

    # Last page: nothing can continue -> coerce to ends.
    if next_page is None:
        for d in cont.decisions:
            if d.decision != "ends":
                d.decision = "ends"
        return cont, judge_record

    # --- B2 Option B: route unconfirmed main-point closes to the judge -------------
    # If a decision would CLOSE a main point but the next main number is NOT visible on
    # this page (no fresh is_main_point block appears later in reading order), the "next
    # main number" would live on next_page, which we must not text-scan (no-regex rule).
    # Flip it to "uncertain" so the judge (an LLM, which MAY read next_page) decides.
    if config.JUDGE_ON_UNCONFIRMED_MAIN_CLOSE:
        blocks_by_id = {b.block_id: b for b in extraction.blocks}
        pos = {b.block_id: i for i, b in enumerate(extraction.blocks)}  # reading order
        open_main = {u.stitch_id for u in carry.open_units if getattr(u, "is_main_point", False)}

        # The label each open main point is waiting for, so the T11 guard can check that a fresh
        # main is a plausible successor rather than a colliding shredded field-number.
        expected_by_stitch = {
            u.stitch_id: u.expected_next_main_point
            for u in carry.open_units if getattr(u, "is_main_point", False)
        }

        def _closes_main_point(d) -> bool:
            b = blocks_by_id.get(d.block_id)
            return bool(getattr(d, "is_main_point", False)) or (
                b is not None and b.stitch_id is not None and b.stitch_id in open_main)

        def _has_successor_main_point(after_i: int, expected_label: Optional[str]) -> bool:
            # a FRESH main point (stitch_id null) later on THIS page == next main number visible.
            # T11 guard (when CARRY_COLLISION_GUARD): a fresh main only counts if its ordinal
            # plausibly follows the open rule's expected_next_main_point — so a stray shredded
            # field-number can't masquerade as the successor and prematurely close a real rule.
            for d2 in cont.decisions:
                b2 = blocks_by_id.get(d2.block_id)
                if b2 is None or b2.stitch_id is not None:
                    continue
                if getattr(d2, "is_main_point", False) and pos.get(d2.block_id, -1) > after_i:
                    if config.CARRY_COLLISION_GUARD and not carry_rejection.is_plausible_successor(
                            expected_label, block_text(b2)):
                        continue
                    return True
            return False

        for d in cont.decisions:
            if d.decision != "ends" or not _closes_main_point(d):
                continue
            b = blocks_by_id.get(d.block_id)
            expected_label = expected_by_stitch.get(b.stitch_id) if b is not None else None
            if not _has_successor_main_point(pos.get(d.block_id, -1), expected_label):
                d.decision = "uncertain"
                judge_record["b2_flagged"].append(d.block_id)

    # Recompute so any freshly-flagged blocks reach the judge BEFORE the coerce fallback.
    uncertain = [d.block_id for d in cont.decisions if d.decision == "uncertain"]

    if uncertain and config.USE_JUDGE_WHEN_UNCERTAIN:
        judge_record["judge_invoked"] = True
        judge_record["judged_block_ids"] = list(uncertain)
        next_image = ds.load_model_image(next_page.page_image) if judge_vision else None
        judged = continuation_judge.judge(
            page, next_page, extraction, carry, uncertain,
            image if judge_vision else None, next_image,
        )
        judge_record["verdicts"] = [
            {"block_id": d.block_id, "decision": d.decision, "evidence": list(d.evidence or [])}
            for d in judged.decisions
        ]
        judged_by_block = {d.block_id: d for d in judged.decisions}
        for d in cont.decisions:
            jd = judged_by_block.get(d.block_id)
            if jd is not None:
                d.decision = jd.decision
                if jd.evidence:
                    d.evidence = jd.evidence

    still = [d for d in cont.decisions if d.decision == "uncertain"]
    if still:
        if unresolved_policy == "stop":
            raise UnresolvedContinuationError(
                f"page {page.page_number}: unresolved continuation for "
                f"{[d.block_id for d in still]}"
            )
        for d in still:  # 'coerce': conservatively close
            d.decision = "ends"
    return cont, judge_record


def run_document(
    doc_name: str,
    run_dir: Path,
    page_start: int = 0,
    page_end: Optional[int] = None,
    run_final_validator: bool = True,
    unresolved_policy: Optional[str] = None,
    overwrite_images: bool = False,
    stop_on_page_error: bool = False,
    extractor_model: Optional[Dict[str, Any]] = None,
    model_label: Optional[str] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    unresolved_policy = unresolved_policy or config.UNRESOLVED_POLICY
    registry = get_registry()
    prefix = f"[{model_label}] " if model_label else ""

    # Read-only load of pre-materialised pages (fallback: materialise if absent).
    pages = ds.load_materialized_pages(doc_name)
    if not pages:
        pages = ds.materialize_document(doc_name, overwrite=overwrite_images)
    pages = pages[page_start:page_end] if page_end is not None else pages[page_start:]
    if not pages:
        raise PipelineError(f"No pages for {doc_name} in range [{page_start}:{page_end}]")
    document_id = pages[0].document_id

    # Per-stage vision resolution (extractor from override; others from config).
    extractor_model = extractor_model or config.model_for("extractor")
    extractor_vision = bool(extractor_model.get("vision", False))
    router_vision = config.model_for("skill_router")["vision"]
    validator_vision = config.model_for("validator")["vision"]
    super_validator_vision = config.model_for("super_validator")["vision"]
    final_decider_vision = config.model_for("final_decider")["vision"]
    judge_vision = config.model_for("continuation_judge")["vision"]
    region_mode = config.RETRIEVAL_MODE == "region"
    region_vision = config.model_for("region_analyzer")["vision"] if region_mode else False
    need_image = (extractor_vision or router_vision or validator_vision or judge_vision
                  or region_vision or super_validator_vision or final_decider_vision)

    carry = CarryState.empty()
    extractions: List[PageExtraction] = []
    page_summaries: List[Dict[str, Any]] = []
    page_errors: List[Dict[str, Any]] = []
    resume_index = 0
    if resume:
        # Page-level resume: rehydrate the longest contiguous prefix of already-persisted pages
        # and restore the state the loop would have had (carry, extractions, stitch allocator) so
        # finished pages are NOT re-extracted. The torn/incomplete tail is cleaned before we resume.
        rs = resume_mod.compute_resume_state(
            run_dir, doc_name, model_label, pages,
            min_chars_nontrivial=config.MIN_CHARS_NONTRIVIAL_PAGE)
        resume_index = rs.index
        extractions = rs.extractions
        carry = rs.carry
        page_summaries = rs.page_summaries
        allocator = StitchIdAllocator(document_id, start=rs.allocator_start)
        removed = resume_mod.clean_incomplete_tail(run_dir, doc_name, model_label, pages, resume_index)
        print(f"{prefix}RESUME {doc_name}: {resume_index}/{len(pages)} page(s) reused, "
              f"allocator@{rs.allocator_start}"
              + (f", cleaned tail {removed}" if removed else ""), flush=True)
    else:
        allocator = StitchIdAllocator(document_id)

    for i, page in enumerate(pages):
        if i < resume_index:
            continue  # already persisted on a prior run; kept in `pages` for next_page look-ahead
        next_page = pages[i + 1] if i + 1 < len(pages) else None
        print(f"{prefix}[page {page.page_number}] extracting…", flush=True)
        image = ds.load_model_image(page.page_image) if need_image else None

        # --- ESSENTIAL: route + extract. Failure here loses the page's content. ---
        selection_to_persist: Any = None
        region_analysis = None
        try:
            if region_mode:
                region_analysis = region_analyzer.analyze(
                    page, carry, image if region_vision else None)
                table_pins = _open_table_pins(carry)
                per_region_candidates = {
                    r.region_id: retrieve_skills(
                        r.description or page.markdown[:400], ["main"], config.MAIN_TOP_K,
                        pin_skill_ids=table_pins)
                    for r in region_analysis.regions
                }
                region_selection = skill_router.route_regions(
                    page, carry, region_analysis, per_region_candidates,
                    image if router_vision else None)
                selection_to_persist = region_selection
                selected = (registry.resolve(region_selection.all_selected_skills())
                            or registry.by_status("main"))
                extraction = extractor.extract(
                    page, next_page, carry, selected,
                    image if extractor_vision else None,
                    model_override=extractor_model,
                    regions=_regions_guidance(region_analysis, region_selection),
                )
            else:
                query = _build_query(page, carry)
                candidates = retrieve_skills(query, ["main"], config.MAIN_TOP_K,
                                             pin_skill_ids=_open_table_pins(carry))
                selection = skill_router.route(page, carry, candidates,
                                               image if router_vision else None)
                selection_to_persist = selection
                selected = registry.resolve(selection.skills) or candidates
                extraction = extractor.extract(
                    page, next_page, carry, selected,
                    image if extractor_vision else None,
                    model_override=extractor_model,
                )
        except Exception as exc:  # noqa: BLE001
            if stop_on_page_error:
                raise
            print(f"{prefix}  EXTRACTION ERROR on page {page.page_number}: {exc}")
            page_errors.append({"page_number": page.page_number, "stage": "extract", "error": str(exc)})
            # Graceful fallback (FAILURE_ANALYSIS_V8 mixed-OCR fragility): do NOT drop the page.
            # First retry the extractor in the most robust mode — page-mode, markdown-only OCR, all
            # 'main' skills, no region guidance (this recovers region_analyzer/skill_router schema
            # failures and format-tripped extractor failures). If that also fails, keep the page's
            # OCR text as one flagged text_block so a page with real OCR never yields zero content.
            recovery = "none"
            extraction = None
            try:
                extraction = extractor.extract(
                    page, next_page, carry, registry.by_status("main"),
                    image if extractor_vision else None,
                    model_override=extractor_model, regions=None, ocr_format="markdown")
                if extraction.blocks:
                    recovery = "markdown_retry"
                    print(f"{prefix}    recovered via markdown-only retry "
                          f"({len(extraction.blocks)} blocks)")
                else:
                    extraction = None
            except Exception as exc2:  # noqa: BLE001
                print(f"{prefix}    markdown-only retry failed: {exc2}")
            if extraction is None:
                ocr_text = (page.markdown or "").strip()
                if ocr_text:
                    recovery = "ocr_text_block"
                    extraction = PageExtraction(
                        page_number=page.page_number,
                        blocks=[Block(
                            block_id=block_id(document_id, page.page_number, 0),
                            block_type="text_block", parent_block_id=None, stitch_id=None,
                            content={"text": ocr_text, "parsing_status": "fallback"})],
                        continuation=ContinuationDecision(page.page_number, []))
                else:
                    extraction = PageExtraction(
                        page_number=page.page_number, blocks=[],
                        continuation=ContinuationDecision(page.page_number, []))
            extractions.append(extraction)
            # C3: preserve the incoming carry (don't wipe it) so the next page's
            # continuation is not orphaned — a page drop becomes a localized gap.
            carry = CarryState(from_page=page.page_number, open_units=carry.open_units)
            page_summaries.append({
                "page_number": page.page_number, "num_blocks": len(extraction.blocks),
                "validation_status": ("extract_error_recovered" if extraction.blocks
                                      else "extract_error"),
                "empty_nontrivial": not extraction.blocks, "recovery": recovery,
                "open_units_after": [u.stitch_id for u in carry.open_units]})
            continue

        # --- C2: a non-trivial page that returned ZERO blocks is a (possibly silent)
        #     extraction failure. Retry once, then flag it. ---
        nontrivial_empty = (not extraction.blocks
                            and len(page.markdown.strip()) >= config.MIN_CHARS_NONTRIVIAL_PAGE)
        if nontrivial_empty and config.RETRY_EMPTY_PAGE:
            print(f"{prefix}  page {page.page_number}: 0 blocks on a non-trivial page — retrying once")
            try:
                extraction = extractor.extract(
                    page, next_page, carry, selected,
                    image if extractor_vision else None,
                    model_override=extractor_model,
                    regions=(_regions_guidance(region_analysis, region_selection)
                             if region_mode else None),
                )
                nontrivial_empty = not extraction.blocks
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}    retry failed: {exc}")
        if nontrivial_empty:
            page_errors.append({"page_number": page.page_number, "stage": "extract",
                                "error": "empty_extraction"})

        # --- P2 (T1/T4): schedule over-segmentation. The detector is a deterministic diagnostic
        #     and always runs (over_segmentation.json). SCHEDULE_COLLAPSE additionally re-nests a
        #     flagged uniform run under its schedule cover (or its own first item) — rewriting only
        #     parent_block_id, never the text. Runs before validate/continuation/carry so every
        #     downstream stage sees the collapsed structure. Must never break a run. ---
        try:
            over_segmentation_record = schedule_guard.analyze_over_segmentation(
                schedule_guard.page_blocks_of(extraction.blocks),
                config.SCHEDULE_MIN_RUN, config.SCHEDULE_MAX_ITEM_CHARS)
            over_segmentation_record["page_number"] = page.page_number
            if config.SCHEDULE_COLLAPSE and over_segmentation_record["runs"]:
                extraction.blocks = schedule_guard.collapse_uniform_runs(
                    extraction.blocks, over_segmentation_record["runs"])
                print(f"{prefix}  page {page.page_number}: collapsed "
                      f"{over_segmentation_record['run_count']} uniform schedule run(s) "
                      f"({over_segmentation_record['flagged_block_count']} block(s) re-nested)")
        except Exception as exc:  # noqa: BLE001 — a diagnostic must never break a run
            over_segmentation_record = {"page_number": page.page_number, "runs": [],
                                        "run_count": 0, "flagged_block_count": 0, "error": str(exc)}

        if config.USE_SUPERVALIDATOR:
            # --- SUPERVALIDATOR (Critic) & FINALDECIDER (Corrector) ---
            super_validation = None
            judge_record = {
                "page_number": page.page_number,
                "judge_invoked": False,
                "extractor_uncertain": [],
                "b2_flagged": [],
                "judged_block_ids": [],
                "verdicts": [],
            }

            try:
                super_validation = super_validator.validate(
                    page, next_page, extraction, carry,
                    image if super_validator_vision else None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}  super_validator error (ignored): {exc}")
                page_errors.append({"page_number": page.page_number, "stage": "super_validator", "error": str(exc)})

            # Determine if FinalDecider (Corrector) should be invoked:
            CRITICAL_ISSUES = {"missing_text", "wrong_hierarchy", "split_clause", "merged_clause", "wrong_block_type"}
            needs_correction = False
            if super_validation and any(i.issue_type in CRITICAL_ISSUES for i in super_validation.issues):
                needs_correction = True
            elif not extraction.blocks:
                needs_correction = True

            if needs_correction and super_validation:
                try:
                    flagged_count = sum(1 for i in super_validation.issues if i.issue_type in CRITICAL_ISSUES)
                    print(f"{prefix}  page {page.page_number}: SuperValidator flagged {flagged_count} structural issue(s) — invoking FinalDecider corrector")
                    judge_record["judge_invoked"] = True
                    corrected_extraction = final_decider.decide(
                        page, next_page, extraction, super_validation, carry,
                        image if final_decider_vision else None,
                    )
                    if corrected_extraction and corrected_extraction.blocks:
                        extraction = corrected_extraction
                except Exception as exc:  # noqa: BLE001
                    print(f"{prefix}  final_decider error (keeping draft extraction): {exc}")
                    page_errors.append({"page_number": page.page_number, "stage": "final_decider", "error": str(exc)})

            # Finalize continuation decisions
            final_cont = extraction.continuation or ContinuationDecision(page.page_number, [])
            if next_page is None:
                # Last page -> all must end
                for d in final_cont.decisions:
                    d.decision = "ends"
            else:
                if super_validation and super_validation.continuation_review:
                    rev_by_id = {c.block_id: c for c in super_validation.continuation_review}
                    for d in final_cont.decisions:
                        rev = rev_by_id.get(d.block_id)
                        if rev and rev.decision in ("continues", "ends"):
                            d.decision = rev.decision
                for d in final_cont.decisions:
                    if d.decision == "uncertain":
                        d.decision = "ends"

            # Build page_validation object for reporting
            if super_validation:
                issues_list = [
                    {"block_ids": [i.target_block_id] if i.target_block_id else [], "type": i.issue_type, "description": i.description}
                    for i in super_validation.issues
                ]
                page_validation = PageValidation.from_dict({
                    "page_number": page.page_number,
                    "status": "pass" if not super_validation.has_issues else "fail",
                    "issues": issues_list,
                })
            else:
                page_validation = PageValidation(page_number=page.page_number, status="pass", issues=[])
        else:
            # --- BASELINE PIPELINE ---
            try:
                page_validation = validator.validate(page, extraction, carry,
                                                     image if validator_vision else None)
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}  validator error (ignored, report-only): {exc}")
                page_validation = PageValidation(page_number=page.page_number, status="error", issues=[])
                page_errors.append({"page_number": page.page_number, "stage": "validate", "error": str(exc)})

            try:
                final_cont, judge_record = _resolve_continuation(
                    extraction, page, next_page, carry, image, unresolved_policy, judge_vision
                )
            except UnresolvedContinuationError:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}  continuation error (ignored): {exc}")
                page_errors.append({"page_number": page.page_number, "stage": "continuation", "error": str(exc)})
                final_cont = extraction.continuation or ContinuationDecision(page.page_number, [])
                for d in final_cont.decisions:
                    if d.decision == "uncertain":
                        d.decision = "ends"
                judge_record = {"page_number": page.page_number, "judge_invoked": False,
                                "error": str(exc), "extractor_uncertain": [], "b2_flagged": [],
                                "judged_block_ids": [], "verdicts": []}

        # Pass the OLD carry (RHS evaluated before assignment) so B1 can keep a main
        # point's open-state/expected-next signals alive while only its sub-points stream.
        incoming_carry = carry
        carry = carry_builder.build_carry_state(extraction, final_cont, allocator, incoming_carry)
        # C3: a zero-block page must not silently wipe the carry — keep the incoming open units.
        if not extraction.blocks and incoming_carry.open_units:
            carry = CarryState(from_page=page.page_number, open_units=incoming_carry.open_units)
        # Carry-rejection diagnostic: which incoming open units (of ANY type) did the extractor
        # NOT continue on this page? Deterministic; must never break a run.
        errset = {pe.get("page_number") for pe in page_errors}
        try:
            carry_rejection_record = carry_rejection.analyze_carry(
                [{"stitch_id": u.stitch_id, "block_type": u.block_type,
                  "is_main_point": u.is_main_point, "state": u.state, "open_text": u.open_text,
                  "expected_next_main_point": u.expected_next_main_point}
                 for u in incoming_carry.open_units],
                [(b.block_id, b.block_type, b.parent_block_id, b.stitch_id, block_text(b))
                 for b in extraction.blocks],
                error_adjacent=(page.page_number in errset or (page.page_number - 1) in errset),
                collision_guard=config.CARRY_COLLISION_GUARD,
            )
            carry_rejection_record["page_number"] = page.page_number
        except Exception as exc:  # noqa: BLE001 — a diagnostic must never break a run
            carry_rejection_record = {"page_number": page.page_number, "units": [],
                                      "rejected_count": 0, "error": str(exc)}
        _persist_page(run_dir, doc_name, page, selection_to_persist, extraction,
                      page_validation, carry, model_label, region_analysis, judge_record,
                      carry_rejection_record, over_segmentation_record)
        extractions.append(extraction)
        page_summaries.append({
            "page_number": page.page_number,
            "num_blocks": len(extraction.blocks),
            "validation_status": ("empty_page" if nontrivial_empty else page_validation.status),
            "empty_nontrivial": nontrivial_empty,
            "open_units_after": [u.stitch_id for u in carry.open_units],
            "judge_invoked": bool(judge_record.get("judge_invoked")),
            "judged_count": len(judge_record.get("judged_block_ids") or []),
            "carry_rejections": carry_rejection_record.get("rejected_count", 0),
            "over_seg_runs": over_segmentation_record.get("run_count", 0),
        })
        if page_validation.status == "fail":
            print(f"{prefix}  validator: FAIL ({len(page_validation.issues)} issue(s)) [report-only]")

    final_doc = assembler.assemble(document_id, extractions)
    ddir = run_store.doc_dir(run_dir, doc_name, model_label)
    run_store.write_json(ddir / "final_document.json", final_doc.to_dict())

    final_validation = None
    if run_final_validator:
        try:
            final_validation = final_validator.validate(document_id, final_doc, page_summaries)
            run_store.write_json(ddir / "final_validation.json", final_validation.to_dict())
        except Exception as exc:  # noqa: BLE001
            print(f"{prefix}  final validator error: {exc}")

    summary = {
        "document_id": document_id,
        "model_label": model_label,
        "extractor_model": extractor_model.get("model"),
        "extractor_vision": extractor_vision,
        "pages": len(pages),
        "units": len(final_doc.units),
        "stitch_ids_allocated": allocator.issued,
        "page_errors": page_errors,
        "empty_pages": sum(1 for s in page_summaries if s.get("empty_nontrivial")),
        "judge_pages": sum(1 for s in page_summaries if s.get("judge_invoked")),
        "judge_blocks": sum(s.get("judged_count", 0) for s in page_summaries),
        "carry_rejections": sum(s.get("carry_rejections", 0) for s in page_summaries),
        "over_seg_runs": sum(s.get("over_seg_runs", 0) for s in page_summaries),
        "final_validation_status": (final_validation.status if final_validation else None),
    }
    run_store.write_json(ddir / "run_summary.json", summary)
    return summary


def _persist_page(run_dir, doc_name, page, selection, extraction, validation, carry,
                  model_label=None, region_analysis=None, judge_record=None,
                  carry_rejection_record=None, over_segmentation_record=None) -> None:
    pdir = run_store.page_dir(run_dir, doc_name, page.page_number, model_label)
    if region_analysis is not None:
        run_store.write_json(pdir / "region_analysis.json", region_analysis.to_dict())
    if selection is not None:
        run_store.write_json(pdir / "skill_selection.json", selection.to_dict())
    run_store.write_json(pdir / "extraction.json", extraction.to_dict())
    run_store.write_json(pdir / "validation.json", validation.to_dict())
    run_store.write_json(pdir / "carry_state.json", carry.to_dict())
    # Continuation-judge activity for this page (written every page so the failure detector
    # can tell "no boundary routed" from "judge actually ran" — 'uncertain' itself never
    # survives to extraction.json, so it can't be counted there).
    if judge_record is not None:
        run_store.write_json(pdir / "continuation_judge.json", judge_record)
    # Carry-rejection ledger: which incoming open units the extractor continued vs rejected.
    if carry_rejection_record is not None:
        run_store.write_json(pdir / "carry_rejection.json", carry_rejection_record)
    # Over-segmentation ledger (T1/T4): uniform numbered runs flagged (and, if enabled, re-nested).
    if over_segmentation_record is not None:
        run_store.write_json(pdir / "over_segmentation.json", over_segmentation_record)
