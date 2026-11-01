"""
Orchestrator — the v1 stable-core per-page loop (spec §20 minus the §21 failure branch)
followed by deterministic assembly + final validation (spec §29).

Per page (config.USE_SUPERVALIDATOR = True, the active architecture, as in the TIDE pipeline):
           retrieve main skills -> route -> extract (draft) -> SuperValidator (page verifier)
           -> FinalDecider (page corrector; only on a correction-triggering issue or an empty draft)
           -> resolve continuation (deterministic: verifier review overrides the draft's decisions,
              uncertain -> ends, last page -> all ends) -> build carry state -> persist
Ablation (also SKIP_VERIFIER_AND_DECIDER = True): the same path with NO SuperValidator / FinalDecider call —
           the draft goes straight to the deterministic continuation step (uncertain -> ends).
Legacy (USE_SUPERVALIDATOR = False): extract -> validate (report-only) -> resolve continuation
           (judge only when uncertain) -> build carry state -> persist
Then:      assemble -> final validate -> persist FinalDocument + FinalValidation.

Runs for ONE (document, extractor_model). The extractor model is the compared variable;
all other agents use their configured (fixed) models. Each stage sends the page image only
if its resolved model is vision-capable (text-only models run on OCR markdown+html).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.contracts.block import Block
from src.contracts.continuation import CarryState, ContinuationDecision
from src.contracts.page import PageData, PageExtraction
from src.contracts.validation import PageValidation
from src.core import config
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
    underproduction,
    validator,
)
from src.llm import usage
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

    # Simple continuation: the blocks the extractor marked 'uncertain' are what the judge resolves.
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
    domain: Optional[str] = None,
    on_page_done: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """`on_page_done(page_number)` (optional) is called once per processed page — normal or recovered —
    e.g. to drive run.py's page-level progress bar. A failing callback never breaks the run."""
    unresolved_policy = unresolved_policy or config.UNRESOLVED_POLICY

    def _page_done(page_number: int) -> None:
        if on_page_done is not None:
            try:
                on_page_done(page_number)
            except Exception:  # noqa: BLE001 — progress reporting must never break a run
                pass

    registry = get_registry()
    prefix = f"[{model_label}] " if model_label else ""

    # per-domain profile (empty by default -> no behavior change). Skills to force-include
    # (unioned with runtime pins) + additive retrieval weights; guidance rides into the prompts.
    domain_prof = config.domain_profile(domain)
    domain_pins: List[str] = domain_prof["pin_skills"]
    domain_weights: Dict[str, float] = domain_prof["weight_skills"]

    # Read-only load of pre-materialised pages (fallback: materialise if absent).
    pages = ds.load_materialized_pages(doc_name)
    if not pages:
        pages = ds.materialize_document(doc_name, overwrite=overwrite_images)
    pages = pages[page_start:page_end] if page_end is not None else pages[page_start:]
    if not pages:
        raise PipelineError(f"No pages for {doc_name} in range [{page_start}:{page_end}]")
    document_id = pages[0].document_id
    allocator = StitchIdAllocator(document_id)

    # Per-stage vision resolution (extractor from override; others from config).
    extractor_model = extractor_model or config.model_for("extractor")
    extractor_vision = bool(extractor_model.get("vision", False))
    router_vision = config.model_for("skill_router")["vision"]
    validator_vision = config.model_for("validator")["vision"]
    judge_vision = config.model_for("continuation_judge")["vision"]
    region_mode = config.RETRIEVAL_MODE == "region"
    region_vision = config.model_for("region_analyzer")["vision"] if region_mode else False
    use_sv = config.USE_SUPERVALIDATOR
    # Ablation (extractor only): stay on the critic-corrector path but skip both of its LLM calls.
    skip_vd = use_sv and config.SKIP_VERIFIER_AND_DECIDER
    super_validator_vision = config.model_for("super_validator")["vision"] if use_sv and not skip_vd else False
    final_decider_vision = config.model_for("final_decider")["vision"] if use_sv and not skip_vd else False
    if use_sv:  # the legacy validator / judge do not run on the critic-corrector path
        validator_vision = judge_vision = False
    need_image = (extractor_vision or router_vision or validator_vision or judge_vision
                  or region_vision or super_validator_vision or final_decider_vision)

    carry = CarryState.empty()
    extractions: List[PageExtraction] = []
    page_summaries: List[Dict[str, Any]] = []
    page_errors: List[Dict[str, Any]] = []

    for i, page in enumerate(pages):
        next_page = pages[i + 1] if i + 1 < len(pages) else None
        usage.set_tags(page=page.page_number)  # cost/time ledger: tag this page's LLM calls
        print(f"{prefix}[page {page.page_number}] extracting…", flush=True)
        image = ds.load_model_image(page.page_image) if need_image else None

        # --- ESSENTIAL: route + extract. Failure here loses the page's content. ---
        selection_to_persist: Any = None
        region_analysis = None
        try:
            if region_mode:
                region_analysis = region_analyzer.analyze(
                    page, carry, image if region_vision else None, domain=domain)
                table_pins = _open_table_pins(carry)
                per_region_candidates = {
                    r.region_id: retrieve_skills(
                        r.description or page.markdown[:400], ["main"], config.MAIN_TOP_K,
                        pin_skill_ids=table_pins + domain_pins, weight_skill_ids=domain_weights)
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
                    domain=domain,
                )
            else:
                query = _build_query(page, carry)
                candidates = retrieve_skills(query, ["main"], config.MAIN_TOP_K,
                                             pin_skill_ids=_open_table_pins(carry) + domain_pins,
                                             weight_skill_ids=domain_weights)
                selection = skill_router.route(page, carry, candidates,
                                               image if router_vision else None)
                selection_to_persist = selection
                selected = registry.resolve(selection.skills) or candidates
                extraction = extractor.extract(
                    page, next_page, carry, selected,
                    image if extractor_vision else None,
                    model_override=extractor_model,
                    domain=domain,
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
                    model_override=extractor_model, regions=None, ocr_format="markdown",
                    domain=domain)
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
            _page_done(page.page_number)
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
                    domain=domain,
                )
                nontrivial_empty = not extraction.blocks
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}    retry failed: {exc}")
        if nontrivial_empty:
            page_errors.append({"page_number": page.page_number, "stage": "extract",
                                "error": "empty_extraction"})

        # --- Track 2: page under-production guard. The detector always runs (page_underproduction.json);
        #     when PAGE_UNDERPRODUCTION_RETRY is on it re-extracts a flagged page with a corrective nudge,
        #     keeping the highest-coverage attempt. Catches the "1-2 near-empty blocks for a text-rich
        #     page" bail-out that C2's zero-block check misses. Runs before validate/continuation/carry so
        #     downstream sees the recovered blocks. region_analysis is only present in region mode; without
        #     it the region-count clause can't trip, so the guard is inert (by design) outside region mode. ---
        n_regions = len(region_analysis.regions) if region_analysis is not None else 0
        try:
            underproduction_record = underproduction.analyze_underproduction(
                n_regions, [block_text(b) for b in extraction.blocks], page.markdown or "",
                min_regions=config.PAGE_UNDERPRODUCTION_MIN_REGIONS,
                block_ratio=config.PAGE_UNDERPRODUCTION_BLOCK_RATIO,
                min_coverage=config.PAGE_UNDERPRODUCTION_MIN_COVERAGE)
        except Exception as exc:  # noqa: BLE001 — a diagnostic must never break a run
            underproduction_record = {"under_produced": False, "error": str(exc)}
        underproduction_record["page_number"] = page.page_number
        underproduction_record["retried"] = 0
        # Retry only nonzero-but-thin pages (a true zero-block page is C2's job above).
        if (config.PAGE_UNDERPRODUCTION_RETRY and underproduction_record.get("under_produced")
                and extraction.blocks):
            best, best_rec, retried = extraction, underproduction_record, 0
            for attempt in range(1, config.PAGE_UNDERPRODUCTION_MAX_RETRIES + 1):
                retried = attempt
                note = (f"your previous extraction returned only {best_rec['n_blocks']} block(s), but "
                        f"this page has about {n_regions} distinct regions and {best_rec['ocr_chars']} "
                        f"characters of text — you omitted most of the page. Extract EVERY element on "
                        f"the page in reading order; do not stop after the first block.")
                print(f"{prefix}  page {page.page_number}: under-produced "
                      f"({best_rec['n_blocks']} blocks / {n_regions} regions, cov {best_rec['coverage']}) "
                      f"— re-extracting (attempt {attempt})")
                try:
                    retry = extractor.extract(
                        page, next_page, carry, selected, image if extractor_vision else None,
                        model_override=extractor_model,
                        regions=(_regions_guidance(region_analysis, region_selection)
                                 if region_mode else None),
                        domain=domain, retry_note=note)
                except Exception as exc:  # noqa: BLE001 — never break a run on a retry
                    print(f"{prefix}    under-production retry failed: {exc}")
                    break
                rec = underproduction.analyze_underproduction(
                    n_regions, [block_text(b) for b in retry.blocks], page.markdown or "",
                    min_regions=config.PAGE_UNDERPRODUCTION_MIN_REGIONS,
                    block_ratio=config.PAGE_UNDERPRODUCTION_BLOCK_RATIO,
                    min_coverage=config.PAGE_UNDERPRODUCTION_MIN_COVERAGE)
                rec["page_number"] = page.page_number
                if rec["coverage"] > best_rec["coverage"]:
                    best, best_rec = retry, rec
                    print(f"{prefix}    recovered {rec['n_blocks']} blocks (cov {rec['coverage']})")
                if not rec["under_produced"]:
                    break
            extraction, underproduction_record = best, best_rec
            underproduction_record["retried"] = retried

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

        super_validation_record = None
        if use_sv:
            # --- Critic-corrector (as in the TIDE pipeline). NON-FATAL throughout: a critic or
            #     corrector failure keeps the draft; nothing here can drop the page. ---
            super_validation = None
            judge_record = {"page_number": page.page_number, "judge_invoked": False,
                            "corrector_adopted": False, "review_applied": False,
                            "extractor_uncertain": [], "b2_flagged": [], "judged_block_ids": [],
                            "verdicts": []}
            try:
                if not skip_vd:
                    super_validation = super_validator.validate(
                        page, next_page, extraction, carry,
                        image if super_validator_vision else None, domain=domain,
                        draft_coverage=super_validator.coverage_hint(underproduction_record))
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}  super_validator error (ignored, draft kept): {exc}")
                page_errors.append({"page_number": page.page_number, "stage": "super_validator",
                                    "error": str(exc)})

            corrected = False
            if not skip_vd and super_validator.needs_correction(super_validation, extraction):
                judge_record["judge_invoked"] = True  # semantics (as in the TIDE pipeline): corrector fired
                try:
                    corrected_extraction = final_decider.decide(
                        page, next_page, extraction, super_validation, carry,
                        image if final_decider_vision else None, domain=domain)
                    if corrected_extraction and corrected_extraction.blocks:
                        extraction = corrected_extraction
                        corrected = True
                        judge_record["corrector_adopted"] = True
                    else:
                        print(f"{prefix}  final_decider returned 0 blocks (draft kept)")
                except Exception as exc:  # noqa: BLE001
                    print(f"{prefix}  final_decider error (draft kept): {exc}")
                    page_errors.append({"page_number": page.page_number, "stage": "final_decider",
                                        "error": str(exc)})

            # Deterministic continuation resolution.
            final_cont = extraction.continuation or ContinuationDecision(page.page_number, [])
            if next_page is None:
                for d in final_cont.decisions:
                    d.decision = "ends"
            else:
                # The verifier's review refers to the DRAFT's block ids. The corrector's output is
                # renumbered by reading order, so the review is applied only when the draft was kept;
                # an adopted correction already carries the review (its prompt applies it).
                if super_validation and super_validation.continuation_review and not corrected:
                    rev_by_id = {c.block_id: c for c in super_validation.continuation_review}
                    for d in final_cont.decisions:
                        rev = rev_by_id.get(d.block_id)
                        if rev and rev.decision in ("continues", "ends"):
                            d.decision = rev.decision
                            judge_record["review_applied"] = True
                for d in final_cont.decisions:
                    if d.decision == "uncertain":
                        d.decision = "ends"
            extraction.continuation = final_cont

            if super_validation:
                super_validation_record = super_validation.to_dict()
                page_validation = PageValidation.from_dict({
                    "page_number": page.page_number,
                    "status": "fail" if super_validation.has_issues else "pass",
                    "issues": [{"block_ids": [i.target_block_id] if i.target_block_id else [],
                                "type": i.issue_type, "description": i.description}
                               for i in super_validation.issues]})
            else:
                page_validation = PageValidation(page_number=page.page_number,
                                                 status="skipped" if skip_vd else "error", issues=[])
        else:
            # --- LEGACY: validator is report-only; never discard the extraction. ---
            try:
                page_validation = validator.validate(page, extraction, carry,
                                                     image if validator_vision else None)
            except Exception as exc:  # noqa: BLE001
                print(f"{prefix}  validator error (ignored, report-only): {exc}")
                page_validation = PageValidation(page_number=page.page_number, status="error", issues=[])
                page_errors.append({"page_number": page.page_number, "stage": "validate", "error": str(exc)})

            # --- NON-FATAL: continuation resolution + carry. On judge failure, coerce. ---
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
                  "state": u.state, "open_text": u.open_text}
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
                      carry_rejection_record, over_segmentation_record, underproduction_record,
                      super_validation_record)
        extractions.append(extraction)
        _page_done(page.page_number)
        page_summaries.append({
            "page_number": page.page_number,
            "num_blocks": len(extraction.blocks),
            "validation_status": ("empty_page" if nontrivial_empty else page_validation.status),
            "empty_nontrivial": nontrivial_empty,
            "open_units_after": [u.stitch_id for u in carry.open_units],
            "judge_invoked": bool(judge_record.get("judge_invoked")),
            "corrector_adopted": bool(judge_record.get("corrector_adopted")),
            "judged_count": len(judge_record.get("judged_block_ids") or []),
            "carry_rejections": carry_rejection_record.get("rejected_count", 0),
            "over_seg_runs": over_segmentation_record.get("run_count", 0),
        })
        if page_validation.status == "fail":
            if use_sv:
                outcome = ("adopted" if judge_record.get("corrector_adopted")
                           else "draft kept" if judge_record.get("judge_invoked") else "not needed")
                print(f"{prefix}  super_validator: {len(page_validation.issues)} issue(s); corrector {outcome}")
            else:
                print(f"{prefix}  validator: FAIL ({len(page_validation.issues)} issue(s)) [report-only]")

    final_doc = assembler.assemble(document_id, extractions)
    ddir = run_store.doc_dir(run_dir, doc_name, model_label)
    run_store.write_json(ddir / "final_document.json", final_doc.to_dict())

    final_validation = None
    usage.set_tags(page=None)  # the final validator's call is per document, not per page
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
        "use_supervalidator": use_sv,
        "skip_verifier_and_decider": skip_vd,
        # On the critic-corrector path judge_pages = pages where the FinalDecider fired (TIDE pipeline
        # semantics); corrector_adopted_pages = pages whose corrected extraction replaced the draft.
        "judge_pages": sum(1 for s in page_summaries if s.get("judge_invoked")),
        "corrector_adopted_pages": sum(1 for s in page_summaries if s.get("corrector_adopted")),
        "judge_blocks": sum(s.get("judged_count", 0) for s in page_summaries),
        "carry_rejections": sum(s.get("carry_rejections", 0) for s in page_summaries),
        "over_seg_runs": sum(s.get("over_seg_runs", 0) for s in page_summaries),
        "final_validation_status": (final_validation.status if final_validation else None),
    }
    run_store.write_json(ddir / "run_summary.json", summary)
    return summary


def _persist_page(run_dir, doc_name, page, selection, extraction, validation, carry,
                  model_label=None, region_analysis=None, judge_record=None,
                  carry_rejection_record=None, over_segmentation_record=None,
                  underproduction_record=None, super_validation_record=None) -> None:
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
    # Track 2 page-drop guard ledger: whether the page looked under-produced (and any retry outcome).
    if underproduction_record is not None:
        run_store.write_json(pdir / "page_underproduction.json", underproduction_record)
    # SuperValidator critique in full (issues + suggested fixes + continuation review); validation.json
    # holds the issues in PageValidation shape. Critic-corrector path only.
    if super_validation_record is not None:
        run_store.write_json(pdir / "super_validation.json", super_validation_record)
