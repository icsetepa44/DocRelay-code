"""
Single entry point for the TIDE extraction pipeline.

No CLI — configure the run by editing the GLOBALS below, then (from the repository root):

    python tide/run.py

Stages (each gated by a boolean, run in this order):
  RUN_OCR      -> generate Chandra OCR (data/tide/pdf/ -> data/tide/ocr/); needs the Chandra model + a GPU
  RUN_PIPELINE -> run the extraction pipeline (materialise page assets + per-model/per-document loop)
  RUN_EVAL     -> adapt + evaluate (writes runs/<RUN_ID>/_eval/: metrics.json, summary.md, final_documents/)
                  and the paper's TIDE metrics (paper_metrics.json: Text Edit, Table TEDS, C-P, C-P P/R/F1)

RESUMABLE writes into a FIXED run dir (RUN_ID) so an interrupted run RESUMES: completed
(doc, model) pairs are skipped and partially-written pairs resume from their last extracted page.
With RESUMABLE = False each run goes to a fresh timestamped dir.

Eval-only on an existing run: set RUN_PIPELINE = False, RUN_EVAL = True, and point RUN_ID at
that run (its dir is reused).

All pipeline behaviour (models, retrieval, carry state, critic-corrector, request parameters) lives in
config/pipeline.yaml and config/models.yaml; this file is orchestration only.
"""

from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DU = Path(__file__).resolve().parent
sys.path.insert(0, str(DU))

from src.core import orchestrator, config          # noqa: E402
from src.ocr import chandra_adapter as ocr          # noqa: E402
from src.storage import document_store as ds         # noqa: E402
from src.storage import run_store                    # noqa: E402
from src.registry import embedding_scorer            # noqa: E402
from src.registry import skill_registry              # noqa: E402
from src.eval import evaluate, paper_metrics          # noqa: E402
# OCR generation is imported lazily inside _run_ocr() so this runner
# stays importable without the heavy OCR deps (chandra / cv2 / pypdfium2).

# ============================================================
# RUN CONFIG (edit these globals)
# ============================================================

# --- what to run ---
RUN_OCR = False        # generate OCR (data/tide/pdf/ -> data/tide/ocr/) via Chandra; needs the model
RUN_PIPELINE = True    # run the extraction pipeline
RUN_EVAL = True        # adapt + evaluate + paper metrics after the pipeline (needs the gold in data/tide/json/)
RESUMABLE = True       # skip completed (doc,model) pairs into a fixed run dir; False -> fresh timestamped run
RESUME_PAGES = True    # when RESUMABLE and a pair is partial, RESUME from its last fully-extracted page
                       # instead of discarding it. Set False for one clean run if you changed prompts/
                       # config/pipeline.yaml since the interrupted run (reused pages keep the OLD config).

# --- run parameters (run-orchestration ONLY; all pipeline config lives in config/pipeline.yaml) ---
RUN_ID = "tide_full"             # fixed dir name under tide/runs/ (used when RESUMABLE, or for eval-only reuse)
DOCUMENTS = None                 # None -> every document with OCR under data/tide/ocr/; or a list of names
MODELS = None                    # None -> all config.extractor_models()
PAGE_START = 0
PAGE_END = None
RUN_FINAL_VALIDATOR = False      # optional report-only document-level LLM check (final_validation.json); it never
                                 # changes the output and is not part of the method described in the paper
STOP_ON_PAGE_ERROR = False
OVERWRITE_IMAGES = False
MAX_WORKERS = None               # concurrent (model, doc) pairs; None -> one worker per extractor model


# ============================================================
# OCR generation
# ============================================================

def _run_ocr() -> None:
    """Generate OCR into data/tide/ocr/."""
    print("=== OCR generation ===", flush=True)
    from src.ocr.generate import chandra_ocr
    chandra_ocr.main()


# ============================================================
# Pipeline
# ============================================================

def _pair_complete(run_dir: Path, doc: str, label: str) -> bool:
    """A (doc, model) pair is complete iff final_document.json AND run_summary.json both
    exist and parse as valid JSON (both written at the end, so a torn write reads as incomplete)."""
    import json
    d = run_dir / doc / label
    for name in ("final_document.json", "run_summary.json"):
        p = d / name
        if not p.exists():
            return False
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
    return True


def _run_one_pair(model: dict, doc: str, run_dir: Path) -> tuple:
    """Run ONE (model, doc). Safe to call concurrently from worker threads: it writes only
    under run_dir/<doc>/<label>/ (namespaced per doc) and uses a per-doc StitchIdAllocator.
    Preconditions (done once before the pool): materialize assets, warm singletons, set config."""
    label = config.model_label(model["model"])
    if RESUMABLE and _pair_complete(run_dir, doc, label):
        print(f"[{label}] SKIP completed: {doc}", flush=True)
        return (label, doc, "skipped")
    pair_dir = run_dir / doc / label
    resume = False
    if RESUMABLE and pair_dir.exists():
        if RESUME_PAGES:
            print(f"[{label}] partial -> RESUME from last extracted page: {doc}", flush=True)
            resume = True
        else:
            print(f"[{label}] partial -> reset & rerun: {doc}", flush=True)
            shutil.rmtree(pair_dir, ignore_errors=True)
    else:
        print(f"[{label}] run: {doc}", flush=True)
    try:
        orchestrator.run_document(
            doc_name=doc,
            run_dir=run_dir,
            page_start=PAGE_START,
            page_end=PAGE_END,
            run_final_validator=RUN_FINAL_VALIDATOR,
            # unresolved_policy omitted -> orchestrator uses config.UNRESOLVED_POLICY (config/pipeline.yaml)
            overwrite_images=False,          # assets pre-materialised once below
            stop_on_page_error=STOP_ON_PAGE_ERROR,
            extractor_model=model,
            model_label=label,
            resume=resume,
        )
        return (label, doc, "done")
    except Exception as exc:  # noqa: BLE001 — keep other pairs going; a rerun retries this one
        print(f"[{label}] ERROR on {doc}: {exc}", flush=True)
        return (label, doc, f"error: {exc}")


def _build_reports(run_dir: Path) -> None:
    """Adapt + evaluate predictions, then the paper's TIDE metrics (order matters)."""
    evaluate.evaluate_run(run_dir)  # -> _eval/{final_documents, metrics.json, summary.md, integrity.json}
    paper_metrics.compute(run_dir)  # -> _eval/{paper_metrics.json, boundary_labels.csv}


def main() -> int:
    if RUN_OCR:
        _run_ocr()

    if not (RUN_PIPELINE or RUN_EVAL):
        print("Nothing else to do (RUN_PIPELINE and RUN_EVAL are both False).")
        return 0

    if RUN_PIPELINE and not config.has_llm_key():
        print("API key not set: please set OPENROUTER_API_KEY in .env (see .env.example). Aborting.")
        return 2

    docs = list(DOCUMENTS) if DOCUMENTS else ocr.list_available_documents()
    models = MODELS or config.extractor_models()
    if not docs or not models:
        print(f"No documents (OCR expected under {config.BENCHMARK_OCR_PARENT}) or no extractor models configured.")
        return 1

    # Fixed run dir when resuming, or when doing eval-only on an existing run; else timestamped.
    use_fixed = RESUMABLE or (RUN_EVAL and not RUN_PIPELINE)
    run_dir = run_store.create_run_dir(RUN_ID if use_fixed else None)
    print(f"Run dir: {run_dir} (resumable={RESUMABLE})")
    print(f"Retrieval: scorer={config.RETRIEVAL_SCORER} mode={config.RETRIEVAL_MODE} "
          f"main_top_k={config.MAIN_TOP_K}")
    print(f"Extractor input: ocr_format={config.OCR_INPUT_FORMAT} "
          f"carry_open_text_max_chars={config.CARRY_OPEN_TEXT_MAX_CHARS}")
    print(f"Critic-corrector: use_supervalidator={config.USE_SUPERVALIDATOR}")
    print(f"Documents: {len(docs)}" + (f" (first: {docs[:3]}…)" if len(docs) > 3 else f": {docs}"))
    print(f"Extractor models: {[m['model'] for m in models]} (vision: {[m['vision'] for m in models]})")

    if RUN_PIPELINE:
        if config.RETRIEVAL_SCORER == "embedding":
            print("Warming up embedding model…", flush=True)
            dim = embedding_scorer.warm_up()
            print(f"Embedding model ready (dim={dim}).", flush=True)
        skill_registry.get_registry()   # pre-warm the lazy singleton before the pool (avoids a first-use race)
        print("Pre-materialising page assets…", flush=True)
        for doc in docs:
            ds.materialize_document(doc, overwrite=OVERWRITE_IMAGES)
        # One task per (model, doc).
        pairs = [(m, doc) for m in models for doc in docs]
        workers = MAX_WORKERS or len(models)
        total = len(pairs)
        print(f"Running {total} (model,doc) pair(s) with {workers} worker thread(s)…", flush=True)
        try:
            from tqdm import tqdm            # optional: live bar if installed
            _bar = tqdm(total=total, unit="doc", desc="docs")
        except Exception:                     # noqa: BLE001 — dependency-free fallback
            _bar = None

        def _hms(s: float) -> str:
            s = int(s); return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_one_pair, m, doc, run_dir): (m, doc) for m, doc in pairs}
            for fut in as_completed(futures):
                m, doc = futures[fut]
                done += 1
                try:
                    label, d, status = fut.result()
                except Exception as exc:  # noqa: BLE001
                    label, d, status = config.model_label(m["model"]), doc, f"error: {exc}"
                el = time.time() - t0
                eta = (el / done) * (total - done) if done else 0.0
                if _bar is not None:
                    _bar.update(1)
                    _bar.set_postfix_str(f"{total-done} left | {d} -> {status}")
                else:
                    print(f"[{done}/{total}] {100*done//total}% | {total-done} left | "
                          f"elapsed {_hms(el)} | ETA ~{_hms(eta)} | {label}/{d} -> {status}", flush=True)
        if _bar is not None:
            _bar.close()

    if RUN_EVAL:
        # Defer eval until EVERY pair is complete, so an interrupted run never emits partial metrics.
        incomplete = [(doc, config.model_label(m["model"]))
                      for m in models for doc in docs
                      if not _pair_complete(run_dir, doc, config.model_label(m["model"]))]
        if incomplete:
            print(f"\n{len(incomplete)} pair(s) still incomplete: {incomplete}")
            print("Re-run to finish them; eval is deferred until all pairs complete.")
            return 3
        print("\nBuilding reports (adapt + evaluate + paper metrics)…", flush=True)
        _build_reports(run_dir)
        print(f"\nDone. See {run_dir / '_eval'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
