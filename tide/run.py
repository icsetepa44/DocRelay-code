from __future__ import annotations

import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DU = Path(__file__).resolve().parent
sys.path.insert(0, str(DU))

from src.core import orchestrator, config
from src.ocr import chandra_adapter as ocr
from src.storage import document_store as ds
from src.storage import run_store
from src.registry import embedding_scorer
from src.registry import skill_registry
from src.eval import evaluate, paper_metrics


RUN_OCR = False
RUN_PIPELINE = True
RUN_EVAL = True
RESUMABLE = True
RESUME_PAGES = True

RUN_ID = "tide_full"
DOCUMENTS = None
MODELS = None
PAGE_START = 0
PAGE_END = None
RUN_FINAL_VALIDATOR = False
STOP_ON_PAGE_ERROR = False
OVERWRITE_IMAGES = False
MAX_WORKERS = None


def _run_ocr() -> None:
    print("=== OCR generation ===", flush=True)
    from src.ocr.generate import chandra_ocr
    chandra_ocr.main()


def _pair_complete(run_dir: Path, doc: str, label: str) -> bool:
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
            overwrite_images=False,
            stop_on_page_error=STOP_ON_PAGE_ERROR,
            extractor_model=model,
            model_label=label,
            resume=resume,
        )
        return (label, doc, "done")
    except Exception as exc:
        print(f"[{label}] ERROR on {doc}: {exc}", flush=True)
        return (label, doc, f"error: {exc}")


def _build_reports(run_dir: Path) -> None:
    evaluate.evaluate_run(run_dir)
    paper_metrics.compute(run_dir)


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
        skill_registry.get_registry()
        print("Pre-materialising page assets…", flush=True)
        for doc in docs:
            ds.materialize_document(doc, overwrite=OVERWRITE_IMAGES)
        pairs = [(m, doc) for m in models for doc in docs]
        workers = MAX_WORKERS or len(models)
        total = len(pairs)
        print(f"Running {total} (model,doc) pair(s) with {workers} worker thread(s)…", flush=True)
        try:
            from tqdm import tqdm
            _bar = tqdm(total=total, unit="doc", desc="docs")
        except Exception:
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
                except Exception as exc:
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
