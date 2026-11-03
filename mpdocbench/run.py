from __future__ import annotations

import contextlib
import datetime
import shutil
import sys
import threading
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
from src.eval import mpdocbench
from src.eval import cost_time, paper_table
from src.llm import usage


RUN_PIPELINE = True
RUN_EVAL = True
RESUMABLE = True
RUN_OCR_PREFLIGHT = True

RUN_ID = "mpdocbench_full"
DOCUMENTS = None
MODELS = None
PAPER_TABLE_RUNS = None
PAGE_START = 0
PAGE_END = None
RUN_FINAL_VALIDATOR = False
STOP_ON_PAGE_ERROR = False
OVERWRITE_IMAGES = False
MAX_WORKERS = 8
QUIET = True


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


def _run_one_pair(model: dict, doc: str, run_dir: Path, on_page_done=None) -> tuple:
    label = config.model_label(model["model"])
    if RESUMABLE and _pair_complete(run_dir, doc, label):
        print(f"[{label}] SKIP completed: {doc}", flush=True)
        return label, doc, "skipped"
    pair_dir = run_dir / doc / label
    if RESUMABLE and pair_dir.exists():
        print(f"[{label}] partial -> reset & rerun: {doc}", flush=True)
        shutil.rmtree(pair_dir, ignore_errors=True)
    else:
        print(f"[{label}] run: {doc}", flush=True)
    started, t0 = usage.now_iso(), time.perf_counter()
    try:
        with usage.tags(doc=doc, label=label):
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
                domain=config.domain_for(doc),
                on_page_done=on_page_done,
            )
        status, error = "done", None
    except Exception as exc:
        print(f"[{label}] ERROR on {doc}: {exc}", flush=True)
        status, error = "error", str(exc)
    usage.append(usage.PAIRS_FILE, {"doc": doc, "label": label, "start": started, "end": usage.now_iso(),
                                    "duration_s": round(time.perf_counter() - t0, 1), "status": status,
                                    "error": error})
    return label, doc, (status if error is None else f"error: {error}")


def _build_reports(run_dir: Path) -> None:
    print("  MPDocBench eval -> _mpdocbench/metrics.json", flush=True)
    mpdocbench.evaluate_run(run_dir)
    runs = [config.RUNS_DIR / r for r in PAPER_TABLE_RUNS] if PAPER_TABLE_RUNS else [run_dir]
    print(f"  paper table ({[r.name for r in runs]}) -> _mpdocbench/paper_table.json", flush=True)
    paper_table.build(runs, out_path=run_dir / "_mpdocbench" / "paper_table.json")
    if cost_time.summarize(run_dir) is not None:
        print("  cost + time -> _cost_time/summary.json", flush=True)


def _pages_in_range(n: int) -> int:
    return len(range(n)[PAGE_START:PAGE_END])


class _TqdmStdout:
    def __init__(self, tqdm_cls, target):
        self._tqdm, self._target, self._buf = tqdm_cls, target, ""
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self._tqdm.write(line, file=self._target)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            rest, self._buf = self._buf, ""
        if rest:
            self._tqdm.write(rest, file=self._target)
        self._target.flush()


def _doc_progress(done: int, total: int, errors: int) -> str:
    return f"docs {done}/{total} done, {errors} errors"


_BAR_FORMAT = "{desc} |{bar}| {n_fmt}/{total_fmt} pages{postfix} | elapsed {elapsed} | ETA {remaining}"


class _Bars:
    def __init__(self, console_err):
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        self.tqdm, self.file = tqdm, console_err
        self.lock = threading.Lock()
        self.run = None
        self.run_pages = 0

    def open_run(self, desc: str, total_pages: int, done_pages: int, total_docs: int, done_docs: int) -> None:
        self.run_pages = done_pages
        if self.tqdm is None:
            return
        try:
            self.run = self.tqdm(total=total_pages, initial=done_pages, unit="page", desc=desc, position=0,
                                 dynamic_ncols=True, smoothing=0.05, file=self.file, bar_format=_BAR_FORMAT,
                                 postfix=_doc_progress(done_docs, total_docs, 0))
        except Exception:
            self.run = None

    def tick(self, _page_number: int = 0) -> None:
        with self.lock:
            self.run_pages += 1
            if self.run is not None:
                self.run.update(1)

    def docs(self, done: int, total: int, errors: int) -> None:
        with self.lock:
            if self.run is not None:
                self.run.set_postfix_str(_doc_progress(done, total, errors))

    def close(self) -> None:
        if self.run is not None:
            self.run.close()
        self.run = None

    def active(self) -> bool:
        return self.run is not None

    def write(self, msg: str) -> None:
        self.tqdm.write(msg, file=self.file)


def _preflight(docs: list, run_dir: Path) -> bool:
    import json
    from src.ocr import preflight
    report = preflight.run_preflight(docs, gold_stems=set(mpdocbench.load_gold_index()))
    (run_dir / "_preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    preflight.format_summary(report)
    return report["ok"]


def main() -> int:
    console_out, console_err = sys.stdout, sys.stderr
    bars = _Bars(console_err)

    def say(msg: str) -> None:
        if bars.active():
            bars.write(msg)
        else:
            print(msg, file=console_out, flush=True)

    if not (RUN_PIPELINE or RUN_EVAL):
        say("Nothing to do (RUN_PIPELINE and RUN_EVAL both False).")
        return 0

    if RUN_PIPELINE and not config.has_openrouter_key():
        say("OPENROUTER_API_KEY not set (repository-root .env, see .env.example). Aborting.")
        return 2

    models = MODELS or config.extractor_models()
    docs = list(DOCUMENTS) if DOCUMENTS else ocr.list_available_documents()
    if not docs or not models:
        say(f"No documents (OCR expected under {config.BENCHMARK_OCR_PARENT}) or no extractor models configured.")
        return 1

    use_fixed = RESUMABLE or (RUN_EVAL and not RUN_PIPELINE)
    run_dir = run_store.create_run_dir(RUN_ID if use_fixed else None)
    log_path = run_dir / "run.log"
    with contextlib.ExitStack() as stack:
        if QUIET:
            log = stack.enter_context(open(log_path, "a", encoding="utf-8", buffering=1))
            log.write(f"\n===== run.py {datetime.datetime.now().isoformat(timespec='seconds')} =====\n")
            stack.enter_context(contextlib.redirect_stdout(log))
            stack.enter_context(contextlib.redirect_stderr(log))
        try:
            return _run(docs, models, run_dir, say, log_path, bars)
        finally:
            bars.close()


def _run(docs: list, models: list, run_dir: Path, say, log_path: Path, bars: _Bars) -> int:
    log_note = f" (log: {log_path})" if QUIET else ""
    print(f"Run dir: {run_dir} (resumable={RESUMABLE})")
    print(f"Retrieval: scorer={config.RETRIEVAL_SCORER} mode={config.RETRIEVAL_MODE} "
          f"main_top_k={config.MAIN_TOP_K}")
    print(f"Extractor input: ocr_format={config.OCR_INPUT_FORMAT} "
          f"carry_open_text_max_chars={config.CARRY_OPEN_TEXT_MAX_CHARS}")
    print(f"Critic-corrector: use_supervalidator={config.USE_SUPERVALIDATOR} "
          f"(super_validator={config.model_for('super_validator')['model']}, "
          f"final_decider={config.model_for('final_decider')['model']}); "
          f"skip_verifier_and_decider={config.SKIP_VERIFIER_AND_DECIDER}")
    print(f"Documents: {len(docs)} from {config.BENCHMARK_OCR_PARENT}")
    print(f"Extractor models: {[m['model'] for m in models]} (vision: {[m['vision'] for m in models]})")

    if RUN_PIPELINE and RUN_OCR_PREFLIGHT and not _preflight(docs, run_dir):
        say(f"OCR preflight found HARD problems (see {run_dir / '_preflight.json'}). "
            f"Aborting before any API call.{log_note}")
        return 4

    done, errors, n_pairs = 0, 0, 0
    if RUN_PIPELINE:
        if config.RETRIEVAL_SCORER == "embedding":
            print("Warming up embedding model…", flush=True)
            dim = embedding_scorer.warm_up()
            print(f"Embedding model ready (dim={dim}).", flush=True)
        print("Pre-materialising page assets…", flush=True)
        n_pages = {doc: len(ds.materialize_document(doc, overwrite=OVERWRITE_IMAGES)) for doc in docs}
        pairs = [(m, doc) for m in models for doc in sorted(docs, key=lambda d: -n_pages[d])]
        n_pairs = len(pairs)
        workers = MAX_WORKERS or len(models)
        print(f"Running {n_pairs} (model,doc) pair(s) with {workers} worker thread(s)…", flush=True)

        complete = {(config.model_label(m["model"]), doc) for m, doc in pairs
                    if RESUMABLE and _pair_complete(run_dir, doc, config.model_label(m["model"]))}
        total_pages = sum(_pages_in_range(n_pages[doc]) for _, doc in pairs)
        done_pages = sum(_pages_in_range(n_pages[doc]) for m, doc in pairs
                         if (config.model_label(m["model"]), doc) in complete)
        done = len(complete)
        bars.open_run("pages", total_pages, done_pages, n_pairs, done)

        redirect = (contextlib.redirect_stdout(_TqdmStdout(bars.tqdm, sys.stdout))
                    if bars.run is not None and not QUIET else contextlib.nullcontext())
        finished = 0
        session, t0 = usage.now_iso(), time.perf_counter()
        with usage.ledger(run_dir, session):
            try:
                with redirect, ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_run_one_pair, m, doc, run_dir, bars.tick): (m, doc) for m, doc in pairs}
                    for fut in as_completed(futures):
                        m, doc = futures[fut]
                        finished += 1
                        try:
                            label, d, status = fut.result()
                        except Exception as exc:
                            label, d, status = config.model_label(m["model"]), doc, f"error: {exc}"
                        if status.startswith("error"):
                            errors += 1
                        elif status != "skipped":
                            done += 1
                        print(f"=== [{finished}/{n_pairs}] {label} {d} -> {status} ===", flush=True)
                        bars.docs(done, n_pairs, errors)
            finally:
                usage.append(usage.SESSIONS_FILE, {"start": session, "end": usage.now_iso(),
                                                   "wall_s": round(time.perf_counter() - t0, 1), "pairs": n_pairs,
                                                   "skipped": len(complete), "done": done - len(complete),
                                                   "errors": errors, "workers": workers})
        pages_done = bars.run_pages
        bars.close()
        print(f"Pages: {pages_done}/{total_pages} | {_doc_progress(done, n_pairs, errors)}", flush=True)

    if RUN_EVAL:
        incomplete = [(doc, config.model_label(m["model"]))
                      for m in models for doc in docs
                      if not _pair_complete(run_dir, doc, config.model_label(m["model"]))]
        if incomplete:
            print(f"\n{len(incomplete)} pair(s) still incomplete: {incomplete}")
            say(f"{len(incomplete)} doc(s) still incomplete ({errors} error(s) this run) — re-run to finish "
                f"them; eval is deferred until all docs complete.{log_note}")
            return 3
        print("\nBuilding reports…", flush=True)
        _build_reports(run_dir)
        out = run_dir / "_mpdocbench"
        print(f"\nDone. See {out}")
        docs_note = f"{done}/{n_pairs} docs, {errors} errors -> " if RUN_PIPELINE else ""
        say(f"Done: {docs_note}results in {out}{log_note}")
    elif RUN_PIPELINE:
        say(f"Done: {_doc_progress(done, n_pairs, errors)}{log_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
