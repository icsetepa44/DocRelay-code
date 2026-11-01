# DocRelay: page-by-page extraction of long, visually complex documents

This is the code for the DocRelay extraction pipeline and its evaluation on two datasets: **TIDE**, a corpus of
Bangladeshi customs documents (Bangla, English, mixed), and **MPDocBench-Parse** (English and Chinese).

The pipeline reads a document one page at a time. For each page it extracts structured blocks, using the page's OCR
and page image. A short **carry state** passes from page to page and records which units are still open, so a
paragraph, numbered point or table that crosses a page break is rebuilt as one unit. A deterministic assembler then
builds the final document.

## What runs, in order

| Paper (Methodology) | Code (`src/`) | Model (`config/models.yaml`) |
|---|---|---|
| OCR of every page (made in advance) | `ocr/generate/chandra_ocr.py` (TIDE), `scripts/ocr_images.py` (MPDocBench) | `datalab-to/chandra-ocr-2` |
| Region analysis | `stages/region_analyzer.py` | `google/gemma-4-31b-it` |
| Guideline retrieval, top 2 | `registry/skill_retriever.py`; the guidelines are `skills/records/` | `intfloat/multilingual-e5-small` |
| Guideline selection | `stages/skill_router.py` | `google/gemma-4-31b-it` |
| Page extractor (draft) | `stages/extractor.py` | `google/gemma-4-31b-it` |
| Page verifier | `stages/super_validator.py` | `deepseek/deepseek-v4.1-flash` |
| Page corrector (only when the verifier flags an issue) | `stages/final_decider.py` | `deepseek/deepseek-v4.1-flash` |
| Continuation resolution and state update | `core/orchestrator.py`, `stages/carry_builder.py` | — (deterministic) |
| Document assembly | `stages/assembler.py` | — (deterministic) |

`core/orchestrator.py` runs this per-page loop. The prompts are in `prompts/` and the JSON output schemas in
`schemas/`. Sampling and request settings are in `config/pipeline.yaml`:
- temperature 0.0, top-p 0.1;
- at most 16,384 output tokens;
- page images sent to the models at 1,024 px on the long side (TIDE pages are rendered at 300 DPI);
- OpenRouter with data collection denied, and a per-model precision pin.

## Repository layout

```
DocRelay-code/
├── README.md
├── requirements.txt           pipeline + evaluation (Python 3.11)
├── requirements-ocr.txt       only for regenerating OCR (GPU)
├── .env.example               OPENROUTER_API_KEY
├── data/README.md             where the datasets go (no data is shipped)
├── tide/                      TIDE pipeline + evaluation
│   ├── run.py                 single entry point (edit the toggles at the top)
│   ├── config/                pipeline.yaml, models.yaml, block_types.json
│   ├── prompts/  schemas/  skills/
│   ├── src/{core,contracts,stages,llm,registry,storage,ocr,eval}/
│   └── tests/unit/
└── mpdocbench/                MPDocBench-Parse pipeline + evaluation
    ├── run.py
    ├── config/                pipeline.yaml, models.yaml, block_types.json, domain_profiles.yaml
    ├── prompts/  schemas/  skills/
    ├── scripts/ocr_images.py  Chandra OCR of the benchmark's page images
    ├── src/…                  same layout as tide/src
    └── tests/unit/
```

The two packages share the same architecture, models and settings. They differ where the datasets differ, so each is
self-contained:
- **TIDE** renders PDF pages and has legal clause and sub-point rules.
- **MPDocBench** reads the benchmark's page images and Chandra layout boxes. It also has more block types (figure,
  equation, caption, list, code), per-domain guidance and the official MPDocBench metric code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.11
pip install -r requirements.txt
pip install -r requirements-ocr.txt                      # only to regenerate OCR (needs a GPU)
cp .env.example .env                                     # then set OPENROUTER_API_KEY
```

- All language models are called through [OpenRouter](https://openrouter.ai), and each pipeline run is billed to your
  key.
- The embedding model and the Chandra OCR model download from Hugging Face on first use.
- Put the datasets in `data/` as described in [`data/README.md`](data/README.md).

## TIDE

Configure `tide/run.py` with the globals at the top; there is no command line. Then run from the repository root:

```bash
python tide/run.py
```

| Toggle | Default | Does |
|---|---|---|
| `RUN_OCR` | `False` | Chandra OCR of `data/tide/pdf/` → `data/tide/ocr/` (GPU, `requirements-ocr.txt`) |
| `RUN_PIPELINE` | `True` | extracts every document that has OCR (or the names in `DOCUMENTS`) |
| `RUN_EVAL` | `True` | scores the run against `data/tide/json/` |
| `RESUMABLE` | `True` | fixed run folder `tide/runs/<RUN_ID>/`; finished documents are skipped, and interrupted ones resume at their last page |

To evaluate an existing run without new model calls, set `RUN_PIPELINE = False`, `RUN_EVAL = True` and `RUN_ID` to
that run's folder name.

**Outputs** go to `tide/runs/<RUN_ID>/`:
- `<doc>/<model>/final_document.json`: the extracted document. Its units have a `parent_unit_id` hierarchy, and a
  unit that crosses pages is one unit.
- `<doc>/<model>/pages/<page>/`: per-page intermediate files: region analysis, guideline selection, extraction,
  validation, carry state, continuation records.
- `<doc>/<model>/run_summary.json`.
- `_eval/metrics.json`, `_eval/summary.md`, `_eval/final_documents/`: per-document scores and adapted predictions.
- `_eval/paper_metrics.json`: the metrics reported in the paper (below). `_eval/boundary_labels.csv` holds the label
  of every page boundary.

**Metrics** (`src/eval/paper_metrics.py`):

| Metric | Direction | Definition |
|---|---|---|
| Text Edit | ↓ | 1 − character edit similarity of the whole document text, averaged over documents (`evaluate.py`) |
| Table TEDS | ↑ | tree-edit similarity of the tables, averaged over documents that have tables (`table_structure.py`) |
| C-P | ↓ | normalised edit distance of every gold clause that crosses a page against its best-matching single predicted item (text cleaned and normalised first: case, spacing, punctuation and digit script ignored); per document, summed edits / summed lengths, then averaged over documents (`cp_edit.py`) |
| C-P precision / recall / F1 | ↑ | every page boundary is labelled: joined correctly (TP), wrongly joined (FP) or wrongly split (FN); pooled over all boundaries (`boundary_labels.py`) |

## MPDocBench-Parse

1. **OCR.** Make it once from the page images: `python mpdocbench/scripts/ocr_images.py`. This needs a GPU and
   `requirements-ocr.txt`.
2. **Pipeline and evaluation.** Configure `mpdocbench/run.py` with the globals at the top, then run
   `python mpdocbench/run.py`.

| Toggle | Default | Does |
|---|---|---|
| `RUN_OCR_PREFLIGHT` | `True` | read-only OCR check; a hard problem (missing pages, images, layout or gold) stops the run before any model call |
| `RUN_PIPELINE` | `True` | extracts every document under `data/mpdocbench/ocr/` (or the stems in `DOCUMENTS`) |
| `RUN_EVAL` | `True` | official MPDocBench metrics, the paper table, and cost and time |
| `PAPER_TABLE_RUNS` | `None` | run ids whose documents are pooled into one paper table, e.g. a separate English and Chinese run |
| `QUIET` | `True` | console shows a progress bar only; the full log is `runs/<RUN_ID>/run.log` |

**Outputs** go to `mpdocbench/runs/<RUN_ID>/`:
- `<doc>/<model>/…`: the same files as TIDE.
- `_mpdocbench/metrics.json` and `weak_spots.md`: per-document and aggregate scores.
- `_mpdocbench/paper_table.json`: the paper's six columns and Overall.
- `_cost_time/summary.json`: provider-reported cost and wall time, from a ledger of every model call.

**Metrics.** These come from the official MPDocBench code, imported from `data/mpdocbench/repo/MPDocBench` and used
unchanged:
- Text Edit, Truncated Text Edit and Read Order (↓);
- Table TEDS, Truncated Table TEDS and Heading TEDS (↑);
- Overall = the mean of these six, with each edit distance counted as (1 − edit) × 100 (`src/eval/paper_table.py`).

The evaluator also reports formula (CDM) and figure scores, which are not part of Overall.
- CDM needs xelatex, ImageMagick, node, scikit-image and a CJK font. That toolchain was available under WSL/Linux.
- Without it, the formula score falls back to a LaTeX edit distance, and the method label says so.

## Configuration notes

- **Single source of truth.** Pipeline behaviour is set only in `config/pipeline.yaml` and `config/models.yaml`.
  `run.py` holds run orchestration only.
- **Critic and corrector.** `use_supervalidator: true` turns on the page verifier and the page corrector.
  - `false` gives an older baseline, where a report-only validator and a continuation judge run instead. Its code is
    kept for that switch.
  - In `mpdocbench/config/pipeline.yaml`, `skip_verifier_and_decider: true` keeps the draft without any verifier or
    corrector call.
- **Final validator.** `RUN_FINAL_VALIDATOR` in `run.py` is off by default. It is an optional document-level check
  that is only reported: it never changes the output and is not part of the method in the paper.
- **Repeat runs.** Model outputs can vary between runs even at temperature 0, because providers do not guarantee
  determinism.

## Difficulty router (not included)

The paper's difficulty router is released separately. It decides, for each document, whether the document goes through
this pipeline or to a proprietary model. To combine the two:
1. Run the router first.
2. Put the documents routed to the pipeline in `DOCUMENTS` in `tide/run.py`.
3. Score the two groups with the same evaluation code.

## Tests

```bash
(cd tide && python -m pytest)
(cd mpdocbench && python -m pytest)
```

The tests are synthetic and make no model calls.
- A few tests run only when their inputs are present, and otherwise skip: the real-OCR smoke test, and the CDM
  toolchain.
- The MPDocBench metric tests need the official code in `data/mpdocbench/repo/`.

## Licences

Before you use the models and the official MPDocBench code, check their terms:
- Gemma, DeepSeek, multilingual-e5 and Chandra OCR each have their own licence.
- The official MPDocBench code is Apache-2.0.
