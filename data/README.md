# Data

No dataset files ship with the code. Put the two datasets here, in this layout. The whole folder except this README
is git-ignored.

```
data/
├── tide/                                   TIDE corpus
│   ├── pdf/<name>.pdf                      source PDFs (page images + OCR input)
│   ├── json/<name>_output.json             gold annotations (evaluation only)
│   └── ocr/<name>/                         Chandra OCR, made by `tide/run.py` with RUN_OCR = True
│       ├── markdown/page_0000.md
│       ├── html/page_0000.html
│       ├── layout/page_0000.json
│       └── <name>.json                     completion marker
└── mpdocbench/                             MPDocBench-Parse
    ├── repo/MPDocBench/                    official code + gold (MPDocBench.json)
    ├── images/<doc>/page_1.jpg …           page images
    └── ocr/<doc>/                          Chandra OCR, made by `mpdocbench/scripts/ocr_images.py`
        ├── markdown/ html/ layout/ page_images/
        └── <doc>.json                      completion marker
```

## TIDE

Get the corpus from its release (see the paper). Put each document's PDF in `pdf/` and its gold file in `json/`. The
two names must match: `<name>.pdf` and `<name>_output.json`. Then make the OCR (see the main README), or put an
existing Chandra OCR tree in `ocr/`.

## MPDocBench-Parse

1. **Official code and gold.** This is the Tongyi-Zhiwen/Qwen-Doc repository, Apache-2.0. The evaluator imports its
   metric code unchanged, so it must be at `repo/MPDocBench/`:

   ```bash
   git clone --filter=blob:none --sparse https://github.com/Tongyi-Zhiwen/Qwen-Doc data/mpdocbench/repo
   git -C data/mpdocbench/repo sparse-checkout set MPDocBench
   git -C data/mpdocbench/repo checkout 4e5aee32280febb7b0844237ba087f837e3c8905   # the version used
   ```

2. **Page images.** Download `MPDocBench_data.zip` from the ModelScope dataset `zhoubb/MPDocBench`. Extract it so that
   the page images sit at `images/<doc>/page_N.jpg`.
   - Some `pptSlides` documents come from SlideVQA and are not in the archive. The official repository has a
     downloader for them, `tools/download_data_from_slidevqa.py`.

3. **OCR.** Run `python mpdocbench/scripts/ocr_images.py`. It writes `ocr/<doc>/`.
   - Flat `ocr/<doc>/` folders and category-nested `ocr/<domain>/<doc>/` folders are both read.
   - To run English and Chinese as separate runs, give each its own OCR root (`inputs.ocr_root` in
     `mpdocbench/config/pipeline.yaml`).
