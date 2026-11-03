# Data

Put the two datasets here, in this layout.

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
    ├── repo/MPDocBench/                    MPDocBench repository and gold (MPDocBench.json)
    ├── images/<doc>/page_1.jpg …           page images
    └── ocr/<doc>/                          Chandra OCR, made by `mpdocbench/scripts/ocr_images.py`
        ├── markdown/ html/ layout/ page_images/
        └── <doc>.json                      completion marker
```
