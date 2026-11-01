"""OCR generation (chandra): turns the PDFs in config.BENCHMARK_PDF_ROOT (data/tide/pdf/) into chandra OCR
under config.BENCHMARK_OCR_PARENT (data/tide/ocr/). Heavy chandra imports are lazy, so importing this package
is light. Driven from run.py via RUN_OCR, or standalone."""
