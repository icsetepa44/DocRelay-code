# Router (DINOv3)

Fast PDF difficulty classification (HARD vs SOFT).

---

## 1. Installation

```bash
pip install -r requirements.txt
```

---

## 2. CLI Usage

### Predict a single PDF
```bash
python router.py predict --pdf /path/to/document.pdf
```

### High-recall mode (catches >=85% difficult documents)
```bash
python router.py predict --pdf /path/to/document.pdf --mode protect_quality
```

### Batch route a directory of PDFs
```bash
python router.py batch --dir /path/to/pdf_folder/ --output results.csv
```

---

## 3. Python API

```python
from router import DocumentRouter

router = DocumentRouter()

# Single PDF
result = router.route("document.pdf", mode="balanced")
print(result["label"])  # "HARD" or "SOFT"

# Batch PDFs
results = router.route_batch(["doc1.pdf", "doc2.pdf"])
```

---

## 4. Run Tests

```bash
python -m unittest discover -s tests
```

