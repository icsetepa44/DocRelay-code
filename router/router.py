
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import MultimodalDocumentRouter
from calibration import apply_calibration, fit_calibration_on_partition

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models/dino_best.joblib"
DEFAULT_ENCODER = "camenduru/dinov3-vitl16-pretrain-lvd1689m"


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def positions(n: int) -> torch.Tensor:
    """
    Computes 4 normalized geometric positional signals for a document of n pages:
    [i / (n - 1), (n - 1 - i) / (n - 1), log(1 + i), log(n)]
    """
    if n < 1:
        raise ValueError("Document must have at least 1 page")
    if n == 1:
        return torch.zeros(1, 4, dtype=torch.float32)
    p = torch.zeros(n, 4, dtype=torch.float32)
    for i in range(n):
        p[i, 0] = float(i) / float(n - 1)
        p[i, 1] = float(n - 1 - i) / float(n - 1)
        p[i, 2] = math.log(1.0 + float(i))
        p[i, 3] = math.log(float(n))
    return p


def collate(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pads page features across documents in a batch to max_pages."""
    lengths = [len(x["pages"]) for x in items]
    max_len = max(lengths)
    batch_size = len(items)
    feat_dim = items[0]["pages"].shape[-1]
    
    padded_pages = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
    padded_positions = torch.zeros(batch_size, max_len, 4, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    targets = []
    
    for idx, item in enumerate(items):
        cur_len = lengths[idx]
        padded_pages[idx, :cur_len] = item["pages"]
        padded_positions[idx, :cur_len] = item["positions"]
        mask[idx, :cur_len] = True
        if "target" in item:
            targets.append(item["target"])
            
    res = {
        "pages": padded_pages,
        "positions": padded_positions,
        "mask": mask
    }
    if targets:
        res["targets"] = torch.stack(targets)
    return res


def extract_pdf_pages_dino(
    pdf_path: Union[str, Path],
    model: torch.nn.Module,
    processor,
    device: str = "cpu",
    batch_size: int = 4
) -> torch.Tensor:
    """
    Renders PDF pages at 200 DPI, resizes to 448x448, and extracts DINOv3 CLS embeddings.
    """
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(doc)
    if n_pages < 1:
        doc.close()
        raise ValueError(f"PDF has no pages: {pdf_path}")
        
    cls_list = []
    try:
        for b_start in range(0, n_pages, batch_size):
            images = []
            for i in range(b_start, min(b_start + batch_size, n_pages)):
                page = doc[i]
                bitmap = page.render(scale=200.0 / 72.0)
                images.append(bitmap.to_pil().convert("RGB"))
                bitmap.close()
                page.close()
                
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                cls_tokens = out.last_hidden_state[:, 0, :].cpu().clone()
                cls_list.append(cls_tokens)
    finally:
        doc.close()
        
    return torch.cat(cls_list, dim=0)


class DocumentRouter:
    """
    High-level API for Document Routing.
    
    Usage:
        router = DocumentRouter()
        result = router.route("path/to/document.pdf", mode="balanced")
        print(result["label"])  # "HARD" or "SOFT"
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        encoder_source: str = DEFAULT_ENCODER,
        device: Optional[str] = None,
        offline: bool = False
    ):
        self.device = device or get_default_device()
        self.encoder_source = encoder_source
        self.offline = offline
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Router checkpoint not found: {self.checkpoint_path}")
            
        self.bundle = joblib.load(self.checkpoint_path)
        self.thresholds = self.bundle.get("thresholds", {"balanced": 0.4294, "protect_quality": 0.2394})
        self.calibrator = self.bundle.get("calibrator")
        
        # Initialize Router Neural Net
        self.model = MultimodalDocumentRouter(
            visual_dim=1024,
            use_visual=True,
            use_text=False,
            use_pdf=False,
            pooling_mode="hybrid",
            dropout=0.25,
            is_regression=False
        )
        self.model.load_state_dict(self.bundle["model_state_dict"])
        self.model.eval().to(self.device)
        
        # Lazy-loaded visual encoder
        self._encoder_model = None
        self._encoder_processor = None

    def _load_encoder(self):
        if self._encoder_model is None:
            from transformers import AutoImageProcessor, AutoModel
            kwargs = {"local_files_only": self.offline}
            if not Path(self.encoder_source).is_dir():
                kwargs["revision"] = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
            self._encoder_processor = AutoImageProcessor.from_pretrained(
                self.encoder_source, size={"height": 448, "width": 448}, **kwargs
            )
            self._encoder_model = AutoModel.from_pretrained(
                self.encoder_source, **kwargs
            ).to(self.device).eval()

    def extract_features(self, pdf_path: Union[str, Path], batch_size: int = 4) -> torch.Tensor:
        """Extracts DINOv3 448px CLS page embeddings [N, 1024] from a PDF."""
        self._load_encoder()
        return extract_pdf_pages_dino(
            pdf_path,
            model=self._encoder_model,
            processor=self._encoder_processor,
            device=self.device,
            batch_size=batch_size
        )

    def route_features(
        self,
        page_features: torch.Tensor,
        mode: str = "balanced"
    ) -> Dict[str, Union[str, float, int]]:
        """
        Routes precomputed page features [N, 1024].
        """
        if mode not in self.thresholds:
            raise ValueError(f"Unknown mode: {mode}. Choose from {list(self.thresholds.keys())}")
            
        n_pages = page_features.shape[0]
        pos = positions(n_pages)
        batch = collate([{"pages": page_features, "positions": pos}])
        
        with torch.no_grad():
            out, _, _ = self.model(
                pages=batch["pages"].to(self.device),
                positions=batch["positions"].to(self.device),
                mask=batch["mask"].to(self.device)
            )
            raw_logit = out.squeeze(-1).cpu().item()
            raw_score = 1.0 / (1.0 + math.exp(-raw_logit))
            
        if self.calibrator is not None:
            prob = float(apply_calibration(self.calibrator, np.array([raw_score]))[0])
        else:
            prob = raw_score
            
        threshold = float(self.thresholds[mode])
        is_hard = (prob >= threshold)
        
        return {
            "label": "HARD" if is_hard else "SOFT",
            "hard_probability": round(prob, 4),
            "threshold": round(threshold, 4),
            "mode": mode,
            "page_count": n_pages,
            "action": "escalate_to_vlm_or_ocr" if is_hard else "standard_fast_extraction"
        }

    def route(
        self,
        pdf_path: Union[str, Path],
        mode: str = "balanced",
        batch_size: int = 4
    ) -> Dict[str, Union[str, float, int]]:
        """
        Routes a single PDF file.
        """
        p = Path(pdf_path)
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")
        features = self.extract_features(p, batch_size=batch_size)
        res = self.route_features(features, mode=mode)
        res["pdf"] = str(p.resolve())
        return res

    def route_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        mode: str = "balanced",
        batch_size: int = 4
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Routes a list of PDF files sequentially.
        """
        results = []
        for p in pdf_paths:
            try:
                results.append(self.route(p, mode=mode, batch_size=batch_size))
            except Exception as e:
                results.append({
                    "pdf": str(p),
                    "error": str(e),
                    "label": "ERROR"
                })
        return results


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 Visual Document Router: Fast, intelligent PDF difficulty classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. predict command (single PDF)
    p_pred = subparsers.add_parser("predict", help="Predict difficulty for a single PDF.")
    p_pred.add_argument("--pdf", type=Path, required=True, help="Path to input PDF file.")
    p_pred.add_argument("--mode", choices=["balanced", "protect_quality"], default="balanced",
                        help="Routing policy.")
    p_pred.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to trained router checkpoint.")
    p_pred.add_argument("--encoder", default=DEFAULT_ENCODER, help="DINOv3 HF model ID or local directory.")
    p_pred.add_argument("--device", default=get_default_device(), help="Compute device (cpu, cuda, mps).")
    p_pred.add_argument("--offline", action="store_true", help="Only use local cached weights.")

    # 2. batch command (directory of PDFs)
    p_batch = subparsers.add_parser("batch", help="Route all PDFs in a directory.")
    p_batch.add_argument("--dir", type=Path, required=True, help="Directory containing PDFs.")
    p_batch.add_argument("--output", type=Path, default=Path("routing_results.csv"),
                         help="Output CSV path.")
    p_batch.add_argument("--mode", choices=["balanced", "protect_quality"], default="balanced",
                         help="Routing policy.")
    p_batch.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                         help="Path to trained router checkpoint.")
    p_batch.add_argument("--device", default=get_default_device(), help="Compute device.")
    p_batch.add_argument("--offline", action="store_true", help="Only use local cached weights.")

    args = parser.parse_args()

    router = DocumentRouter(
        checkpoint_path=args.checkpoint,
        encoder_source=getattr(args, "encoder", DEFAULT_ENCODER),
        device=args.device,
        offline=args.offline
    )

    if args.command == "predict":
        res = router.route(args.pdf, mode=args.mode)
        print(json.dumps(res, indent=2))

    elif args.command == "batch":
        pdfs = sorted(list(args.dir.rglob("*.pdf")))
        print(f"Found {len(pdfs)} PDF documents in {args.dir}. Processing...")
        records = []
        for idx, pdf in enumerate(pdfs, 1):
            res = router.route(pdf, mode=args.mode)
            records.append(res)
            status = f"[{idx}/{len(pdfs)}] {pdf.name}: {res.get('label')} (p={res.get('hard_probability', 'N/A')})"
            print(status)
            
        df = pd.DataFrame(records)
        df.to_csv(args.output, index=False)
        print(f"\nSaved batch predictions to: {args.output}")


if __name__ == "__main__":
    main()
