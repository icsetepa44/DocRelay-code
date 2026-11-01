
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from model import MultimodalDocumentRouter
from calibration import apply_calibration, fit_calibration_on_partition
import router


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_position_encodings(self):
        # Single page document
        p1 = router.positions(1)
        self.assertEqual(p1.shape, (1, 4))
        self.assertTrue(torch.equal(p1, torch.zeros(1, 4)))

        # Multi-page document
        n = 5
        pn = router.positions(n)
        self.assertEqual(pn.shape, (n, 4))
        # First page relative position = 0, reverse = 1
        self.assertAlmostEqual(float(pn[0, 0]), 0.0)
        self.assertAlmostEqual(float(pn[0, 1]), 1.0)
        # Last page relative position = 1, reverse = 0
        self.assertAlmostEqual(float(pn[-1, 0]), 1.0)
        self.assertAlmostEqual(float(pn[-1, 1]), 0.0)
        # Log n scale constant
        self.assertAlmostEqual(float(pn[0, 3]), math.log(n))

    def test_batch_collate_and_padding_mask(self):
        item1 = {"pages": torch.randn(2, 1024), "positions": router.positions(2), "target": torch.tensor(1.0)}
        item2 = {"pages": torch.randn(4, 1024), "positions": router.positions(4), "target": torch.tensor(0.0)}
        batch = router.collate([item1, item2])
        
        self.assertEqual(batch["pages"].shape, (2, 4, 1024))
        self.assertEqual(batch["positions"].shape, (2, 4, 4))
        self.assertEqual(batch["mask"].shape, (2, 4))
        self.assertEqual(batch["mask"][0].tolist(), [True, True, False, False])
        self.assertEqual(batch["mask"][1].tolist(), [True, True, True, True])
        self.assertEqual(batch["targets"].shape, (2,))

    def test_model_forward_pass(self):
        net = MultimodalDocumentRouter(
            visual_dim=1024,
            use_visual=True,
            use_text=False,
            use_pdf=False,
            pooling_mode="hybrid"
        )
        B, N = 2, 3
        pages = torch.randn(B, N, 1024)
        pos = torch.randn(B, N, 4)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        
        out, attn, entropy = net(pages=pages, positions=pos, mask=mask)
        self.assertEqual(out.shape, (B, 1))
        self.assertEqual(attn.shape, (B, N))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(entropy).all())
        # Masked padding page attention must be 0
        self.assertAlmostEqual(float(attn[0, 2].detach()), 0.0, places=5)
        # Softmax over valid pages sums to 1.0
        self.assertAlmostEqual(float(attn[0].detach().sum()), 1.0, places=4)
        self.assertAlmostEqual(float(attn[1].detach().sum()), 1.0, places=4)

    def test_calibration_platt_scaling(self):
        np.random.seed(42)
        raw_scores = np.array([0.1, 0.2, 0.35, 0.45, 0.55, 0.7, 0.85, 0.95])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        calibrator, thresholds = fit_calibration_on_partition(raw_scores, labels, target_hard_recall=0.85)
        self.assertIn("balanced", thresholds)
        self.assertIn("protect_quality", thresholds)
        self.assertTrue(thresholds["protect_quality"] <= thresholds["balanced"])
        
        cal_probs = apply_calibration(calibrator, raw_scores)
        self.assertEqual(len(cal_probs), len(raw_scores))
        self.assertTrue((cal_probs >= 0.0).all() and (cal_probs <= 1.0).all())
        # Monotonically increasing
        self.assertTrue(np.all(np.diff(cal_probs) >= 0))

    def test_saved_checkpoint_routing(self):
        doc_router = router.DocumentRouter(offline=True)
        self.assertEqual(sum(p.numel() for p in doc_router.model.parameters()), 403105)
        
        # Test routing precomputed page embeddings
        dummy_pages = torch.randn(3, 1024)
        res_bal = doc_router.route_features(dummy_pages, mode="balanced")
        res_pq = doc_router.route_features(dummy_pages, mode="protect_quality")
        
        self.assertIn(res_bal["label"], ["HARD", "SOFT"])
        self.assertIn(res_pq["label"], ["HARD", "SOFT"])
        self.assertEqual(res_bal["page_count"], 3)
        self.assertTrue(0.0 <= res_bal["hard_probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
