
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def fit_calibration_on_partition(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    target_hard_recall: float = 0.85
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Fits Platt scaling calibrator and computes dual operating thresholds.
    
    Args:
        cal_scores: Raw model output probabilities or logits in (0, 1).
        cal_labels: Ground truth binary difficulty labels (1 = HARD, 0 = SOFT).
        target_hard_recall: Target recall for the protect_quality policy.
        
    Returns:
        calibrator: Fitted LogisticRegression model.
        thresholds_dict: Dictionary containing 'balanced' and 'protect_quality' thresholds.
    """
    eps = 1e-6
    clipped = np.clip(cal_scores, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(logits, cal_labels)
    
    cal_probs = calibrator.predict_proba(logits)[:, 1]
    
    # 1. Balanced threshold (maximizes balanced accuracy)
    candidate_thresholds = np.unique(np.concatenate([[0.0], cal_probs, [1.0]]))
    best_bacc = -1.0
    t_balanced = 0.5
    for t in candidate_thresholds:
        pred = (cal_probs >= t).astype(int)
        bacc = balanced_accuracy_score(cal_labels, pred)
        if bacc > best_bacc or (bacc == best_bacc and abs(t - 0.5) < abs(t_balanced - 0.5)):
            best_bacc = bacc
            t_balanced = float(t)
            
    # 2. Quality protection threshold (targeting >= target_hard_recall)
    hard_probs = cal_probs[cal_labels == 1]
    if len(hard_probs) > 0:
        quantile = max(0.01, 1.0 - target_hard_recall - 0.03)
        t_protect = float(np.quantile(hard_probs, quantile))
        t_protect = min(float(t_balanced), max(0.02, t_protect))
    else:
        t_protect = min(float(t_balanced), 0.10)
        
    thresholds_dict = {
        "balanced": float(t_balanced),
        "protect_quality": float(t_protect),
        "target_hard_recall": float(target_hard_recall)
    }
    return calibrator, thresholds_dict


def apply_calibration(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    """
    Applies fitted Platt scaling calibrator to raw model scores.
    """
    eps = 1e-6
    clipped = np.clip(scores, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]
