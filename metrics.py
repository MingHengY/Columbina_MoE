"""Shared evaluation metrics for Columbina 4.0."""

import numpy as np


def precision_at_k(y_true, y_score, k=10):
    """Return the positive fraction among the highest-scoring k examples."""
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score).reshape(-1)
    if labels.size != scores.size:
        raise ValueError(
            f"y_true and y_score lengths differ: {labels.size} vs {scores.size}"
        )
    if labels.size == 0 or k <= 0:
        return 0.0
    top_count = min(int(k), labels.size)
    top_indices = np.argsort(-scores, kind='mergesort')[:top_count]
    return float((labels[top_indices] > 0.5).mean())
