from __future__ import annotations

import math


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimator used by code-generation benchmarks."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be between 0 and num_samples")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be between 1 and num_samples")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)
