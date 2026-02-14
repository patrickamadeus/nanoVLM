from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import compute_batch_effective_token_metrics, compute_effective_token_ratio


def test_compute_batch_effective_token_metrics_uses_non_pad_tokens():
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
        ],
        dtype=torch.int64,
    )
    effective_tokens, token_capacity, effective_ratio = compute_batch_effective_token_metrics(attention_mask)

    assert effective_tokens == 5
    assert token_capacity == 8
    assert effective_ratio == 5 / 8


def test_compute_effective_token_ratio_requires_positive_capacity():
    with pytest.raises(ValueError, match="Token capacity must be > 0"):
        compute_effective_token_ratio(10, 0)
