from typing import cast

from pytest import approx
from transformers.trainer_utils import EvalPrediction

from icft.datasets.multinerd import Multinerd


def test_compute_multinerd_metrics():
    labels = [0, 0, 1, 1, 2, 2]
    logits = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
    ]

    eval_pred = cast(EvalPrediction, (logits, labels))
    eval_metrics = Multinerd.compute_metrics(eval_pred)

    assert eval_metrics["accuracy"] == approx(0.5)
    assert eval_metrics["precision"] == approx(0.5)
    assert eval_metrics["recall"] == approx(0.5)
    assert eval_metrics["f1"] == approx(0.4)
