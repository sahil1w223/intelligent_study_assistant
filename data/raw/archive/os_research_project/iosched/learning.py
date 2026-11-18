"""Training helpers for the paper's latency and workload neural networks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from .features import encode_request, encode_workload_window
from .mlp import MLP
from .models import IORequest


class LatencyPredictor:
    """Predict request service time from unordered request metadata and completed history."""

    def __init__(self, model: MLP, history_size: int = 16, fallback_ms: float = 1.0) -> None:
        if model.mode != "regression" or model.output_size != 1:
            raise ValueError("latency predictor requires a one-output regression MLP")
        if history_size < 0 or fallback_ms <= 0:
            raise ValueError("history_size must be non-negative and fallback_ms positive")
        self.model, self.history_size, self.fallback_ms = model, history_size, fallback_ms

    def predict_ms(self, request: IORequest, completed: Sequence[IORequest]) -> float:
        if not completed:
            return self.fallback_ms
        raw = self.model.predict(encode_request(request, completed, self.history_size))[0]
        baseline = statistics.median(item.service_ms for item in completed[-self.history_size:]) if self.history_size else self.fallback_ms
        prediction = baseline * math.exp(max(-20.0, min(20.0, raw)))
        return max(0.001, prediction) if math.isfinite(prediction) else self.fallback_ms


def train_latency_model(requests: Sequence[IORequest], history_size: int = 16,
                        hidden_sizes: list[int] | None = None, epochs: int = 250,
                        learning_rate: float = 0.01) -> MLP:
    """Train on trace labels, predicting log(service / recent-history median).

    Trace order is treated as completion-history order for supervised training.
    This is a simulation approximation: the paper trained using kernel-collected
    completion events and withheld order-dependent features.
    """
    if len(requests) < 2:
        raise ValueError("at least two trace requests are required to train a latency model")
    if history_size < 0:
        raise ValueError("history_size must be non-negative")
    width = _feature_width(history_size)
    features: list[list[float]] = []
    targets: list[list[float]] = []
    history: list[IORequest] = []
    for request in requests:
        if history:
            baseline = statistics.median(item.service_ms for item in history[-history_size:]) if history_size else 1.0
            baseline = max(0.001, baseline)
            features.append(encode_request(request, history, history_size))
            targets.append([math.log(max(0.001, request.service_ms) / baseline)])
        history.append(request)
    model = MLP(width, hidden_sizes or [32, 16, 8, 4], 1, "regression")
    model.fit(features, targets, epochs=epochs, learning_rate=learning_rate)
    return model


def train_classifier(requests: Sequence[IORequest], history_size: int = 16,
                     hidden_sizes: list[int] | None = None, epochs: int = 250) -> tuple[MLP, list[str]]:
    """Train a workload classifier; history order follows the supplied trace."""
    labels = sorted({request.workload for request in requests})
    if len(labels) < 2:
        raise ValueError("classification training requires at least two workload labels")
    if len(requests) < 2:
        raise ValueError("at least two trace requests are required")
    if history_size < 1:
        raise ValueError("history_size must be positive")
    features = [encode_workload_window(requests[max(0, i-history_size):i], history_size)
                for i in range(len(requests))]
    targets = [[1.0 if request.workload == label else 0.0 for label in labels] for request in requests]
    model = MLP(_workload_feature_width(history_size), hidden_sizes or [128, 64, 32, 16], len(labels), "classification")
    model.fit(features, targets, epochs=epochs)
    return model, labels


def _workload_feature_width(history_size: int) -> int:
    # Workload classifier: operation + size + 23 tags + latency + LBA delta per prior request.
    return 28 * history_size


def _feature_width(history_size: int) -> int:
    # Two operation bits + size + 23 tag slots; see features._TAGS.
    return 26 * (history_size + 1)
