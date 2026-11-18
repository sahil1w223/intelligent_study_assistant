"""Discrete-event single-device simulator for FIFO and ML-SJF scheduling."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence

from .learning import LatencyPredictor
from .models import IORequest, RequestResult, SimulationMetrics
from .scheduler import choose_fifo, choose_sjf

PredictorFunction = Callable[[IORequest, Sequence[IORequest]], float]


def simulate(requests: Sequence[IORequest], policy: str = "sjf",
             predictor: PredictorFunction | LatencyPredictor | None = None,
             aging_ms: float = 0.01, immediate_threshold: int = 1) -> tuple[list[RequestResult], SimulationMetrics]:
    """Simulate a serialized block device and return results plus summary metrics.

    If fewer than ``immediate_threshold`` requests wait, dispatch is FIFO without
    model inference (the paper's sparse-queue fast path). With a congested queue,
    choose by predicted latency. ``IORequest.service_ms`` is the measured ground-
    truth service time; the predictor never receives it for the request being scored.
    """
    if policy not in {"fifo", "sjf"}:
        raise ValueError("policy must be 'fifo' or 'sjf'")
    if immediate_threshold < 1:
        raise ValueError("immediate_threshold must be at least 1")
    if aging_ms < 0:
        raise ValueError("aging_ms must be non-negative")
    ordered = sorted(enumerate(requests), key=lambda item: (item[1].arrival_ms, item[0]))
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request_id values must be unique")
    if not requests:
        results: list[RequestResult] = []
        return results, SimulationMetrics.from_results(policy, results)
    if predictor is None:
        predict: PredictorFunction = lambda request, completed: request.service_ms
    elif isinstance(predictor, LatencyPredictor):
        predict = predictor.predict_ms
    else:
        predict = predictor

    pending: list[IORequest] = []
    completed_requests: list[IORequest] = []
    results: list[RequestResult] = []
    arrivals = iter(ordered)
    next_arrival = next(arrivals, None)
    now = 0.0
    busy_until = 0.0

    while next_arrival is not None or pending or now < busy_until:
        if not pending and now < busy_until:
            now = busy_until
        while next_arrival is not None and next_arrival[1].arrival_ms <= now:
            pending.append(next_arrival[1])
            next_arrival = next(arrivals, None)
        if now < busy_until:
            if next_arrival is None:
                now = busy_until
                continue
            now = min(busy_until, next_arrival[1].arrival_ms)
            continue
        if not pending:
            if next_arrival is None:
                break
            now = max(now, next_arrival[1].arrival_ms)
            continue

        # Sparse arrivals get the cheap FIFO route. In a burst, use SJF.
        if policy == "fifo" or len(pending) < immediate_threshold:
            request = choose_fifo(pending, completed_requests)
        else:
            request = choose_sjf(pending, completed_requests, predict, aging_ms)
        pending.remove(request)
        start = max(now, request.arrival_ms)
        completion = start + request.service_ms
        results.append(RequestResult(request, start, completion))
        completed_requests.append(request)
        now, busy_until = start, completion

    results.sort(key=lambda item: (item.completion_ms, item.request.request_id))
    return results, SimulationMetrics.from_results(policy, results)
