"""Request selection policies inspired by the paper's two-queue scheduler."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .models import IORequest

Predictor = Callable[[IORequest, Sequence[IORequest]], float]


def choose_fifo(queue: Sequence[IORequest], completed: Sequence[IORequest]) -> IORequest:
    """Return the earliest-arriving queued request, breaking ties by stable input order."""
    if not queue:
        raise ValueError("cannot select from an empty queue")
    return min(enumerate(queue), key=lambda pair: (pair[1].arrival_ms, pair[0]))[1]


def choose_sjf(queue: Sequence[IORequest], completed: Sequence[IORequest],
               predictor: Predictor, aging_ms: float = 0.01) -> IORequest:
    """Select the lowest predicted latency plus a small queue-age penalty.

    The penalty is δ times the request's position in arrival order. It gives
    older queued requests a strictly better tie-breaking priority and is intended
    to be many orders smaller than service latency, as described in the paper.
    """
    if not queue:
        raise ValueError("cannot select from an empty queue")
    if aging_ms < 0:
        raise ValueError("aging_ms must be non-negative")
    ordered = sorted(enumerate(queue), key=lambda pair: (pair[1].arrival_ms, pair[0]))
    priorities = []
    for rank, (_, request) in enumerate(ordered):
        predicted = predictor(request, completed)
        if predicted < 0:
            raise ValueError("latency predictor returned a negative estimate")
        priorities.append((predicted + rank * aging_ms, rank, request))
    return min(priorities, key=lambda item: (item[0], item[1]))[2]
