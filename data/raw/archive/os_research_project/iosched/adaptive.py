"""Expected-throughput model for adaptive scheduler switching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdaptiveEstimate:
    best_scheduler_by_predicted_workload: dict[str, str]
    throughput_by_true_workload: dict[str, float]
    overall_throughput: float


def estimate_adaptive_throughput(
    confusion: Mapping[str, Mapping[str, float]],
    throughput: Mapping[str, Mapping[str, float]],
    workloads: Sequence[str] | None = None,
) -> AdaptiveEstimate:
    """Calculate the paper's expected throughput from measured input tables.

    ``confusion[predicted][true]`` holds P(predicted | true), expressed either
    as probabilities or percentages. ``throughput[scheduler][workload]`` holds
    measured operations/second. For each prediction, the model selects the
    scheduler with the best observed throughput for that predicted workload.
    It then averages realized throughput over prediction outcomes for each true
    workload, matching the equation in Section 5.
    """
    if not confusion or not throughput:
        raise ValueError("confusion and throughput tables must not be empty")
    predicted_labels = list(confusion)
    inferred_workloads = list(workloads or confusion[predicted_labels[0]])
    if not inferred_workloads:
        raise ValueError("at least one workload is required")
    for predicted in predicted_labels:
        for true in inferred_workloads:
            if true not in confusion[predicted]:
                raise ValueError(f"confusion table missing P({predicted}|{true})")
            value = float(confusion[predicted][true])
            if value < 0:
                raise ValueError("confusion probabilities cannot be negative")
    probabilities: dict[tuple[str, str], float] = {}
    for true in inferred_workloads:
        total = sum(float(confusion[predicted][true]) for predicted in predicted_labels)
        if total <= 0:
            raise ValueError(f"confusion probabilities for true workload {true!r} sum to zero")
        scale = 100.0 if total > 1.000001 else 1.0
        if abs(total / scale - 1.0) > 0.02:
            raise ValueError(f"confusion probabilities for {true!r} must sum to 1 or 100")
        for predicted in predicted_labels:
            probabilities[predicted, true] = float(confusion[predicted][true]) / total

    best: dict[str, str] = {}
    for predicted in predicted_labels:
        candidates = [(float(results[predicted]), scheduler)
                      for scheduler, results in throughput.items() if predicted in results]
        if not candidates:
            raise ValueError(f"no measured scheduler throughput for predicted workload {predicted!r}")
        if any(value < 0 for value, _ in candidates):
            raise ValueError("throughput values cannot be negative")
        best[predicted] = max(candidates, key=lambda item: (item[0], item[1]))[1]

    by_true: dict[str, float] = {}
    for true in inferred_workloads:
        realized = 0.0
        for predicted in predicted_labels:
            scheduler = best[predicted]
            if true not in throughput[scheduler]:
                raise ValueError(f"scheduler {scheduler!r} lacks throughput for {true!r}")
            realized += probabilities[predicted, true] * float(throughput[scheduler][true])
        by_true[true] = realized
    overall = sum(by_true.values()) / len(by_true)
    return AdaptiveEstimate(best, by_true, overall)
