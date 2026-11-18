"""Core data models shared by feature extraction and scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IORequest:
    """One disk request. Times and service durations are expressed in milliseconds."""

    request_id: str
    arrival_ms: float
    operation: str
    tag: str
    size_bytes: int
    lba: int
    service_ms: float
    workload: str = "unknown"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.arrival_ms < 0 or self.service_ms < 0:
            raise ValueError("arrival_ms and service_ms must be non-negative")
        if self.operation.lower() not in {"read", "write"}:
            raise ValueError("operation must be 'read' or 'write'")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if self.lba < 0:
            raise ValueError("lba must be non-negative")
        object.__setattr__(self, "operation", self.operation.lower())
        object.__setattr__(self, "tag", self.tag.upper())


@dataclass(frozen=True, slots=True)
class RequestResult:
    request: IORequest
    start_ms: float
    completion_ms: float

    @property
    def response_ms(self) -> float:
        return self.completion_ms - self.request.arrival_ms

    @property
    def waiting_ms(self) -> float:
        return self.start_ms - self.request.arrival_ms


@dataclass(frozen=True, slots=True)
class SimulationMetrics:
    policy: str
    request_count: int
    makespan_ms: float
    average_response_ms: float
    average_waiting_ms: float
    throughput_ops_sec: float

    @classmethod
    def from_results(cls, policy: str, results: list[RequestResult]) -> "SimulationMetrics":
        if not results:
            return cls(policy, 0, 0.0, 0.0, 0.0, 0.0)
        first_arrival = min(item.request.arrival_ms for item in results)
        last_completion = max(item.completion_ms for item in results)
        makespan = max(0.0, last_completion - first_arrival)
        return cls(
            policy=policy,
            request_count=len(results),
            makespan_ms=makespan,
            average_response_ms=sum(item.response_ms for item in results) / len(results),
            average_waiting_ms=sum(item.waiting_ms for item in results) / len(results),
            throughput_ops_sec=(len(results) * 1000.0 / makespan) if makespan > 0 else 0.0,
        )
