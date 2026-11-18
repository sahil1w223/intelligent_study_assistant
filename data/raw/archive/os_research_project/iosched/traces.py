"""CSV trace input and small synthetic workloads for experiments."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import TextIO

from .models import IORequest

_REQUIRED = {"request_id", "arrival_ms", "operation", "tag", "size_bytes", "lba", "service_ms"}


def load_trace(path: str | Path) -> list[IORequest]:
    """Load requests from a CSV file with the documented trace columns."""
    try:
        with Path(path).open("r", newline="", encoding="utf-8") as stream:
            return read_trace(stream)
    except OSError as exc:
        raise ValueError(f"cannot read trace {path}: {exc}") from exc


def read_trace(stream: TextIO) -> list[IORequest]:
    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise ValueError("trace CSV is empty or missing a header")
    missing = _REQUIRED - set(reader.fieldnames)
    if missing:
        raise ValueError(f"trace CSV missing columns: {', '.join(sorted(missing))}")
    requests: list[IORequest] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            requests.append(IORequest(
                request_id=row["request_id"],
                arrival_ms=float(row["arrival_ms"]),
                operation=row["operation"],
                tag=row["tag"],
                size_bytes=int(row["size_bytes"]),
                lba=int(row["lba"]),
                service_ms=float(row["service_ms"]),
                workload=row.get("workload") or "unknown",
            ))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid trace row {line_number}: {exc}") from exc
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("trace contains duplicate request_id values")
    return requests


def synthetic_burst(count: int = 100, seed: int = 17) -> list[IORequest]:
    """Create a reproducible burst with a small fast-I/O class and longer data I/O."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    requests = []
    for index in range(count):
        fast = index % 4 == 0
        requests.append(IORequest(
            request_id=f"req-{index:04d}", arrival_ms=0.0,
            operation="read" if rng.random() < 0.8 else "write",
            tag="META" if fast else "DATA_DIR", size_bytes=4096 if fast else 65536,
            lba=rng.randrange(0, 1_000_000), service_ms=rng.uniform(0.05, 0.15) if fast else rng.uniform(2, 8),
            workload="fast" if fast else "data",
        ))
    return requests
