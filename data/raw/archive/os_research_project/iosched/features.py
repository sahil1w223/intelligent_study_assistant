"""Order-independent feature encoding used by the latency estimator."""

from __future__ import annotations

from collections.abc import Sequence
import math

from .models import IORequest

# Values are numeric and intentionally fixed so saved models/features remain reproducible.
_TAGS = (
    "META", "SUPER", "GROUP", "DIRECT", "INDIRECT", "EXTENT", "DATA_DIR",
    "BLK_BMP", "INO_BMP", "INODE", "XATTR", "JOURNAL", "FILE_BULK",
    "FILE_4KB", "FILE_16KB", "FILE_64KB", "FILE_256KB", "FILE_1MB",
    "FILE_4MB", "FILE_16MB", "FILE_64MB", "FILE_256MB", "FILE_1GB",
)


def encode_request(request: IORequest, completed: Sequence[IORequest], history_size: int = 16) -> list[float]:
    """Encode operation/size/tag plus N recent completed-request context features.

    The paper uses one-hot request metadata and up to N recent completed requests;
    this compact deterministic encoding preserves those inputs for this simulator.
    It deliberately excludes queue depth and order-dependent LBA deltas.
    """
    if history_size < 0:
        raise ValueError("history_size must be non-negative")
    recent = list(completed[-history_size:]) if history_size else []
    current = _base_features(request)
    context: list[float] = []
    for previous in recent:
        context.extend(_base_features(previous))
    # Fixed width lets the estimator consume a uniform feature vector.
    width = 2 + 1 + len(_TAGS)
    context = context[-(history_size * width):] if history_size else []
    context = [0.0] * (history_size * width - len(context)) + context
    return current + context


def encode_workload_window(requests: Sequence[IORequest], history_size: int = 16) -> list[float]:
    """Encode the N preceding requests for workload classification.

    Every request contributes operation, filesystem tag, size, log latency, and
    signed log-scaled LBA delta. Earlier missing requests are zero-padded so the
    classifier always receives a fixed-width window.
    """
    if history_size < 1:
        raise ValueError("history_size must be positive")
    recent = list(requests[-history_size:])
    vectors: list[list[float]] = []
    previous_lba: int | None = None
    for request in recent:
        tag = request.tag.upper()
        tag_vector = [1.0 if tag == candidate else 0.0 for candidate in _TAGS]
        delta = 0 if previous_lba is None else request.lba - previous_lba
        vectors.append([
            1.0 if request.operation == "read" else 0.0,
            1.0 if request.operation == "write" else 0.0,
            min(1.0, max(0.0, request.size_bytes.bit_length() / 40.0)),
            *tag_vector,
            math.log(max(request.service_ms, 0.001)),
            math.copysign(math.log1p(abs(delta)), delta) / 32.0,
        ])
        previous_lba = request.lba
    width = 2 + 1 + len(_TAGS) + 2
    padding = [[0.0] * width for _ in range(history_size - len(vectors))]
    return [value for vector in padding + vectors for value in vector]


def _base_features(request: IORequest) -> list[float]:
    tag = request.tag.upper()
    tag_vector = [1.0 if tag == candidate else 0.0 for candidate in _TAGS]
    # Log2 scaling avoids large feature magnitudes while retaining size ordering.
    return [
        1.0 if request.operation == "read" else 0.0,
        1.0 if request.operation == "write" else 0.0,
        min(1.0, max(0.0, request.size_bytes.bit_length() / 40.0)),
        *tag_vector,
    ]
