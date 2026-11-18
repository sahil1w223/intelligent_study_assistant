"""Command-line entry point for the I/O scheduling simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .models import IORequest
from .simulation import simulate
from .traces import load_trace, synthetic_burst


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate FIFO or ML-inspired SJF I/O scheduling")
    parser.add_argument("--trace", help="CSV trace path; absent means use a deterministic synthetic burst")
    parser.add_argument("--policy", choices=("fifo", "sjf", "both"), default="both")
    parser.add_argument("--count", type=int, default=100, help="synthetic request count (default: 100)")
    parser.add_argument("--aging-ms", type=float, default=0.01, help="per-request SJF aging penalty")
    parser.add_argument("--immediate-threshold", type=int, default=2,
                        help="use FIFO below this pending depth (default: 2)")
    parser.add_argument("--details", action="store_true", help="include each request's dispatch and finish times")
    args = parser.parse_args()
    requests: list[IORequest] = load_trace(args.trace) if args.trace else synthetic_burst(args.count)
    policies = ["fifo", "sjf"] if args.policy == "both" else [args.policy]
    output = []
    for policy in policies:
        results, metrics = simulate(requests, policy=policy, aging_ms=args.aging_ms,
                                    immediate_threshold=args.immediate_threshold)
        entry = {"metrics": asdict(metrics)}
        if args.details:
            entry["requests"] = [
                {"request_id": item.request.request_id, "start_ms": item.start_ms,
                 "completion_ms": item.completion_ms, "response_ms": item.response_ms}
                for item in results
            ]
        output.append(entry)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
