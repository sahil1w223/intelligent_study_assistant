# Machine-Learning I/O Scheduling Simulator

This project is a runnable, user-space simulator inspired by **“Towards Machine Learning-Based I/O Scheduling” (Technical Report FSL-25-07)**. It demonstrates the key algorithmic ideas without patching or replacing the Linux kernel scheduler.

## Paper in simple language

A disk request scheduler chooses which pending read or write a storage device handles next. Traditional schedulers rely on fixed rules, which can perform differently for different workloads.

The paper evaluates two ML-driven ideas:

1. **Adaptive scheduler switching:** observe recent requests to classify a workload, then select the scheduler that historically gives that workload the best throughput. Its expected throughput is computed from the workload confusion matrix and measured scheduler/workload throughput table.
2. **KML-IOSched (ML shortest-job-first):** predict each queued request’s service latency, and dispatch the request expected to finish quickest. To avoid unnecessary model cost when the queue is sparse, a FIFO fast path dispatches immediately. A small increasing priority penalty for later-arriving queue entries helps prevent starvation.

The paper’s workload classifier uses the operation, filesystem tag, request size, measured latency and LBA delta from the preceding **N = 16** requests, with a neural net configured with **L = 128, M = 4**. The latency estimator instead uses current-request operation, size and tag plus recently completed requests, deliberately excluding queue depth and other request-order-dependent inputs. It log-transforms and normalizes latency; its selected architecture uses **L = 32, M = 4, N = 16**. The code provides separate encoders for these classifier and predictor feature sets. Reported findings include 86% merged-class workload classification accuracy, PCC 0.90 for latency predictions, 7.5% mean throughput gain for adaptive switching, and a 43.9% average latency reduction in a deliberately idealized burst experiment. These are the paper’s reported results, **not results reproduced by this project**.

### Problem statement

Fixed I/O scheduling heuristics cannot adapt well to varying workloads and may fail to reduce latency or maximize throughput. For an SJF policy, useful service-time estimates must not depend on the dispatch order the scheduler itself changes.

### Proposed solution

Use learned recent-request patterns for scheduler selection, and use an ML predictor based on order-independent request features to rank pending requests by estimated service time. Use FIFO when sparse and aging to reduce starvation under SJF.

## Scope and fidelity

The implementation is a portable **algorithm-level simulation**, not the patched Intel-OST Linux 4.13.2 kernel, KML inference runtime, Ext4 tagging changes, actual device driver, or a reproduction of the paper’s experiments. The original kernel work needs its specific legacy environment and hardware. Here, Python contains a dependency-free MLP, trace loader, FIFO/SJF selection, and single-device discrete-event simulator. It supports the core SJF and prediction mechanisms; a training helper is included for trace-driven experiments.

Adaptive expected-throughput calculation is implemented in `iosched.adaptive.estimate_adaptive_throughput`; it needs measured scheduler/workload throughput and a confusion matrix. The simulator does not fabricate those missing benchmark artifacts or claim to reproduce the paper's adaptive result.

## Project structure

```text
.
├── iosched/
│   ├── __init__.py       # Public package API
│   ├── cli.py            # Command line interface
│   ├── features.py       # Request and completed-history encoding
│   ├── learning.py       # Latency and classifier model training wrappers
│   ├── mlp.py            # Pure Python leaky-ReLU MLP
│   ├── models.py         # Validated requests and metrics
│   ├── scheduler.py      # FIFO and aging-aware ML-SJF selection
│   ├── simulation.py    # Single-device discrete-event simulation
│   └── traces.py         # CSV traces and synthetic sample workload
├── tests/
│   ├── test_ml.py
│   ├── test_scheduler.py
│   └── test_traces.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Run on Ubuntu Linux

Python 3.10 or newer is required. No third-party runtime packages are needed.

```bash
sudo apt update
sudo apt install -y python3 python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run a generated 100-request burst through both policies:

```bash
python -m iosched.cli --policy both --count 100
```

Or use the installed command:

```bash
iosched --policy sjf --count 100 --details
```

For a real trace, create a CSV with the following required columns. `workload` is optional:

```csv
request_id,arrival_ms,operation,tag,size_bytes,lba,service_ms,workload
r1,0,read,DATA_DIR,4096,120,2.4,readseq
r2,0.1,write,META,4096,55,0.3,readseq
r3,0.2,read,INODE,8192,400,1.1,readseq
```

Then run:

```bash
iosched --trace trace.csv --policy both --details
```

`service_ms` is the known/observed service time used by the simulator to advance the device clock and evaluate resulting scheduling. The built-in fallback predictor uses that value as an **oracle**, so it is for illustrating the policy’s upper-bound behavior, not prediction quality. For a learned predictor, use `train_latency_model` from `iosched.learning` on a labeled trace, then pass a `LatencyPredictor` into `simulate`. The API is intentionally exposed rather than silently training on the evaluation requests.

The SJF policy enters its prediction path when at least `--immediate-threshold` requests are pending (default 2); otherwise it uses FIFO. `--aging-ms` sets a small deterministic penalty for queue age. Response time includes waiting plus service time.

## Tests

Run from the project root:

```bash
python3 -m unittest discover -s tests -v
```

## Important file explanations

- `models.py`: the fields describing one I/O request, validation, and summary metrics.
- `features.py`: encodes the current operation, size and filesystem tag, plus the most recent completed-request context; it excludes queue depth and LBA delta for the SJF predictor.
- `mlp.py`: a small fully connected neural network with leaky-ReLU hidden layers; classifier output uses softmax. It can save and reload JSON weights.
- `learning.py`: trains a latency regressor against log-normalized service times and a workload classifier. Training traces should be separate from the evaluation trace for meaningful evaluation.
- `scheduler.py`: orders requests with FIFO or predicted service time plus the age penalty.
- `adaptive.py`: computes expected adaptive-scheduler throughput from a supplied confusion matrix and measured throughput table.
- `simulation.py`: simulates a single serialized device, dynamic arrivals and pending requests.
- `traces.py`: validates CSV input and provides deterministic generated requests.
- `cli.py`: parses user options and prints JSON metrics.

No benchmark results from the paper are asserted by the tests. Running tests verifies code paths and small synthetic examples only.
