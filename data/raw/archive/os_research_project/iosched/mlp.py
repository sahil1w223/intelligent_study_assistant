"""A small pure-Python MLP matching the paper's leaky-ReLU/softmax design."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Literal

Mode = Literal["regression", "classification"]


class MLP:
    """Fully connected network trained with stochastic gradient descent.

    Hidden layers use leaky ReLU. Classification uses a softmax output and
    cross-entropy loss; latency estimation uses one linear regression output.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int,
                 mode: Mode = "regression", seed: int = 7) -> None:
        if input_size <= 0 or output_size <= 0 or not hidden_sizes or any(n <= 0 for n in hidden_sizes):
            raise ValueError("network dimensions must be positive")
        if mode not in ("regression", "classification"):
            raise ValueError("mode must be regression or classification")
        self.input_size, self.hidden_sizes, self.output_size, self.mode = input_size, list(hidden_sizes), output_size, mode
        rng = random.Random(seed)
        dimensions = [input_size, *hidden_sizes, output_size]
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []
        for fan_in, fan_out in zip(dimensions, dimensions[1:]):
            scale = math.sqrt(2.0 / fan_in)
            self.weights.append([[rng.uniform(-scale, scale) for _ in range(fan_in)] for _ in range(fan_out)])
            self.biases.append([0.0] * fan_out)

    def predict(self, features: list[float]) -> list[float]:
        self._check_features(features)
        values = list(features)
        for layer, (matrix, bias) in enumerate(zip(self.weights, self.biases)):
            values = [sum(weight * value for weight, value in zip(row, values)) + offset
                      for row, offset in zip(matrix, bias)]
            if layer < len(self.weights) - 1:
                values = [value if value >= 0 else 0.01 * value for value in values]
            elif self.mode == "classification":
                maximum = max(values)
                exponentials = [math.exp(max(-60.0, value - maximum)) for value in values]
                total = sum(exponentials)
                values = [value / total for value in exponentials]
        return values

    def fit(self, features: list[list[float]], targets: list[list[float]],
            epochs: int = 100, learning_rate: float = 0.01, seed: int = 11) -> None:
        if not features or len(features) != len(targets):
            raise ValueError("features and targets must be non-empty and have equal lengths")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs and learning_rate must be positive")
        for x, y in zip(features, targets):
            self._check_features(x)
            if len(y) != self.output_size or any(not math.isfinite(v) for v in y):
                raise ValueError("target dimensions or values are invalid")
        order = list(range(len(features)))
        rng = random.Random(seed)
        for _ in range(epochs):
            rng.shuffle(order)
            for index in order:
                self._train_one(features[index], targets[index], learning_rate)

    def _train_one(self, features: list[float], target: list[float], rate: float) -> None:
        activations = [list(features)]
        preactivations: list[list[float]] = []
        values = list(features)
        for layer, (matrix, bias) in enumerate(zip(self.weights, self.biases)):
            raw = [sum(w * x for w, x in zip(row, values)) + b for row, b in zip(matrix, bias)]
            preactivations.append(raw)
            if layer < len(self.weights) - 1:
                values = [v if v >= 0 else 0.01 * v for v in raw]
            elif self.mode == "classification":
                maximum = max(raw)
                exp_values = [math.exp(max(-60.0, v - maximum)) for v in raw]
                total = sum(exp_values)
                values = [v / total for v in exp_values]
            else:
                values = raw
            activations.append(values)
        # Softmax + cross entropy has the compact derivative p - y.
        deltas = [[value - expected for value, expected in zip(values, target)]]
        for layer in range(len(self.weights) - 2, -1, -1):
            next_weights = self.weights[layer + 1]
            next_delta = deltas[0]
            derivative = [1.0 if z >= 0 else 0.01 for z in preactivations[layer]]
            deltas.insert(0, [derivative[i] * sum(next_weights[j][i] * next_delta[j]
                                                   for j in range(len(next_delta)))
                              for i in range(len(derivative))])
        for layer, delta in enumerate(deltas):
            previous = activations[layer]
            for row, error in zip(self.weights[layer], delta):
                for i, value in enumerate(previous):
                    row[i] -= rate * error * value
            for i, error in enumerate(delta):
                self.biases[layer][i] -= rate * error

    def _check_features(self, features: list[float]) -> None:
        if len(features) != self.input_size or any(not math.isfinite(v) for v in features):
            raise ValueError(f"expected {self.input_size} finite input features")

    def save(self, path: str | Path) -> None:
        payload = {"input_size": self.input_size, "hidden_sizes": self.hidden_sizes,
                   "output_size": self.output_size, "mode": self.mode,
                   "weights": self.weights, "biases": self.biases}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            model = cls(payload["input_size"], payload["hidden_sizes"], payload["output_size"], payload["mode"])
            if len(payload["weights"]) != len(model.weights) or len(payload["biases"]) != len(model.biases):
                raise ValueError("model layer count mismatch")
            for old, new in zip(payload["weights"], model.weights):
                if len(old) != len(new) or any(len(a) != len(b) for a, b in zip(old, new)):
                    raise ValueError("model weight dimensions mismatch")
            model.weights, model.biases = payload["weights"], payload["biases"]
            return model
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load model from {path}: {exc}") from exc
