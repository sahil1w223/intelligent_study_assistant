import tempfile
import unittest
from pathlib import Path

from iosched.features import encode_request, encode_workload_window
from iosched.learning import LatencyPredictor, train_classifier, train_latency_model
from iosched.mlp import MLP
from iosched.models import IORequest


def make(index, operation="read", tag="DATA_DIR", service=1.0, workload="a"):
    return IORequest(str(index), float(index), operation, tag, 4096, index * 8, service, workload)


class MLTests(unittest.TestCase):
    def test_feature_width_is_fixed_and_history_bounded(self):
        request = make(0)
        self.assertEqual(len(encode_request(request, [], 16)), 26 * 17)
        self.assertEqual(len(encode_workload_window([request], 16)), 28 * 16)
        self.assertEqual(len(encode_request(request, [make(i) for i in range(30)], 16)), 26 * 17)

    def test_classification_predicts_probabilities_and_trains(self):
        model = MLP(2, [4, 2], 2, "classification", seed=3)
        model.fit([[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], epochs=120, learning_rate=0.05)
        prediction = model.predict([0.0, 0.0])
        self.assertAlmostEqual(sum(prediction), 1.0, places=7)
        self.assertGreater(prediction[0], prediction[1])

    def test_model_round_trip(self):
        model = MLP(2, [3], 1)
        expected = model.predict([0.2, -0.4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            actual = MLP.load(path).predict([0.2, -0.4])
        self.assertEqual(expected, actual)

    def test_training_helpers_return_usable_models(self):
        trace = [make(i, service=(0.1 if i % 2 else 2.0), workload=("a" if i % 2 else "b")) for i in range(12)]
        latency = train_latency_model(trace, history_size=2, hidden_sizes=[4, 2], epochs=5)
        estimate = LatencyPredictor(latency, history_size=2).predict_ms(make(20), trace)
        self.assertGreater(estimate, 0)
        classifier, labels = train_classifier(trace, history_size=2, hidden_sizes=[4, 2], epochs=5)
        self.assertEqual(len(classifier.predict(encode_workload_window(trace[-2:], 2))), len(labels))

    def test_invalid_features_rejected(self):
        model = MLP(2, [2], 1)
        with self.assertRaises(ValueError):
            model.predict([1.0])


if __name__ == "__main__":
    unittest.main()
