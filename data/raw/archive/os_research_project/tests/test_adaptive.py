import unittest

from iosched.adaptive import estimate_adaptive_throughput


class AdaptiveTests(unittest.TestCase):
    def test_expected_throughput_uses_best_scheduler_per_prediction(self):
        confusion = {
            "seq": {"seq": 0.9, "rand": 0.2},
            "rand": {"seq": 0.1, "rand": 0.8},
        }
        throughput = {
            "fast-seq": {"seq": 100.0, "rand": 10.0},
            "fast-rand": {"seq": 20.0, "rand": 80.0},
        }
        estimate = estimate_adaptive_throughput(confusion, throughput)
        self.assertEqual(estimate.best_scheduler_by_predicted_workload, {"seq": "fast-seq", "rand": "fast-rand"})
        self.assertAlmostEqual(estimate.throughput_by_true_workload["seq"], 92.0)
        self.assertAlmostEqual(estimate.throughput_by_true_workload["rand"], 66.0)
        self.assertAlmostEqual(estimate.overall_throughput, 79.0)

    def test_accepts_percentage_confusion_matrix(self):
        estimate = estimate_adaptive_throughput(
            {"one": {"one": 90, "two": 10}, "two": {"one": 10, "two": 90}},
            {"scheduler": {"one": 12, "two": 8}},
        )
        self.assertEqual(estimate.throughput_by_true_workload, {"one": 12.0, "two": 8.0})

    def test_invalid_probability_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            estimate_adaptive_throughput({"x": {"x": 0.5}}, {"s": {"x": 1}})


if __name__ == "__main__":
    unittest.main()
