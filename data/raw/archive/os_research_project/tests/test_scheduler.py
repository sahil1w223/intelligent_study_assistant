import unittest

from iosched.models import IORequest
from iosched.scheduler import choose_fifo, choose_sjf
from iosched.simulation import simulate


def req(name, arrival, service):
    return IORequest(name, arrival, "read", "DATA_DIR", 4096, 100, service)


class SchedulerTests(unittest.TestCase):
    def test_fifo_uses_arrival_order(self):
        queue = [req("late", 3, 1), req("early", 1, 9)]
        self.assertEqual(choose_fifo(queue, []).request_id, "early")

    def test_sjf_prefers_low_predicted_latency(self):
        queue = [req("slow", 0, 4), req("fast", 0, 1)]
        selected = choose_sjf(queue, [], lambda request, _: request.service_ms)
        self.assertEqual(selected.request_id, "fast")

    def test_aging_breaks_equal_prediction_in_favor_of_older_request(self):
        queue = [req("old", 0, 5), req("new", 0, 5)]
        selected = choose_sjf(queue, [], lambda _request, _: 5.0, aging_ms=0.01)
        self.assertEqual(selected.request_id, "old")

    def test_sjf_burst_reduces_mean_response_time(self):
        requests = [req("long", 0, 10), req("short-1", 0, 1), req("short-2", 0, 1)]
        _, fifo = simulate(requests, "fifo")
        _, sjf = simulate(requests, "sjf", immediate_threshold=2)
        self.assertLess(sjf.average_response_ms, fifo.average_response_ms)

    def test_future_arrival_does_not_wait_for_idle_service(self):
        requests = [req("a", 0, 2), req("b", 10, 1)]
        results, _ = simulate(requests, "fifo")
        by_id = {result.request.request_id: result for result in results}
        self.assertEqual(by_id["b"].start_ms, 10)

    def test_invalid_request_rejected(self):
        with self.assertRaises(ValueError):
            IORequest("bad", 0, "execute", "DATA", 512, 0, 1)


if __name__ == "__main__":
    unittest.main()
