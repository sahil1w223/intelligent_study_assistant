import io
import unittest

from iosched.traces import read_trace, synthetic_burst


class TraceTests(unittest.TestCase):
    def test_reads_trace_with_optional_workload(self):
        trace = read_trace(io.StringIO(
            "request_id,arrival_ms,operation,tag,size_bytes,lba,service_ms\n"
            "a,0,read,META,4096,5,1.25\n"
        ))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0].request_id, "a")
        self.assertEqual(trace[0].workload, "unknown")

    def test_missing_columns_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            read_trace(io.StringIO("request_id,arrival_ms\na,0\n"))

    def test_invalid_row_has_line_number(self):
        with self.assertRaisesRegex(ValueError, "row 2"):
            read_trace(io.StringIO(
                "request_id,arrival_ms,operation,tag,size_bytes,lba,service_ms\n"
                "a,nope,read,META,4096,5,1\n"
            ))

    def test_synthetic_burst_is_reproducible(self):
        self.assertEqual(synthetic_burst(10, 1), synthetic_burst(10, 1))


if __name__ == "__main__":
    unittest.main()
