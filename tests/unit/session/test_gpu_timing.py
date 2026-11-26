# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Protocol tests; also runnable without installing the CLI's ML dependencies."""

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path


_path = Path(__file__).resolve().parents[3] / "src/winml/modelkit/session/gpu_timing.py"
_spec = importlib.util.spec_from_file_location("_gpu_timing_under_test", _path.resolve())
g = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)


class GpuTimingTests(unittest.TestCase):
    def sample(self, run=1, **changes):
        return replace(g.GpuSubmission(g.RunKey(100, run), 2, 1000, 0, 2, 5, 8), **changes)

    def analyze(self, samples, runs=(1,), **changes):
        options = {"events_lost": 0, "consumer_drained": True}
        options.update(changes)
        return g.analyze_gpu_timing(samples, [g.RunKey(100, r) for r in runs], **options)

    def test_reordered_delivery_warmup_and_other_process(self):
        result = self.analyze(
            [
                self.sample(2),
                self.sample(9),
                self.sample(1, path=1),
                self.sample(key=g.RunKey(200, 1)),
            ],
            (1, 2),
        )
        self.assertEqual(result.samples_ms, (3.0, 3.0))
        self.assertEqual(result.outside_submissions, 2)
        self.assertEqual(result.to_dict()["median_ms"], 3.0)

    def test_recorded_excluded(self):
        result = self.analyze([self.sample(1, path=0), self.sample(2)], (1, 2))
        self.assertEqual(result.recorded_runs, 1)
        self.assertEqual(len(result.samples_ms), 1)

    def test_large_ticks_preserve_precision(self):
        n = 2**60
        sample = self.sample(submit_start=n, body_start=n + 1, body_end=n + 2, submit_end=n + 3)
        self.assertEqual(self.analyze([sample]).samples_ms, (1.0,))

    def test_invalid_or_incomplete_data_rejected(self):
        cases = [
            ([], {}),
            ([self.sample(), self.sample()], {}),
            ([self.sample(path=0)], {}),
            ([self.sample(path=3)], {}),
            ([self.sample(frequency=0)], {}),
            ([self.sample(body_end=1)], {}),
            ([self.sample()], {"events_lost": 1}),
            ([self.sample()], {"consumer_drained": False}),
        ]
        for samples, options in cases:
            with self.subTest(samples=samples, options=options):  # noqa: SIM117
                with self.assertRaises(g.GpuTimingError):
                    self.analyze(samples, **options)

    def test_duplicate_run_ids_rejected(self):
        with self.assertRaises(g.GpuTimingError):
            self.analyze([self.sample()], (1, 1))


if __name__ == "__main__":
    unittest.main()
