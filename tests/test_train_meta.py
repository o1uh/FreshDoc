"""Tests for src.models.train_meta helpers (FD-15)."""
import unittest

import numpy as np
import pandas as pd

from src.models.train_meta import (
    bootstrap_auc_ci,
    evaluate_metadata,
    real_train_frame,
    train_metadata_model,
)


def _row(split="train", source="codocbench", label=1, **kw):
    base = dict(
        func_len=10, param_cnt=2, has_args=1, has_returns=0, has_raises=0,
        label_stale=label, split=split, source=source,
    )
    base.update(kw)
    return pd.DataFrame([base])


class TestRealTrainFrame(unittest.TestCase):
    def test_excludes_synth_and_gold(self):
        df = pd.concat([
            _row(split="train", source="codocbench"),
            _row(split="train", source="history_neg"),
            _row(split="train", source="synth_ast"),
            _row(split="train", source="synth_llm"),
            _row(split="gold", source="gold", label=0),
            _row(split="val", source="codocbench"),
        ], ignore_index=True)
        out = real_train_frame(df)
        self.assertEqual(len(out), 2)
        self.assertTrue(set(out["source"]) <= {"codocbench", "history_neg"})


class TestBootstrapCi(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([0] * 20 + [1] * 20)
        s = np.array([0.1] * 20 + [0.9] * 20)
        lo, hi = bootstrap_auc_ci(y, s, n_boot=200, seed=0)
        self.assertGreaterEqual(lo, 0.99)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, hi)

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 100)
        s = rng.random(100)
        self.assertEqual(bootstrap_auc_ci(y, s), bootstrap_auc_ci(y, s))


class TestTrainEvaluate(unittest.TestCase):
    def test_separable(self):
        rows = []
        for i in range(10):
            rows.append(_row(split="train", source="codocbench",
                            label=1, param_cnt=5))
            rows.append(_row(split="train", source="history_neg",
                            label=0, param_cnt=1))
        for i in range(4):
            rows.append(_row(split="val", source="codocbench",
                            label=1, param_cnt=5))
            rows.append(_row(split="val", source="history_neg",
                            label=0, param_cnt=1))
        df = pd.concat(rows, ignore_index=True)
        model = train_metadata_model(df[df["split"] == "train"])
        metrics = evaluate_metadata(df, model)
        self.assertIn("val", metrics)
        self.assertGreater(metrics["val"]["auc"], 0.9)


if __name__ == "__main__":
    unittest.main()
