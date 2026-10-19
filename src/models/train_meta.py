"""FD-15: logistic regression on metadata features (library + CLI).

Trains on **real train rows only** (``source`` in ``{codocbench,
history_neg}``) per workflow.md FD-15, evaluates ROC-AUC on ``val`` and
``test_real``. The unified E8 report (``reports/e8_shortcut.md``) is built
by ``scripts/e8_shortcut_analysis.py``, which reuses the helpers below —
single source of truth for the numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.metadata_extractor import add_metadata

META_FEATURES = [
    "func_len",
    "param_cnt",
    "has_args",
    "has_returns",
    "has_raises",
]

#: Sources used for training per workflow.md FD-15 (real rows only).
REAL_SOURCES = ("codocbench", "history_neg")

RANDOM_STATE = 42
N_BOOTSTRAP = 1000


def ensure_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach metadata columns if missing; binarise section flags to int."""
    if not all(col in df.columns for col in META_FEATURES):
        df = add_metadata(df)
    df = df.copy()
    for c in ("has_args", "has_returns", "has_raises"):
        df[c] = df[c].astype(int)
    return df


def real_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Real train rows only (no synth, no gold)."""
    return df[
        (df["split"] == "train") & (df["source"].isin(REAL_SOURCES))
    ]


def train_metadata_model(train_df: pd.DataFrame) -> LogisticRegression:
    """Fit balanced LogisticRegression on metadata features."""
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(train_df[META_FEATURES].values,
              train_df["label_stale"].values)
    return model


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """95% bootstrap CI for ROC-AUC (percentile, fixed seed)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_metadata(
    df: pd.DataFrame, model: LogisticRegression
) -> dict:
    """ROC-AUC (+ bootstrap CI, sizes, balance) on val / test_real."""
    results = {}
    for split in ("val", "test_real"):
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        y = sub["label_stale"].to_numpy(dtype=int)
        probs = model.predict_proba(sub[META_FEATURES].values)[:, 1]
        lo, hi = bootstrap_auc_ci(y, probs)
        results[split] = {
            "auc": float(roc_auc_score(y, probs)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": int(len(sub)),
            "pos": int(y.sum()),
            "neg": int(len(sub) - y.sum()),
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="FD-15: train metadata logistic regression"
    )
    p.add_argument("--input", type=str, required=True,
                   help="Path to dataset_v1.parquet")
    p.add_argument("--model-out", type=str,
                   default="checkpoints/meta_lr.pkl",
                   help="Where to save the trained model")
    p.add_argument("--report", type=str, default=None,
                   help="Optional: write short metrics markdown here "
                        "(unified E8 report is built by "
                        "scripts/e8_shortcut_analysis.py)")
    args = p.parse_args()

    df = ensure_metadata(pd.read_parquet(Path(args.input)))
    train_df = real_train_frame(df)
    model = train_metadata_model(train_df)
    metrics = evaluate_metadata(df, model)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    for split, m in metrics.items():
        print(f"  {split}: AUC={m['auc']:.4f} "
              f"[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] "
              f"(n={m['n']}, pos={m['pos']}, neg={m['neg']})")
    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with rp.open("w", encoding="utf-8") as f:
            f.write("# FD-15 Metadata Logistic Regression Metrics\n\n")
            f.write(f"Training: {len(train_df)} real train rows\n\n")
            f.write("| Split | Rows | Pos | Neg | ROC-AUC | 95% CI |\n"
                    "|---|---|---|---|---|---|\n")
            for split, m in metrics.items():
                f.write(f"| {split} | {m['n']} | {m['pos']} | {m['neg']} "
                        f"| **{m['auc']:.4f}** "
                        f"| [{m['ci_lo']:.4f}, {m['ci_hi']:.4f}] |\n")


if __name__ == "__main__":
    main()
