"""FD-15: Train logistic regression on metadata features.

This script loads ``dataset_v1.parquet`` (produced by ``build_dataset.py``),
selects only the columns required for the metadata model, trains a
``LogisticRegression`` classifier, evaluates ROC‑AUC on ``val`` and ``test_real``
splits and saves the model to ``models/meta_lr.pkl``.
"""

from __future__ import annotations

import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Keep only needed columns
    cols = [
        "func_len",
        "param_cnt",
        "has_args",
        "has_returns",
        "has_raises",
        "label_stale",
        "split",
    ]
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns for FD-15 model: {missing}")
    return df[cols]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert boolean flags to int for the model
    df = df.copy()
    bool_cols = ["has_args", "has_returns", "has_raises"]
    for c in bool_cols:
        df[c] = df[c].astype(int)
    return df


def evaluate(df: pd.DataFrame, model: LogisticRegression) -> dict:
    results = {}
    for split in ["val", "test_real"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        X = sub.drop(columns=["label_stale", "split"])
        y = sub["label_stale"]
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        results[split] = auc
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="FD-15: train metadata logistic regression")
    p.add_argument("--input", type=str, required=True, help="Path to dataset_v1.parquet")
    p.add_argument("--model-out", type=str, default="models/meta_lr.pkl", help="Where to save the trained model")
    p.add_argument("--report", type=str, default="reports/e8_meta_metrics.md", help="Markdown report path")
    args = p.parse_args()

    df = load_data(Path(args.input))
    df = prepare_features(df)

    # Split into train (train+val) for fitting – we fit on train only
    train_df = df[df["split"] == "train"]
    X_train = train_df.drop(columns=["label_stale", "split"])
    y_train = train_df["label_stale"]

    model = LogisticRegression(max_iter=1000, n_jobs=1)
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate(df, model)

    # Save model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    # Write markdown report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FD-15 Metadata Logistic Regression Metrics\n\n")
        for split, auc in metrics.items():
            f.write(f"- ROC‑AUC on **{split}**: {auc:.4f}\n")
        f.write("\n---\n\nGenerated automatically by FD‑15 pipeline.\n")

    print("Training complete. Metrics:")
    for s, a in metrics.items():
        print(f"{s}: {a:.4f}")


if __name__ == "__main__":
    main()
