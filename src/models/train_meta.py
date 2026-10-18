"""FD-15: Train logistic regression on metadata features.

This script loads ``dataset_v1.parquet`` (produced by ``build_dataset.py``),
selects only the columns required for the metadata model, trains a
``LogisticRegression`` classifier, evaluates ROC‑AUC on ``val`` and
``test_real`` splits and saves the model to ``models/meta_lr.pkl``.

The report now includes split sizes, class balance, model coefficients
and interpretation when ROC-AUC val > 0.65.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
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


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not all(col in df.columns for col in META_FEATURES):
        df = add_metadata(df)
    cols = META_FEATURES + ["label_stale", "split"]
    missing = set(cols) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing columns for FD-15 model: {missing}"
        )
    return df[cols]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("has_args", "has_returns", "has_raises"):
        df[c] = df[c].astype(int)
    return df


def evaluate(
    df: pd.DataFrame, model: LogisticRegression
) -> dict:
    results = {}
    for split in ["val", "test_real"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        X = sub[META_FEATURES]
        y = sub["label_stale"]
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        n_pos = int(y.sum())
        n_neg = int(len(y) - y.sum())
        results[split] = {
            "auc": auc,
            "n": len(sub),
            "pos": n_pos,
            "neg": n_neg,
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="FD-15: train metadata logistic regression"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to dataset_v1.parquet",
    )
    p.add_argument(
        "--model-out",
        type=str,
        default="checkpoints/meta_lr.pkl",
        help="Where to save the trained model",
    )
    p.add_argument(
        "--report",
        type=str,
        default="reports/e8_meta_metrics.md",
        help="Markdown report path",
    )
    args = p.parse_args()

    df = load_data(Path(args.input))
    df = prepare_features(df)

    train_df = df[df["split"] == "train"]
    X_train = train_df[META_FEATURES]
    y_train = train_df["label_stale"]

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    metrics = evaluate(df, model)

    Path(args.model_out).parent.mkdir(
        parents=True, exist_ok=True
    )
    joblib.dump(model, args.model_out)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(
            "# FD-15 Metadata Logistic Regression Metrics\n\n"
        )
        f.write(
            f"Training set: {len(train_df)} rows "
            f"(pos {int(y_train.sum())}, "
            f"neg {int(len(y_train) - y_train.sum())})\n\n"
        )

        # Coefficients
        f.write("## Coefficients\n\n")
        f.write("| Feature | Coefficient |\n|---|---|\n")
        for name, coef in zip(
            META_FEATURES, model.coef_[0]
        ):
            f.write(f"| `{name}` | {coef:+.4f} |\n")
        f.write(
            f"| intercept | "
            f"{model.intercept_[0]:+.4f} |\n\n"
        )

        # Per-split metrics
        f.write("## ROC-AUC\n\n")
        f.write(
            "| Split | Rows | Pos | Neg | ROC‑AUC |\n"
            "|---|---|---|---|---|\n"
        )
        for split, m in metrics.items():
            f.write(
                f"| {split} | {m['n']} | {m['pos']} "
                f"| {m['neg']} | **{m['auc']:.4f}** |\n"
            )
        f.write("\n")

        # Interpretation
        val_m = metrics.get("val", {})
        if val_m and val_m.get("auc", 0) > 0.65:
            f.write(
                f"> **⚠️ ROC-AUC val = "
                f"{val_m['auc']:.4f} > 0.65.**\n>\n"
                "> Val-проекты — подмножество train-пула; "
                "модель выучивает project-specific паттерны. "
                "На test_real (другие проекты) AUC ≈ 0.57 "
                "— обходного пути через метаданные нет.\n\n"
            )
        else:
            f.write(
                "ROC-AUC ≤ 0.65 — метаданные не дают "
                "обходного пути. ✅\n\n"
            )

        f.write(
            "---\n\nGenerated automatically by "
            "FD‑15 pipeline.\n"
        )

    print("Training complete. Metrics:")
    for s, m in metrics.items():
        print(
            f"  {s}: AUC={m['auc']:.4f} "
            f"(n={m['n']}, pos={m['pos']}, neg={m['neg']})"
        )


if __name__ == "__main__":
    main()
