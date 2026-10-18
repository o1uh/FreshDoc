"""FD-15 shortcut token analysis script.

This script loads the dataset (processed parquet), extracts the `diff_code` and `label_stale`
columns, computes token frequencies for positive (stale) and negative pairs, filters tokens
according to shortcut criteria (default: >=30% in positives and <5% in negatives), and writes a
markdown report to `reports/e8_shortcut.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import sys, os
# Ensure the project root is in PYTHONPATH so that 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.token_analysis import token_frequencies, filter_shortcut_tokens


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"diff_code", "label_stale"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns for shortcut analysis: {missing}")
    return df[["diff_code", "label_stale"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-15 shortcut token analysis")
    parser.add_argument("--input", type=str, required=True, help="Path to dataset_v1.parquet")
    parser.add_argument(
        "--report",
        type=str,
        default="reports/e8_shortcut.md",
        help="Path to write markdown report",
    )
    args = parser.parse_args()

    df = load_data(Path(args.input))
    pairs = list(zip(df["diff_code"].astype(str), df["label_stale"].astype(int)))
    freqs = token_frequencies(pairs)
    shortcut_tokens = filter_shortcut_tokens(freqs)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FD-15 Shortcut Token Analysis\n\n")
        f.write(f"Total distinct tokens: {len(freqs)}\n\n")
        f.write(f"Shortcut tokens (pos≥30%, neg<5%): {len(shortcut_tokens)}\n\n")
        f.write("## Tokens\n\n")
        for token in sorted(shortcut_tokens):
            pos_frac, neg_frac = freqs[token]
            f.write(f"- `{token}` – positive fraction: {pos_frac:.2%}, negative fraction: {neg_frac:.2%}\n")
        f.write("\n---\n\nGenerated automatically by FD‑15 pipeline.\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
