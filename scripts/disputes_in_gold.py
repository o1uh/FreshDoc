from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIS = PROJECT_ROOT / "reports" / "disagreements.csv"
GOLD = PROJECT_ROOT / "data" / "interim" / "gold_candidates_full.parquet"
OUT = PROJECT_ROOT / "reports" / "disputes_in_gold.csv"


def main() -> None:
    dis = pd.read_csv(DIS)
    gold = pd.read_parquet(GOLD)
    hit = dis[dis["pair_id"].isin(set(gold["pair_id"]))].copy()
    hit.to_csv(OUT, index=False)
    print(f"споров внутри gold-200: {len(hit)}")
    print(hit.groupby(["label", "label_stale"]).size().to_string())


if __name__ == "__main__":
    main()
