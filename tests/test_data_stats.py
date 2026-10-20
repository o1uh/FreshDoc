"""Регресс-тест: reports/data_stats.md не должен отставать от датасета.

Закрывает замечание «data_stats устарел (27143 без gold-патча)»: таблица
и итоги в md сверяются с parquet + манифестом. Обновляется скриптом
scripts/refresh_data_stats.py (без пересборки датасета). Тест выполняется
только там, где есть данные (локально / Kaggle); в CI датасета нет
(data/ в .gitignore по R15) — тест скипается, конвейер гоняется на фикстурах.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "reports" / "data_stats.md"
MANIFEST = ROOT / "reports" / "dataset_v1.manifest.json"
DATASET = ROOT / "data" / "processed" / "dataset_v1.parquet"


def _parse_table(text: str) -> dict:
    rows: dict[tuple, int] = {}
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] == "split" or not cells[4].isdigit():
            continue
        key = (cells[0], cells[1], int(cells[2]), cells[3])
        rows[key] = int(cells[4])
    return rows


def test_data_stats_matches_dataset_and_manifest():
    if not DATASET.exists():
        pytest.skip("нет data/processed/dataset_v1.parquet "
                    "(data/ в .gitignore по R15; проверка — локально перед заливкой в Kaggle)")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    df = pd.read_parquet(DATASET)
    text = MD.read_text(encoding="utf-8")

    # итоги в шапке = манифест = parquet
    assert str(man["totals"]["rows"]) in text.splitlines()[2]
    assert len(df) == man["totals"]["rows"]
    assert df["split"].value_counts().to_dict() == man["totals"]["by_split"]

    # каждая строка таблицы = groupby parquet
    expected = {(s, src, int(lbl), d): int(n)
                for (s, src, lbl, d), n in
                df.groupby(["split", "source", "label_stale", "drift_type"]).size().items()}
    assert _parse_table(text) == expected
