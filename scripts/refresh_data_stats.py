"""Обновление reports/data_stats.md из актуального parquet + манифест (FD-12 + gold-патч FD-18).

Исходный файл описывал сборку 27143 строк (до gold-патча). Пересборка всего
dataset_v1 ради одной витрины рискованна (нужно повторять gold-патч), поэтому
статистика регенерируется из проверенного parquet (validate_dataset.py PASS)
и манифеста. Формат — как у build_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_MANIFEST = PROJECT_ROOT / "reports" / "dataset_v1.manifest.json"
OUT_STATS = PROJECT_ROOT / "reports" / "data_stats.md"


def main() -> int:
    ap = argparse.ArgumentParser(description="Обновить data_stats.md из parquet+манифест.")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--manifest", default=str(IN_MANIFEST))
    ap.add_argument("--out", default=str(OUT_STATS))
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    n = len(df)
    assert n == man["totals"]["rows"], f"parquet {n} != manifest {man['totals']['rows']}"
    by_split = df["split"].value_counts().to_dict()
    assert by_split == man["totals"]["by_split"], "сплиты не сошлись с манифестом"

    inputs = man["inputs"]
    in_rows = {k: v["rows"] for k, v in inputs.items()}
    params = man.get("params", {})
    dedup = int(params.get("dedup_removed", 0))
    nums = re.findall(r"\d+", params.get("val_isolation", ""))
    n_valsynth, n_tsleak = (int(nums[0]), int(nums[1])) if len(nums) >= 2 else (0, 0)
    sig_mm = int(params.get("sig_mismatch_stored", 0))
    val_projects = sorted(man.get("val_projects", []))
    n_gold = int((df["split"] == "gold").sum())
    w = params.get("weights", {})

    grp = (df.groupby(["split", "source", "label_stale", "drift_type"])
           .size().reset_index(name="n")
           .sort_values(["split", "source", "label_stale", "drift_type"]))

    lines = ["# dataset_v1: статистика сборки (FD-12 + gold-патч FD-18)", "",
             f"- Строк: {n} (входы: codocbench {in_rows.get('pairs_codocbench')}, "
             f"negatives {in_rows.get('pairs_negatives')}, ast_v2 {in_rows.get('synth_ast_v2')}, "
             f"llm {in_rows.get('synth_llm_main700')}; дедуп pair_id −{dedup}, "
             f"синтетика val-проектов −{n_valsynth}, виденные функции из test_synth −{n_tsleak}; "
             f"gold-патч FD-18: +{n_gold} строк split=gold)",
             f"- Сплиты: {by_split}",
             f"- Вал-проектов ({len(val_projects)}): {', '.join(val_projects)}",
             "- Val — только реальные строки; val-проекты отсутствуют в train/test_synth; "
             "test_synth — только функции без реальных строк; "
             "пересечения по function_hash train/val, train/test_real, train/test_synth отсутствуют (assert в сборке)",
             f"- sig_type пересчитан, расхождений со stored: {sig_mm}",
             f"- Веса: real_coupled {w.get('real_coupled')}, strong {w.get('real_strong_neg')}, "
             f"weak {w.get('real_weak_neg')}, ast {w.get('ast')}, llm_p1 {w.get('llm_p1')}*, "
             f"llm_p2 {w.get('llm_p2')} (*решение команды); gold 0.0",
             "", "## split × source × label × drift_type", "",
             "| split | source | label | drift_type | n |", "|---|---|---|---|---|"]
    for _, r in grp.iterrows():
        lines.append(f"| {r['split']} | {r['source']} | {r['label_stale']} | {r['drift_type']} | {r['n']} |")
    lines += ["", "Манифест: `dataset_v1.manifest.json` (хэши входов, параметры, версии).",
              f"Файл: `dataset_v1.parquet` ({n} строк)."]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Статистика: {out} ({n} строк, split={by_split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
