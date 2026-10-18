"""FD-20: бейзлайны M0 (правило) и M1 (градиентный бустинг).

    python scripts/baselines_m0_m1.py [--dataset ...] [--seed 42]

M0 (R10): p_stale = 1, если sig_type != none, иначе 0.
M1 (R10): HistGradientBoostingClassifier на признаках src/models/baselines.py;
обучение на split==train с весами weight; порог — максимум F1 на реальных
строках val (тот же принцип, что в scripts/evaluate.py; сам evaluate.py
пересчитает порог независимо).

Выходы:
- reports/preds/m0.csv, reports/preds/m1.csv (pair_id, p_stale; gitignored);
- checkpoints/m1_hgb.joblib (gitignored);
- reports/m1_report.json (конфиг, порог по val, время, объёмы).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.data.dataset import REAL_SOURCES
from src.models.baselines import (
    FEATURE_COLS,
    extract_features,
    predict_m0,
    predict_proba_m1,
    train_m1,
)
from src.models.metrics import (
    evaluate_binary,
    group_mask,
    pick_threshold_f1,
    recall_at_precision,
)

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_M0 = PROJECT_ROOT / "reports" / "preds" / "m0.csv"
OUT_M1 = PROJECT_ROOT / "reports" / "preds" / "m1.csv"
OUT_MODEL = PROJECT_ROOT / "checkpoints" / "m1_hgb.joblib"
OUT_REPORT = PROJECT_ROOT / "reports" / "m1_report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-20: бейзлайны M0 и M1 (R10).")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--out_m0", default=str(OUT_M0))
    ap.add_argument("--out_m1", default=str(OUT_M1))
    ap.add_argument("--out_model", default=str(OUT_MODEL))
    ap.add_argument("--out_report", default=str(OUT_REPORT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_leaf_nodes", type=int, default=31)
    ap.add_argument("--min_samples_leaf", type=int, default=20)
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Датасет отсутствует: {ds_path}")
        return 2
    df = pd.read_parquet(ds_path)
    need = {"pair_id", "sig_old", "diff_code", "doc_old", "code_old",
            "code_new", "sig_type", "split", "source", "label_stale", "weight"}
    missing = need - set(df.columns)
    if missing:
        print(f"В датасете нет колонок: {sorted(missing)}")
        return 2
    print(f"Датасет: {ds_path} ({len(df)} строк)")
    print(f"Сплиты: {df['split'].value_counts().to_dict()}")

    t0 = time.time()
    # --- M0: правило по sig_type (прогноз для всех строк) ---
    p0 = predict_m0(df)
    out_m0 = Path(args.out_m0)
    out_m0.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair_id": df["pair_id"].astype(str),
                  "p_stale": p0.to_numpy()}).to_csv(out_m0, index=False)
    print(f"M0: {out_m0} ({len(p0)} строк)")

    # --- M1: признаки + обучение на train ---
    t1 = time.time()
    X = extract_features(df)
    print(f"Признаки: {len(X)} строк x {len(FEATURE_COLS)} "
          f"({(time.time() - t1):.1f} c)")
    assert list(X.columns) == FEATURE_COLS
    assert not X.isna().any().any(), "NaN в признаках M1!"

    m_train = df["split"].eq("train").to_numpy()
    y_train = df.loc[m_train, "label_stale"].to_numpy(dtype=int)
    w_train = df.loc[m_train, "weight"].to_numpy(dtype=float)
    X_train = X.loc[m_train]
    print(f"Train: {len(X_train)} строк "
          f"(pos={int(y_train.sum())}, neg={int((y_train == 0).sum())})")

    t2 = time.time()
    clf = train_m1(X_train, y_train, sample_weight=w_train, seed=args.seed,
                   max_iter=args.max_iter, learning_rate=args.learning_rate,
                   max_leaf_nodes=args.max_leaf_nodes,
                   min_samples_leaf=args.min_samples_leaf)
    print(f"Обучение M1: {(time.time() - t2):.1f} c "
          f"(n_iter={clf.n_iter_}, seed={args.seed})")

    p1 = predict_proba_m1(clf, X)
    assert ((p1 >= 0.0) & (p1 <= 1.0)).all()
    out_m1 = Path(args.out_m1)
    out_m1.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair_id": df["pair_id"].astype(str),
                  "p_stale": p1}).to_csv(out_m1, index=False)
    print(f"M1: {out_m1} ({len(p1)} строк)")

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS,
                 "seed": args.seed,
                 "params": {"max_iter": args.max_iter,
                            "learning_rate": args.learning_rate,
                            "max_leaf_nodes": args.max_leaf_nodes,
                            "min_samples_leaf": args.min_samples_leaf}},
                out_model)
    print(f"Модель: {out_model}")

    # Порог по реальным строкам val (для отчёта; evaluate.py считает сам).
    m_val = df["split"].eq("val").to_numpy() & df["source"].isin(
        list(REAL_SOURCES)).to_numpy()
    yv = df.loc[m_val, "label_stale"].to_numpy(dtype=int)
    thr_m0 = pick_threshold_f1(yv, p0.to_numpy()[m_val])
    thr_m1 = pick_threshold_f1(yv, p1[m_val])
    print(f"Порог max-F1 на val real (n={len(yv)}, pos={int(yv.sum())}): "
          f"M0={thr_m0:.4f} M1={thr_m1:.4f}")

    # Диагностика для FD-25 (замечание: глобальный max-F1 даёт экстремальный
    # порог; рабочему порогу нужна калибровка E5 и уровень precision E6).
    # Группы val при глобальных порогах + recall при precision >= 0.8.
    val_diag: dict = {}
    sig_val = df.loc[m_val, "sig_type"].to_numpy()
    for group in ("all", "sig_changed", "sig_same"):
        m = group_mask(sig_val, group)
        yg = yv[m]
        entry = {"n": int(m.sum()), "n_pos": int(yg.sum())}
        for name, scores, thr in (("M0", p0.to_numpy()[m_val], thr_m0),
                                  ("M1", p1[m_val], thr_m1)):
            sg = scores[m]
            ev = evaluate_binary(yg, sg, threshold=float(thr),
                                 n_boot=0, seed=42)
            entry[name] = {
                "threshold": float(thr),
                "precision": ev["precision"],
                "recall": ev["recall"],
                "f1": ev["f1"],
                "recall_at_p80": recall_at_precision(yg, sg, target=0.8),
            }
        val_diag[group] = entry
        print(f"val/{group}: n={entry['n']} pos={entry['n_pos']} | "
              f"M1 P={entry['M1']['precision']:.3f} "
              f"R={entry['M1']['recall']:.3f} "
              f"R@P80={entry['M1']['recall_at_p80']:.3f}")

    report = {
        "task": "FD-20",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(ds_path),
        "dataset_rows": int(len(df)),
        "seed": int(args.seed),
        "params": {"max_iter": args.max_iter,
                   "learning_rate": args.learning_rate,
                   "max_leaf_nodes": args.max_leaf_nodes,
                   "min_samples_leaf": args.min_samples_leaf},
        "model_class": "HistGradientBoostingClassifier",
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "n_train": int(m_train.sum()),
        "n_train_pos": int(y_train.sum()),
        "n_val_real": int(m_val.sum()),
        "n_val_real_pos": int(yv.sum()),
        "threshold_val_m0": float(thr_m0),
        "threshold_val_m1": float(thr_m1),
        "val_group_diagnostics": val_diag,
        "train_time_sec": round(time.time() - t2, 1),
        "total_time_sec": round(time.time() - t0, 1),
        "forbidden_cols_never_read": ["project", "source", "label_stale",
                                      "label_source", "drift_type", "split",
                                      "weight", "pair_id", "_leak_*", "_aux_*"],
        "outputs": {"m0": str(out_m0), "m1": str(out_m1),
                    "model": str(out_model)},
        "next": (f"python scripts/evaluate.py --pred M0={out_m0} "
                 f"--pred M1={out_m1}"),
    }
    out_rep = Path(args.out_report)
    out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"Отчёт: {out_rep}")
    print("Далее: " + report["next"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
