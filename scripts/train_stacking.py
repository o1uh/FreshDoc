"""FD-24 слот 8: single-split стэкинг M1 + M2 на отложенных предсказаниях val (E2b).

    python scripts/train_stacking.py --dataset data/processed/dataset_v1.parquet \
        --m1_model checkpoints/m1_hgb.joblib \
        --m2_checkpoint checkpoints/m2_run1/best \
        --output_dir checkpoints/slot8_stacking --runs_csv reports/runs.csv

или при наличии готовых CSV предсказаний:
    python scripts/train_stacking.py --dataset data/processed/dataset_v1.parquet \
        --m1_preds reports/preds/m1.csv \
        --m2_preds reports/preds/m2.csv \
        --output_dir checkpoints/slot8_stacking --runs_csv reports/runs.csv
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
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from scripts.predict_m2 import resolve_threshold
from scripts.train_m2 import append_runs_csv
from src.data.dataset import REAL_SOURCES
from src.models.baselines import extract_features, predict_proba_m1

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "checkpoints" / "slot8_stacking"
DEFAULT_RUNS_CSV = PROJECT_ROOT / "reports" / "runs.csv"
DEFAULT_OUT_PREDS = PROJECT_ROOT / "reports" / "preds" / "m_stack.csv"


def get_m1_probs(df: pd.DataFrame, m1_preds_path: str | None, m1_model_path: str | None) -> np.ndarray:
    if m1_preds_path and Path(m1_preds_path).exists():
        print(f"Загрузка предсказаний M1: {m1_preds_path}")
        m1_df = pd.read_csv(m1_preds_path)
        merged = df[["pair_id"]].merge(m1_df[["pair_id", "p_stale"]], on="pair_id", how="left")
        if merged["p_stale"].isna().any():
            raise ValueError(f"Не все pair_id найдены в {m1_preds_path}")
        return merged["p_stale"].to_numpy(dtype=float)
    if m1_model_path and Path(m1_model_path).exists():
        print(f"Инференс M1 по модели: {m1_model_path}")
        model = joblib.load(m1_model_path)
        return predict_proba_m1(model, df)
    raise FileNotFoundError(
        "Не указан или не найден источник для M1: укажите --m1_preds или --m1_model"
    )


def get_m2_probs_and_drift(df: pd.DataFrame, m2_preds_path: str | None,
                           m2_ckpt_path: str | None, batch_size: int = 16) -> tuple[np.ndarray, list[str]]:
    if m2_preds_path and Path(m2_preds_path).exists():
        print(f"Загрузка предсказаний M2: {m2_preds_path}")
        m2_df = pd.read_csv(m2_preds_path)
        cols = ["pair_id", "p_stale"]
        if "type_pred" in m2_df.columns:
            cols.append("type_pred")
        merged = df[["pair_id"]].merge(m2_df[cols], on="pair_id", how="left")
        if merged["p_stale"].isna().any():
            raise ValueError(f"Не все pair_id найдены в {m2_preds_path}")
        p_m2 = merged["p_stale"].to_numpy(dtype=float)
        t_m2 = merged["type_pred"].fillna("none").astype(str).tolist() if "type_pred" in cols else ["none"] * len(df)
        return p_m2, t_m2
    if m2_ckpt_path and Path(m2_ckpt_path).exists():
        print(f"Инференс M2 по чекпоинту: {m2_ckpt_path}")
        # Импорт локально, чтобы не требовать GPU если переданы CSV
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from src.models.m2_data import DOC_BUDGET_TOKENS, DRIFT_CLASSES, encode_pair
        from src.models.m2_model import DualHeadCodeClassifier

        ckpt = Path(m2_ckpt_path)
        dual_cfg = {}
        if (ckpt / "dualhead_config.json").exists():
            dual_cfg = json.loads((ckpt / "dualhead_config.json").read_text(encoding="utf-8"))
        model_name = dual_cfg.get("model_name", "microsoft/unixcoder-base")
        max_len = int(dual_cfg.get("max_len", 1024))
        doc_budget = int(dual_cfg.get("doc_budget", DOC_BUDGET_TOKENS))
        drop_diff = bool(dual_cfg.get("drop_diff", False))

        tok_src = str(ckpt) if (ckpt / "tokenizer.json").exists() or (ckpt / "vocab.json").exists() else model_name
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
        if (ckpt / "config.json").exists():
            b_cfg = AutoConfig.from_pretrained(str(ckpt))
            backbone = AutoModel.from_config(b_cfg)
        else:
            backbone = AutoModel.from_pretrained(model_name)
        model = DualHeadCodeClassifier(backbone, hidden_size=backbone.config.hidden_size,
                                       num_drift_classes=len(DRIFT_CLASSES))
        weights_path = ckpt / "pytorch_model.bin" if (ckpt / "pytorch_model.bin").exists() else ckpt / "model.safetensors"
        if weights_path.suffix == ".bin":
            state = torch.load(str(weights_path), map_location="cpu")
        else:
            from safetensors.torch import load_file
            state = load_file(str(weights_path))
        model.load_state_dict(state, strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        sigs = df["sig_old"].fillna("").astype(str).tolist()
        diffs = ["" for _ in range(len(df))] if drop_diff else df["diff_code"].fillna("").astype(str).tolist()
        docs = df["doc_old"].fillna("").astype(str).tolist()

        p_all, t_all = [], []
        # batch_size передан аргументом функции (дефолт 16)
        with torch.no_grad():
            for start in range(0, len(df), batch_size):
                b_ids, b_masks = [], []
                for j in range(start, min(start + batch_size, len(df))):
                    e = encode_pair(tok, sigs[j], diffs[j], docs[j], max_len=max_len, doc_budget=doc_budget)
                    b_ids.append(e["input_ids"])
                    b_masks.append(e["attention_mask"])
                max_n = max(len(x) for x in b_ids)
                pad = tok.pad_token_id or 0
                ids = torch.tensor([x + [pad] * (max_n - len(x)) for x in b_ids], dtype=torch.long).to(device)
                masks = torch.tensor([x + [0] * (max_n - len(x)) for x in b_masks], dtype=torch.long).to(device)
                out = model(input_ids=ids, attention_mask=masks)
                p = torch.sigmoid(out["logits_stale"]).cpu().numpy()
                t = out["logits_drift"].argmax(dim=-1).cpu().numpy()
                p_all.extend([float(v) for v in p])
                t_all.extend([DRIFT_CLASSES[int(v)] for v in t])
        return np.array(p_all, dtype=float), t_all

    raise FileNotFoundError(
        "Не указан или не найден источник для M2: укажите --m2_preds или --m2_checkpoint"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-24 слот 8: single-split стэкинг M1 + M2 (E2b).")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--m1_model", default=None, help="путь к m1_hgb.joblib")
    ap.add_argument("--m1_preds", default=None, help="путь к m1.csv (reports/preds/m1.csv)")
    ap.add_argument("--m2_checkpoint", default=None, help="путь к папке best/ победителя M2")
    ap.add_argument("--m2_preds", default=None, help="путь к m2.csv (reports/preds/m2.csv)")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--runs_csv", default=str(DEFAULT_RUNS_CSV))
    ap.add_argument("--out_preds", default=str(DEFAULT_OUT_PREDS))
    ap.add_argument("--batch_size", type=int, default=16,
                    help="размер батча для инференса M2 (дефолт 16; предотвращает OOM на 2048)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог гейта R10 (по умолчанию из config/risk.yaml или 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"Датасет не найден: {ds_path}", file=sys.stderr)
        return 2

    df = pd.read_parquet(ds_path)
    print(f"Датасет: {ds_path} ({len(df)} строк)")

    # 1. Получить вероятности M1 и M2
    p_m1 = get_m1_probs(df, args.m1_preds, args.m1_model)
    p_m2, type_pred = get_m2_probs_and_drift(df, args.m2_preds, args.m2_checkpoint, batch_size=args.batch_size)

    # 2. Выборка для обучения мета-модели: строго val_real (R11 / model_strategy §4)
    m_val = df["split"].eq("val").to_numpy() & df["source"].isin(list(REAL_SOURCES)).to_numpy()
    m_test_real = df["split"].eq("test_real").to_numpy()
    m_gold = df["split"].eq("gold").to_numpy()

    X = np.column_stack([p_m1, p_m2])
    y = df["label_stale"].to_numpy(dtype=int)

    X_val = X[m_val]
    y_val = y[m_val]

    print(f"Обучение стэкинга на val_real: n={len(X_val)}, pos={int(y_val.sum())}")

    # 3. Мета-модель: логистическая регрессия на вероятностях [p_m1, p_m2]
    clf = LogisticRegression(C=1.0, random_state=args.seed, solver="lbfgs")
    clf.fit(X_val, y_val)

    # 4. Предсказания стэкинга для всего датасета
    p_stack = clf.predict_proba(X)[:, 1]

    val_pr_auc = float(average_precision_score(y_val, p_stack[m_val]))
    test_real_pr_auc = float(average_precision_score(y[m_test_real], p_stack[m_test_real]))
    gold_pr_auc = float(average_precision_score(y[m_gold], p_stack[m_gold]))

    w_m1 = float(clf.coef_[0][0])
    w_m2 = float(clf.coef_[0][1])
    intercept = float(clf.intercept_[0])
    train_time = time.time() - t0

    print("--- Результаты стэкинга M1+M2 ---")
    print(f"Веса логистической регрессии: w_M1={w_m1:.4f}, w_M2={w_m2:.4f}, bias={intercept:.4f}")
    print(f"PR-AUC val_real:  {val_pr_auc:.4f}")
    print(f"PR-AUC test_real: {test_real_pr_auc:.4f}")
    print(f"PR-AUC gold:      {gold_pr_auc:.4f}")

    # 5. Сохранение артефактов
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "stacking_model.joblib"
    joblib.dump({"model": clf, "w_m1": w_m1, "w_m2": w_m2, "intercept": intercept}, model_path)
    print(f"Модель сохранена: {model_path}")

    # Таблица предсказаний
    tau = resolve_threshold(args.threshold)
    gated = [t if p > tau else "none" for p, t in zip(p_stack, type_pred)]
    preds_df = pd.DataFrame({
        "pair_id": df["pair_id"].astype(str),
        "p_stale": p_stack,
        "type_pred": type_pred,
        "type_pred_gated": gated,
    })
    out_preds = Path(args.out_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_preds, index=False)
    print(f"Предсказания сохранены: {out_preds}")

    # Дублирование предсказаний в output_dir/m_stack.csv, если путь отличается
    ckpt_preds = out_dir / "m_stack.csv"
    if ckpt_preds.resolve() != out_preds.resolve():
        ckpt_preds.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(ckpt_preds, index=False)
        print(f"Дубликат предсказаний сохранен в чекпоинт: {ckpt_preds}")

    # run_meta.json
    meta = {
        "task": "FD-24_slot8",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "stacking_m1_m2",
        "n_train": int(m_val.sum()),
        "n_val_real": int(m_val.sum()),
        "train_time_sec": round(train_time, 2),
        "weights": {"w_m1": w_m1, "w_m2": w_m2, "intercept": intercept},
        "eval_val_real": {"eval_pr_auc": val_pr_auc},
        "eval_test_real_pr_auc": test_real_pr_auc,
        "eval_gold_pr_auc": gold_pr_auc,
        "seed": args.seed,
    }
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Мета прогона: {meta_path}")

    # 6. Запись строки в reports/runs.csv
    if args.runs_csv:
        row = {
            "timestamp": meta["built_at"],
            "output_dir": str(out_dir),
            "model_name": "stacking_m1_m2",
            "max_len": 1024,
            "seed": args.seed,
            "epochs": 0.0,
            "train_time_sec": round(train_time, 1),
            "eval_pr_auc": val_pr_auc,
            "n_train": int(m_val.sum()),
            "n_val_real": int(m_val.sum()),
            "train_mix": "val_preds",
            "drop_diff": False,
        }
        append_runs_csv(args.runs_csv, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
