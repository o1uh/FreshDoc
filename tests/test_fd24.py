"""FD-24: тесты подготовки запусков в Kaggle, смесей E2, абляции E3, стэкинга и выбора лучшего конфига."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.finalize_best_config import main as finalize_main
from scripts.train_m2 import append_runs_csv, filter_dataset_by_mix
from scripts.train_stacking import main as stacking_main
from src.data.dataset import REAL_SOURCES


def test_append_runs_csv_conforms_to_fd24_header(tmp_path):
    csv_file = tmp_path / "runs.csv"
    # Файл с заголовком FD-24
    header = ["timestamp", "output_dir", "model_name", "max_len", "seed",
              "epochs", "train_time_sec", "eval_pr_auc", "n_train", "n_val_real",
              "train_mix", "drop_diff"]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

    row = {
        "timestamp": "t1",
        "output_dir": "dir1",
        "model_name": "m1",
        "max_len": 1024,
        "seed": 42,
        "epochs": 3.0,
        "train_time_sec": 100.0,
        "eval_pr_auc": 0.95,
        "n_train": 2000,
        "n_val_real": 500,
        "train_mix": "real_only",
        "drop_diff": True,
        "unknown_extra": "should_be_dropped",
    }
    append_runs_csv(csv_file, row)

    with csv_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert "unknown_extra" not in reader[0]
        assert reader[0]["train_mix"] == "real_only"
        assert reader[0]["drop_diff"] == "True"
        assert float(reader[0]["eval_pr_auc"]) == 0.95


def test_train_mix_filtering_logic():
    # Проверка вызова реальной функции filter_dataset_by_mix из scripts/train_m2.py
    df = pd.DataFrame({
        "pair_id": [f"id_{i}" for i in range(5)],
        "source": ["codocbench", "history_neg", "synth_ast", "synth_llm", "codocbench"],
        "split": ["train", "train", "train", "train", "val"],
        "label_stale": [1, 0, 1, 1, 1],
    })

    # 1. real_only
    df_real, val_real = filter_dataset_by_mix(df, "real_only")
    assert set(df_real["source"]) == {"codocbench", "history_neg"}
    assert len(df_real) == 2
    assert len(val_real) == 1

    # 2. synth_only
    df_synth, _ = filter_dataset_by_mix(df, "synth_only")
    assert set(df_synth["source"]) == {"synth_ast"}
    assert len(df_synth) == 1

    # 3. real_ast
    df_real_ast, _ = filter_dataset_by_mix(df, "real_ast")
    assert set(df_real_ast["source"]) == {"codocbench", "history_neg", "synth_ast"}
    assert len(df_real_ast) == 3

    # 4. full
    df_full, _ = filter_dataset_by_mix(df, "full")
    assert len(df_full) == 4


def test_drop_diff_ablation_logic():
    train_df = pd.DataFrame({"diff_code": ["+ x = 1", "- y = 2"]})
    val_df = pd.DataFrame({"diff_code": ["+ z = 3"]})

    drop_diff = True
    if drop_diff:
        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["diff_code"] = ""
        val_df["diff_code"] = ""

    assert all(x == "" for x in train_df["diff_code"])
    assert all(x == "" for x in val_df["diff_code"])


def test_train_stacking_script_flow(tmp_path, monkeypatch):
    # Генерация синтетического датасета для проверки сквозного флоу стэкинга
    n = 100
    df = pd.DataFrame({
        "pair_id": [f"pair_{i}" for i in range(n)],
        "split": ["val"] * 40 + ["test_real"] * 40 + ["gold"] * 20,
        "source": ["codocbench"] * 20 + ["history_neg"] * 20 + ["codocbench"] * 40 + ["gold"] * 20,
        "label_stale": [i % 2 for i in range(n)],
        "weight": [1.0] * n,
    })
    ds_path = tmp_path / "dataset.parquet"
    df.to_parquet(ds_path)

    # Синтетические прогнозы M1 и M2
    rng = np.random.default_rng(42)
    p_m1 = rng.uniform(0.1, 0.9, size=n)
    p_m2 = rng.uniform(0.1, 0.9, size=n)
    m1_path = tmp_path / "m1.csv"
    m2_path = tmp_path / "m2.csv"
    pd.DataFrame({"pair_id": df["pair_id"], "p_stale": p_m1}).to_csv(m1_path, index=False)
    pd.DataFrame({"pair_id": df["pair_id"], "p_stale": p_m2, "type_pred": ["param_added"] * n}).to_csv(m2_path, index=False)

    out_dir = tmp_path / "stacking_out"
    runs_csv = tmp_path / "runs.csv"
    out_preds = tmp_path / "m_stack.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_stacking.py",
            "--dataset", str(ds_path),
            "--m1_preds", str(m1_path),
            "--m2_preds", str(m2_path),
            "--output_dir", str(out_dir),
            "--runs_csv", str(runs_csv),
            "--out_preds", str(out_preds),
            "--seed", "42",
        ]
    )

    rc = stacking_main()
    assert rc == 0
    assert (out_dir / "stacking_model.joblib").exists()
    assert (out_dir / "run_meta.json").exists()
    assert out_preds.exists()
    assert runs_csv.exists()

    with runs_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["model_name"] == "stacking_m1_m2"
        assert float(rows[0]["eval_pr_auc"]) > 0.0


def test_finalize_best_config(tmp_path, monkeypatch):
    runs_csv = tmp_path / "runs.csv"
    rows = [
        {"timestamp": "2026-10-21T01:00:00", "output_dir": "ckpt_slot1", "model_name": "microsoft/unixcoder-base",
         "max_len": "1024", "seed": "42", "epochs": "3.0", "train_time_sec": "1200", "eval_pr_auc": "0.91000",
         "n_train": "2393", "n_val_real": "559", "train_mix": "real_only", "drop_diff": "False"},
        {"timestamp": "2026-10-21T02:00:00", "output_dir": "ckpt_slot6", "model_name": "Alibaba-NLP/gte-modernbert-base",
         "max_len": "2048", "seed": "42", "epochs": "3.0", "train_time_sec": "2400", "eval_pr_auc": "0.98500",
         "n_train": "21997", "n_val_real": "559", "train_mix": "full", "drop_diff": "False"},
        {"timestamp": "2026-10-21T03:00:00", "output_dir": "ckpt_slot8", "model_name": "stacking_m1_m2",
         "max_len": "1024", "seed": "42", "epochs": "0.0", "train_time_sec": "10", "eval_pr_auc": "0.99900",
         "n_train": "559", "n_val_real": "559", "train_mix": "val_preds", "drop_diff": "False"},
        {"timestamp": "2026-10-21T04:00:00", "output_dir": "ckpt_slot9", "model_name": "Alibaba-NLP/gte-modernbert-base",
         "max_len": "2048", "seed": "42", "epochs": "3.0", "train_time_sec": "2100", "eval_pr_auc": "0.85000",
         "n_train": "21997", "n_val_real": "559", "train_mix": "full", "drop_diff": "True"},
    ]
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base_cfg = tmp_path / "base_train.yaml"
    base_cfg.write_text(yaml.dump({"learning_rate": 2e-5, "num_train_epochs": 3, "fp16": True}), encoding="utf-8")
    out_cfg = tmp_path / "train_best.yaml"

    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_best_config.py",
            "--runs_csv", str(runs_csv),
            "--base_config", str(base_cfg),
            "--out_config", str(out_cfg),
        ]
    )

    rc = finalize_main()
    assert rc == 0
    assert out_cfg.exists()

    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
    # Победитель среди нейросетей — gte-modernbert-base (PR-AUC 0.985, stacking исключен)
    assert cfg["model_name"] == "Alibaba-NLP/gte-modernbert-base"
    assert cfg["max_len"] == 2048
    assert cfg["train_mix"] == "full"
    assert cfg["drop_diff"] is False
    assert cfg["gradient_checkpointing"] is True


def test_finalize_ignores_ablation_even_if_higher_score(tmp_path, monkeypatch):
    runs_csv = tmp_path / "runs.csv"
    rows = [
        {"timestamp": "2026-10-21T01:00:00", "output_dir": "ckpt_slot5", "model_name": "answerdotai/ModernBERT-base",
         "max_len": "1024", "seed": "42", "epochs": "3.0", "train_time_sec": "1200", "eval_pr_auc": "0.98000",
         "n_train": "21997", "n_val_real": "559", "train_mix": "full", "drop_diff": "False"},
        {"timestamp": "2026-10-21T04:00:00", "output_dir": "ckpt_slot9", "model_name": "answerdotai/ModernBERT-base",
         "max_len": "1024", "seed": "42", "epochs": "3.0", "train_time_sec": "1100", "eval_pr_auc": "0.99500",
         "n_train": "21997", "n_val_real": "559", "train_mix": "full", "drop_diff": "True"},
    ]
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base_cfg = tmp_path / "base_train.yaml"
    base_cfg.write_text(yaml.dump({"learning_rate": 2e-5}), encoding="utf-8")
    out_cfg = tmp_path / "train_best.yaml"

    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_best_config.py",
            "--runs_csv", str(runs_csv),
            "--base_config", str(base_cfg),
            "--out_config", str(out_cfg),
        ]
    )

    rc = finalize_main()
    assert rc == 0
    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
    assert cfg["drop_diff"] is False
    assert cfg["model_name"] == "answerdotai/ModernBERT-base"


def test_finalize_ignores_legacy_without_artifacts(tmp_path, monkeypatch):
    runs_csv = tmp_path / "runs.csv"
    real_dir = tmp_path / "slot1_real"
    real_dir.mkdir(parents=True)
    (real_dir / "best").mkdir(parents=True)

    rows = [
        {"timestamp": "2026-10-21T00:00:00", "output_dir": "/kaggle/working/checkpoints/m2_run1",
         "model_name": "microsoft/unixcoder-base", "max_len": "1024", "seed": "42", "epochs": "3.0",
         "train_time_sec": "1700", "eval_pr_auc": "0.99900", "n_train": "21997", "n_val_real": "559",
         "train_mix": "full", "drop_diff": "False"},
        {"timestamp": "2026-10-21T01:00:00", "output_dir": str(real_dir),
         "model_name": "answerdotai/ModernBERT-base", "max_len": "1024", "seed": "42", "epochs": "3.0",
         "train_time_sec": "1200", "eval_pr_auc": "0.98000", "n_train": "21997", "n_val_real": "559",
         "train_mix": "full", "drop_diff": "False"},
    ]
    with runs_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base_cfg = tmp_path / "base_train.yaml"
    base_cfg.write_text(yaml.dump({"learning_rate": 2e-5}), encoding="utf-8")
    out_cfg = tmp_path / "train_best.yaml"

    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_best_config.py",
            "--runs_csv", str(runs_csv),
            "--base_config", str(base_cfg),
            "--out_config", str(out_cfg),
        ]
    )

    rc = finalize_main()
    assert rc == 0
    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8"))
    assert cfg["model_name"] == "answerdotai/ModernBERT-base"
