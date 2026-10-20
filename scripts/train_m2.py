"""FD-21: обучение M2 DualHead (R10, rev.8).

    python scripts/train_m2.py --config config/train.yaml \\
        --data_path data/processed/dataset_v1.parquet \\
        --output_dir checkpoints/m2_run1

    Полный запуск (Kaggle 2xT4, DDP): torchrun --nproc_per_node=2 scripts/train_m2.py ...
    или accelerate launch. Одно-GPU: python scripts/train_m2.py ...

    Режимы отладки:
    - --overfit_test : 200 строк train x 20 эпох, потери -> 0, PR-AUC -> ~1 (шаг 2 FD-21)
    - --pilot_steps 100 : 100 шагов для калибровки с/шаг (аналог FD-05)

    Выбор эпохи — по PR-AUC на реальных строках val (шаг 3 FD-21).
    Предсказания для всех сплитов — scripts/predict_m2.py + scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoTokenizer, Trainer,
                          TrainingArguments)

from src.data.dataset import REAL_SOURCES
from src.models.m2_data import (DOC_BUDGET_TOKENS, DRIFT_CLASSES,
                                drift_to_id, encode_pair)
from src.models.m2_model import (LAMBDA_DRIFT, DualHeadCodeClassifier,
                                 check_input_capacity)

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_CONFIG = PROJECT_ROOT / "config" / "train.yaml"


class M2Dataset(Dataset):
    """Токенизированные пары R10 (пады — в collator)."""

    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int,
                 doc_budget: int = DOC_BUDGET_TOKENS):
        self.tok = tokenizer
        self.max_len = max_len
        self.doc_budget = doc_budget
        self.pair_ids = frame["pair_id"].astype(str).tolist()
        sigs = frame["sig_old"].fillna("").astype(str).tolist()
        diffs = frame["diff_code"].fillna("").astype(str).tolist()
        docs = frame["doc_old"].fillna("").astype(str).tolist()
        self.stale = frame["label_stale"].to_numpy(dtype=np.int64)
        self.weight = frame["weight"].to_numpy(dtype=np.float32) \
            if "weight" in frame.columns else np.ones(len(frame), dtype=np.float32)
        drifts = frame["drift_type"].astype(str).tolist() \
            if "drift_type" in frame.columns else ["unknown"] * len(frame)
        self.drift = np.array([drift_to_id(d) for d in drifts], dtype=np.int64)
        self.encs: list[dict] = []
        n_trunc = n_doc_trunc = 0
        for s, d, doc in zip(sigs, diffs, docs):
            e = encode_pair(tokenizer, s, d, doc, max_len=max_len,
                            doc_budget=doc_budget)
            n_trunc += int(e["truncated"])
            n_doc_trunc += int(e["doc_truncated"])
            self.encs.append(e)
        self.n_truncated = n_trunc
        self.n_doc_truncated = n_doc_trunc

    def __len__(self):
        return len(self.pair_ids)

    def __getitem__(self, i: int) -> dict:
        e = self.encs[i]
        return {
            "input_ids": torch.tensor(e["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(e["attention_mask"], dtype=torch.long),
            "labels_stale": torch.tensor(int(self.stale[i]), dtype=torch.long),
            "labels_drift": torch.tensor(int(self.drift[i]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.weight[i]), dtype=torch.float),
        }


class M2DataCollator:
    """Паддинг батча R10 (верхнеуровневый класс — picklable для num_workers>0).

    Замыкание m2_collator падало в DataLoader-воркерах:
    AttributeError: Can't pickle local object 'm2_collator.<locals>._collate'.
    Храним только pad_id (int), сам токенизатор не pickle'им.
    """

    def __init__(self, tokenizer):
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_n = max(len(b["input_ids"]) for b in batch)
        ids, masks = [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = max_n - n
            ids.append(torch.cat([b["input_ids"],
                                  torch.full((pad,), self.pad_id, dtype=torch.long)]))
            masks.append(torch.cat([b["attention_mask"],
                                    torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels_stale": torch.stack([b["labels_stale"] for b in batch]),
            "labels_drift": torch.stack([b["labels_drift"] for b in batch]),
            "sample_weight": torch.stack([b["sample_weight"] for b in batch]),
        }


def m2_collator(tokenizer):
    """Совместимость: фабрика поверх picklable M2DataCollator."""
    return M2DataCollator(tokenizer)


class M2Trainer(Trainer):
    """Trainer с маскированным условным лоссом R10."""

    def __init__(self, *args, lambda_drift: float = LAMBDA_DRIFT, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_drift = lambda_drift

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_stale = inputs.pop("labels_stale")
        labels_drift = inputs.pop("labels_drift", None)
        sample_weight = inputs.pop("sample_weight", None)
        # Trainer может подсунуть labels (не наши) — убрать
        inputs.pop("labels", None)
        out = model(input_ids=inputs.get("input_ids"),
                    attention_mask=inputs.get("attention_mask"),
                    labels_stale=labels_stale,
                    labels_drift=labels_drift,
                    sample_weight=sample_weight,
                    lambda_drift=self.lambda_drift)
        loss = out["loss"]
        if return_outputs:
            return loss, {"logits_stale": out["logits_stale"],
                          "logits_drift": out["logits_drift"]}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Eval: вернуть stale-логиты как `logits` для compute_metrics."""
        model.eval()
        with torch.no_grad():
            labels_stale = inputs.get("labels_stale")
            has_labels = labels_stale is not None
            out = model(input_ids=inputs.get("input_ids"),
                        attention_mask=inputs.get("attention_mask"),
                        labels_stale=labels_stale if has_labels else None,
                        labels_drift=inputs.get("labels_drift") if has_labels else None,
                        sample_weight=None)
            logits = out["logits_stale"].detach()
            loss = out["loss"].detach() if out["loss"] is not None else None
            labels = labels_stale.detach() if has_labels else None
        return loss, logits, labels


def compute_metrics_pr_auc(eval_pred) -> dict:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    y = np.asarray(labels).astype(int).ravel()
    s = torch.sigmoid(torch.as_tensor(np.asarray(logits).ravel())).numpy()
    if len(np.unique(y)) < 2:
        return {"pr_auc": float("nan")}
    return {"pr_auc": float(average_precision_score(y, s))}


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def append_runs_csv(path, row: dict) -> None:
    """Дописать строку прогона (формат принадлежит FD-24, писатель уже в FD-21).

    Если файл существует (заголовок FD-24), строка конформится под
    существующий заголовок: лишние поля отбрасываются, недостающие -> "".
    Иначе заголовок пишется из строки. Так CSV не разъезжается по колонкам.
    Относительный путь резолвится от PROJECT_ROOT, а не от CWD.
    """
    import csv as _csv

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        with p.open("r", encoding="utf-8", newline="") as _f:
            header = _csv.DictReader(_f).fieldnames or []
        if not header:
            header = list(row)
        out_row = {k: row.get(k, "") for k in header}
        with p.open("a", encoding="utf-8", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=header)
            _w.writerow(out_row)
    else:
        with p.open("a", encoding="utf-8", newline="") as _f:
            _w = _csv.DictWriter(_f, fieldnames=list(row))
            _w.writeheader()
            _w.writerow(row)
    print(f"Строка runs.csv: {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-21: обучение M2 (R10).")
    ap.add_argument("--config", default=str(IN_CONFIG))
    ap.add_argument("--data_path", default=str(IN_DATASET))
    ap.add_argument("--output_dir", default="checkpoints/m2_run1")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--overfit_test", action="store_true",
                    help="шаг 2 FD-21: 200 строк x 20 эпох")
    ap.add_argument("--overfit_lr", type=float, default=5e-3,
                    help="lr протокола переобучения (не рабочего обучения). "
                         "Эмпирика прогона: lr=2e-5 за ~140 шагов не запоминает "
                         "200 строк (PR-AUC 0.63, verdict fail), lr=5e-3 "
                         "запоминает (PR-AUC 0.999, verdict pass). Высоким lr "
                         "проверяем способность+plumbing, не гиперпараметры.")
    ap.add_argument("--pilot_steps", type=int, default=0,
                    help="калибровка скорости: остановить после N шагов")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--per_device_batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--weight_overrides", default=None,
                    help="перевес обучения без пересборки датасета, формат "
                         "'label_source:вес,...' (напр. 'llm_p1:1.0'). "
                         "По умолчанию веса датасета (llm_p1=0.5 по решению "
                         "команды 13.10 после аудита 56-60%% шума; см. S3).")
    ap.add_argument("--runs_csv", default="reports/runs.csv",
                    help="куда дописать строку прогона (формат FD-24). "
                         "Пустая строка отключает запись.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    model_name = args.model_name or cfg.get("model_name", "microsoft/unixcoder-base")
    max_len = int(args.max_len or cfg.get("max_len", 1024))
    doc_budget = int(cfg.get("doc_budget", DOC_BUDGET_TOKENS))
    epochs = float(args.num_train_epochs if args.num_train_epochs is not None
                   else cfg.get("num_train_epochs", 3))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    lr = float(cfg.get("learning_rate", 2e-5))
    wd = float(cfg.get("weight_decay", 0.01))
    # transformers>=5: warmup_ratio удалён; warmup_steps<1 трактуется как доля.
    # Поддержаны оба ключа (старый warmup_ratio — для совместимости).
    warmup = float(cfg.get("warmup_steps", cfg.get("warmup_ratio", 0.06)))
    lambda_drift = float(cfg.get("lambda_drift", LAMBDA_DRIFT))
    per_device = int(args.per_device_batch or cfg.get("per_device_batch_size", 8))
    accum = int(args.accum or cfg.get("gradient_accumulation_steps", 2))
    fp16 = bool(cfg.get("fp16", True))

    df = pd.read_parquet(args.data_path)
    need = {"pair_id", "sig_old", "diff_code", "doc_old", "label_stale",
            "drift_type", "weight", "split", "source"}
    missing = need - set(df.columns)
    if missing:
        print(f"В датасете нет колонок: {sorted(missing)}")
        return 2

    m_train = df["split"].eq("train").to_numpy()
    m_val_real = df["split"].eq("val").to_numpy() & df["source"].isin(
        list(REAL_SOURCES)).to_numpy()
    train_df = df.loc[m_train].reset_index(drop=True)
    val_df = df.loc[m_val_real].reset_index(drop=True)
    print(f"Датасет: {args.data_path} ({len(df)} строк)")
    print(f"train: {len(train_df)} (pos={int(train_df['label_stale'].sum())}), "
          f"val_real: {len(val_df)} (pos={int(val_df['label_stale'].sum())})")

    # Перевес обучения (S3): меняет веса только в этом прогоне, датасет не трогает.
    # Дефолт — веса датасета (llm_p1=0.5 — решение команды 13.10 по аудиту шума).
    weight_overrides: dict[str, float] = {}
    if args.weight_overrides:
        for item in str(args.weight_overrides).split(","):
            src, _, w = item.partition(":")
            src, w = src.strip(), w.strip()
            if not src or not w:
                print(f"ОШИБКА: плохой --weight_overrides: {item!r}")
                return 2
            weight_overrides[src] = float(w)
    if weight_overrides:
        unknown = sorted(set(weight_overrides) - set(train_df["label_source"].unique()))
        if unknown:
            print(f"ОШИБКА: неизвестные label_source в --weight_overrides: {unknown}")
            return 2
        train_df = train_df.copy()
        for src, w in weight_overrides.items():
            train_df.loc[train_df["label_source"] == src, "weight"] = w
        print(f"Перевес обучения: {weight_overrides}")

    if args.overfit_test:
        # Стратифицированная выборка 100 pos + 100 neg: первые 200 строк
        # train состоят из одних pos (head(200) даёт один класс -> PR-AUC nan
        # и срыв выбора лучшей эпохи). Перемешиваем сидом для детерминизма.
        pos = train_df.loc[train_df["label_stale"] == 1].head(100)
        neg = train_df.loc[train_df["label_stale"] == 0].head(100)
        train_df = pd.concat([pos, neg], ignore_index=True).sample(
            n=min(200, len(pos) + len(neg)), random_state=seed).reset_index(drop=True)
        val_df = train_df.copy()
        epochs = 20
        # Протокол переобучения ≠ рабочее обучение: цель — доказать, что
        # модель В ПРИНЦИПЕ запоминает 200 строк (ёмкость + plumbing).
        # Эмпирика: 2e-5 за ~140 шагов не запоминает (fail — и это нормально),
        # 5e-3 запоминает (PR-AUC 0.999). Высокий lr здесь не переносить в FD-24.
        if args.overfit_lr:
            lr = float(args.overfit_lr)
        print(f"OVERFIT-TEST: {len(train_df)} строк "
              f"(pos={int(train_df['label_stale'].sum())}, "
              f"neg={int((train_df['label_stale'] == 0).sum())}) x {epochs} эпох, "
              f"lr={lr} "
              f"(ожидание: loss -> 0, PR-AUC -> ~1)")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"Токенизатор {model_name}: cls={tok.cls_token} sep={tok.sep_token}")
    train_ds = M2Dataset(train_df, tok, max_len, doc_budget)
    val_ds = M2Dataset(val_df, tok, max_len, doc_budget)
    print(f"Усечение A (train): {train_ds.n_truncated}/{len(train_ds)} = "
          f"{train_ds.n_truncated / max(1, len(train_ds)):.4f}; "
          f"doc: {train_ds.n_doc_truncated}/{len(train_ds)} = "
          f"{train_ds.n_doc_truncated / max(1, len(train_ds)):.4f}")
    print(f"Усечение A (val_real): {val_ds.n_truncated}/{len(val_ds)} = "
          f"{val_ds.n_truncated / max(1, len(val_ds)):.4f}; "
          f"doc: {val_ds.n_doc_truncated}/{len(val_ds)} = "
          f"{val_ds.n_doc_truncated / max(1, len(val_ds)):.4f}")
    # Состав обучения для run_meta (S2): источник x метка + эффективные веса.
    _mix = train_df.groupby(["source", "label_stale"]).size()
    train_mix = {f"{s}/{int(l)}": {"n": int(n),
                 "frac": float(n / max(1, len(train_df)))}
                 for (s, l), n in _mix.items()}
    _wa = train_df.groupby(["label_source", "weight"]).size()
    weight_audit = {f"{s}/{float(w)}": int(n) for (s, w), n in _wa.items()}
    val_pos, val_n = int(val_df["label_stale"].sum()), len(val_df)
    print(f"train_mix: {train_mix}")
    print(f"weight_audit: {weight_audit}")

    from transformers import AutoConfig
    backbone = AutoModel.from_pretrained(model_name)
    hidden = backbone.config.hidden_size
    model = DualHeadCodeClassifier(backbone, hidden_size=hidden,
                                   num_drift_classes=len(DRIFT_CLASSES),
                                   dropout=float(cfg.get("dropout", 0.1)))
    print(f"Модель: {model_name} hidden={hidden} drift_classes={DRIFT_CLASSES}")
    try:
        check_input_capacity(backbone, max_len, model_name)
    except ValueError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    targs_kwargs = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=warmup,
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        optim=cfg.get("optim", "adamw_torch"),
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=int(cfg.get("logging_steps", 25)),
        report_to="none",
        fp16=fp16 and torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        # True: голова дрейфа условная (маска y=1 и известный тип) — на шагах
        # без валидных строк drift_head не участвует в backward; с False DDP
        # падает "Expected to have finished reduction...". Дополнительно
        # пустой лосс дрейфа держит граф (logit_d.sum()*0.0) в m2_model.py.
        ddp_find_unused_parameters=True,
    )
    if args.pilot_steps:
        targs_kwargs["max_steps"] = int(args.pilot_steps)
        print(f"PILOT: остановка после {args.pilot_steps} шагов")
    elif args.max_steps:
        targs_kwargs["max_steps"] = int(args.max_steps)
    targs = TrainingArguments(**targs_kwargs)

    trainer = M2Trainer(model=model, args=targs,
                        train_dataset=train_ds, eval_dataset=val_ds,
                        data_collator=M2DataCollator(tok),
                        compute_metrics=compute_metrics_pr_auc,
                        lambda_drift=lambda_drift)
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"Обучение: {train_time:.1f} c ({train_time / 60:.1f} мин)")

    def _json_safe(v):
        # run_meta обязан быть строгим JSON: NaN/inf -> None (minor 2).
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return None
        if isinstance(v, dict):
            return {k: _json_safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        return v

    metrics = trainer.evaluate()
    print(f"Eval val_real: {metrics}")
    # История PR-AUC по эпохам в артефакт (наблюдение прогона: гейт виден
    # только в финале; при провале нужно знать, на какой эпохе поехало).
    eval_history = [
        {"epoch": e.get("epoch"),
         "eval_pr_auc": e.get("eval_pr_auc"),
         "eval_loss": e.get("eval_loss")}
        for e in trainer.state.log_history
        if any(k.startswith("eval_") for k in e)
    ]
    is_main = trainer.is_world_process_zero()
    if is_main:
        best_dir = out_dir / "best"
        # trainer.save_model внутри защищён should_save (только rank 0),
        # остальные записи ниже — нет, поэтому весь блок под гардом (S5/2.1):
        # два процесса torchrun иначе пишут одни файлы конкурентно.
        trainer.save_model(str(best_dir))
        tok.save_pretrained(str(best_dir))
        # DualHead — обычный nn.Module без config.json: для офлайн-инференса
        # сохраняем конфиг бэкбона и мета DualHead рядом с model.safetensors.
        try:
            backbone.config.save_pretrained(str(best_dir))
            (best_dir / "dualhead_config.json").write_text(
                json.dumps({"model_name": model_name, "hidden_size": hidden,
                            "num_drift_classes": len(DRIFT_CLASSES),
                            "drift_classes": DRIFT_CLASSES,
                            "max_len": max_len, "doc_budget": doc_budget,
                            "backbone_config": "config.json (бэкбон)"},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Конфиг бэкбона + dualhead_config.json: {best_dir}")
        except Exception as e:
            print(f"ВНИМАНИЕ: не удалось сохранить конфиг бэкбона ({e})")

    meta = {
        "task": "FD-21",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "max_len": max_len,
        "doc_budget": doc_budget,
        "lambda_drift": lambda_drift,
        "lr": lr, "weight_decay": wd, "warmup_steps": warmup,
        "epochs": epochs, "seed": seed,
        "per_device_batch_size": per_device, "accum": accum,
        "eff_batch": per_device * accum * max(1, torch.cuda.device_count()),
        "n_train": int(len(train_ds)), "n_val_real": int(len(val_ds)),
        "train_truncated_frac": train_ds.n_truncated / max(1, len(train_ds)),
        "val_truncated_frac": val_ds.n_truncated / max(1, len(val_ds)),
        "train_doc_truncated_frac": train_ds.n_doc_truncated / max(1, len(train_ds)),
        "val_doc_truncated_frac": val_ds.n_doc_truncated / max(1, len(val_ds)),
        "train_mix": train_mix,
        "val_pos": val_pos, "val_n": val_n,
        "val_prevalence": float(val_pos / max(1, val_n)),
        "weight_audit": weight_audit,
        "weight_overrides": weight_overrides,
        "train_time_sec": round(train_time, 1),
        "eval_val_real": _json_safe({k: float(v) for k, v in metrics.items()
                                     if isinstance(v, (int, float))}),
        "eval_history": _json_safe(eval_history),
        "overfit_test": bool(args.overfit_test),
        "drift_classes": DRIFT_CLASSES,
    }
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Мета: {out_dir / 'run_meta.json'}")
        # Строка прогона для reports/runs.csv (формат FD-24; minor 6, N2).
        # Только полные прогоны (не пилот/overfit/max_steps).
        if args.runs_csv and not args.overfit_test and not args.pilot_steps \
                and not args.max_steps:
            _row = {"timestamp": meta["built_at"], "output_dir": str(out_dir),
                    "model_name": model_name, "max_len": max_len,
                    "seed": seed, "epochs": epochs,
                    "train_time_sec": meta["train_time_sec"],
                    "eval_pr_auc": (meta["eval_val_real"] or {}).get("eval_pr_auc"),
                    "n_train": meta["n_train"], "n_val_real": meta["n_val_real"]}
            append_runs_csv(args.runs_csv, _row)
    if args.overfit_test:
        # Машинный критерий шага 2 FD-21 (S4): PR-AUC>=0.99 на тех же 200.
        _p = (meta["eval_val_real"] or {}).get("eval_pr_auc")
        _pass = _p is not None and _p == _p and _p >= 0.99
        _verdict = {"overfit_pr_auc": _p, "pass": bool(_pass),
                    "threshold": 0.99, "n": int(len(train_ds))}
        if is_main:
            (out_dir / "overfit_verdict.json").write_text(
                json.dumps(_verdict, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OVERFIT verdict: {_verdict}")
        if not _pass:
            print("ПРОВАЛ критерия шага 2 FD-21: PR-AUC < 0.99", file=sys.stderr)
            return 3
    else:
        print("Далее: scripts/predict_m2.py + scripts/evaluate.py "
              "(шаг 5 FD-21). Выбор эпохи — по val_real PR-AUC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
