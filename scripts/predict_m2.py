"""FD-21 шаг 5: инференс M2 для всех сплитов -> preds CSV для scripts/evaluate.py.

    python scripts/predict_m2.py --checkpoint checkpoints/m2_run1/best \\
        --dataset data/processed/dataset_v1.parquet --out reports/preds/m2.csv

Выход: pair_id, p_stale, type_pred (сырой argmax головы дрейфа),
type_pred_gated (R10: p_stale > tau, иначе "none"). Оценка macro-F1
скриптом evaluate.py идёт по type_pred (качество головы); гейтированный
столбец — для реестра/эксплуатации.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

from src.models.m2_data import DRIFT_CLASSES, encode_pair
from src.models.m2_model import DualHeadCodeClassifier

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-21 шаг 5: инференс M2.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--out", default="reports/preds/m2.csv")
    ap.add_argument("--config", default="config/train.yaml")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="гейт R10: type_pred_gated = type_pred если p_stale > tau иначе 'none'")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    df = pd.read_parquet(args.dataset)
    print(f"Датасет: {args.dataset} ({len(df)} строк)")

    cfg = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        import yaml as _yaml
        cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    model_name = cfg.get("model_name", "microsoft/unixcoder-base")
    max_len = int(cfg.get("max_len", 1024))
    doc_budget = int(cfg.get("doc_budget", 192))

    # checkpoint — директория Trainer (best/): model.safetensors (полный DualHead),
    # config.json (конфиг бэкбона, пишет train_m2.py), dualhead_config.json,
    # tokenizer.json/vocab.json. Офлайн: всё берётся из ckpt без HF Hub.
    dual_cfg = {}
    dual_cfg_path = ckpt / "dualhead_config.json"
    if dual_cfg_path.exists():
        dual_cfg = json.loads(dual_cfg_path.read_text(encoding="utf-8"))
        model_name = dual_cfg.get("model_name", model_name)
        max_len = int(dual_cfg.get("max_len", max_len))
        doc_budget = int(dual_cfg.get("doc_budget", doc_budget))
    tok_src = str(ckpt) if (ckpt / "tokenizer.json").exists() or \
        (ckpt / "vocab.json").exists() else model_name
    try:
        tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True,
                                            trust_remote_code=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True,
                                            trust_remote_code=False)
    from transformers import AutoConfig
    if (ckpt / "config.json").exists():
        # config.json в ckpt — конфиг бэкбона (сохранён train_m2.py)
        backbone_cfg = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=False)
        backbone = AutoModel.from_config(backbone_cfg, trust_remote_code=False)
    else:
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=False)
    # если чекпоинт — полный DualHead (pytorch_model.bin из save_model),
    # backbone уже внутри; иначе веса backbone из HF
    model = DualHeadCodeClassifier(backbone, hidden_size=backbone.config.hidden_size,
                                   num_drift_classes=len(DRIFT_CLASSES))
    state_path = ckpt / "pytorch_model.bin"
    alt_path = ckpt / "model.safetensors"
    if state_path.exists():
        state = torch.load(str(state_path), map_location="cpu")
        # Trainer save_model сохраняет полный DualHead -> прямая загрузка
        try:
            model.load_state_dict(state, strict=True)
            print(f"Веса DualHead загружены: {state_path}")
        except RuntimeError as e:
            print(f"ВНИМАНИЕ: strict загрузка не удалась ({e}); "
                  f"используются веса backbone")
    elif alt_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(alt_path))
        try:
            model.load_state_dict(state, strict=True)
            print(f"Веса DualHead загружены: {alt_path}")
        except RuntimeError as e:
            print(f"ВНИМАНИЕ: strict загрузка не удалась ({e})")
    else:
        print(f"ВНИМАНИЕ: файл весов не найден в {ckpt} "
              f"(pytorch_model.bin/model.safetensors) — инференс на backbone-init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Устройство: {device}, max_len={max_len}")

    p_all, t_all = [], []
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            sub = df.iloc[start:start + args.batch_size]
            batch_ids, batch_masks = [], []
            for _, r in sub.iterrows():
                e = encode_pair(tok, str(r["sig_old"]), str(r["diff_code"]),
                                str(r["doc_old"]), max_len=max_len,
                                doc_budget=doc_budget)
                batch_ids.append(e["input_ids"])
                batch_masks.append(e["attention_mask"])
            max_n = max(len(x) for x in batch_ids)
            pad = tok.pad_token_id
            if pad is None:
                pad = 0
            ids = torch.tensor([x + [pad] * (max_n - len(x)) for x in batch_ids],
                               dtype=torch.long).to(device)
            masks = torch.tensor([x + [0] * (max_n - len(x)) for x in batch_masks],
                                 dtype=torch.long).to(device)
            out = model(input_ids=ids, attention_mask=masks)
            p = torch.sigmoid(out["logits_stale"]).cpu().numpy()
            t = out["logits_drift"].argmax(dim=-1).cpu().numpy()
            p_all.extend([float(v) for v in p])
            t_all.extend([DRIFT_CLASSES[int(v)] for v in t])
            if (start // args.batch_size) % 50 == 0:
                print(f"  {start + len(sub)}/{len(df)}")
    tau = float(args.threshold)
    gated = [t if p > tau else "none" for p, t in zip(p_all, t_all)]
    out_df = pd.DataFrame({"pair_id": df["pair_id"].astype(str),
                           "p_stale": p_all, "type_pred": t_all,
                           "type_pred_gated": gated})
    print(f"Гейт R10 tau={tau}: gated none = {sum(g == 'none' for g in gated)}/{len(gated)}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Предсказания: {out_path} ({len(out_df)} строк)")
    print(f"Далее: python scripts/evaluate.py --pred M0=reports/preds/m0.csv "
          f"--pred M1=reports/preds/m1.csv --pred M2={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
