# FD-21: микрогайд запуска M2 v1 в Kaggle (2×T4)

> Время: пилот ~5 мин, overfit ~10 мин, полный прогон ~20–30 мин (3 эпохи @1024).
> Окружение: GPU T4 x2, Internet ON. Датасет: `freshdoc-dataset/dataset_v1.parquet` (27203 строк).

## 0. Что должно лежать в Kaggle Input

| Датасет | Файлы | Проверка |
|---|---|---|
| `freshdoc-dataset` (read-only) | `dataset_v1.parquet` (27203), `dataset_v1.manifest.json` | хэши входов — см. `reports/dataset_v1.manifest.json`; сплиты `train 21997 / val 559 / test_real 2895 / test_synth 1552 / gold 200` |
| `freshdoc-interim` (опц.) | `pairs_codocbench`, `pairs_negatives`, `synth_ast_v2`, `synth_llm_main700`, `project_pools` + gold-файлы | только если пересобираете dataset_v1; для FD-21 не нужен |

Локальная сверка перед загрузкой (уже зелёная 20.10.2026):

```bash
.venv/Scripts/python -c "import hashlib,json; ..."
# все 7 входов OK (см. docs/data_notes.md §FD-21)
.venv/Scripts/python scripts/validate_dataset.py  # все проверки зелёные
```

Если в Kaggle `dataset_v1.parquet` даёт другое число строк — **стоп, не обучать**,
сверить хэш с `reports/dataset_v1.manifest.json` и перезалить.

## 1. Ноутбук за 6 клеток

Клетка 1 — клон и зависимости:

```bash
%cd /kaggle/working
!git clone https://github.com/o1uh/FreshDoc.git
%cd /kaggle/working/FreshDoc
!git checkout feat/FD-21-m2-train
!pip install -q -r requirements.txt
!export NCCL_P2P_DISABLE=1
```

Клетка 2 — веса и токенизатор (smoke, ~1 мин, FD-02):

```python
import torch
from transformers import AutoModel, AutoTokenizer
for m in ["microsoft/unixcoder-base"]:
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModel.from_pretrained(m).to("cuda")
    out = mod(**tok("def test(): pass", return_tensors="pt").to("cuda"))
    print(m, tuple(out.last_hidden_state.shape))
```

Клетка 3 — пилот скорости (100 шагов, калибровка с/шаг):

```bash
!python scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_pilot --pilot_steps 100
# записать с/шаг -> пересчитать время эпохи: 688 шагов * с/шаг (ожидание 0.7-1.0 с/шаг)
```

Клетка 4 — тест на переобучение (шаг 2 FD-21, критерий приёмки):

```bash
!python scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_overfit --overfit_test
# ожидание: loss -> ~0, PR-AUC на тех же 200 -> ~1. Иначе стоп (маска/метки/обрезка/веса).
```

Клетка 5 — полный прогон (шаг 3 FD-21, DDP 2×T4):

```bash
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_run1
# fallback без DDP: python scripts/train_m2.py ... (медленнее ~1.6x)
```

Клетка 6 — усечение + предсказания + оценка (шаги 4–5 FD-21):

```bash
!python scripts/check_truncation.py --model microsoft/unixcoder-base --max_len 1024 \
  --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out /kaggle/working/m2_truncation.json
!python scripts/predict_m2.py --checkpoint /kaggle/working/checkpoints/m2_run1/best \
  --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out /kaggle/working/m2.csv
!python scripts/evaluate.py --pred M0=reports/preds/m0.csv --pred M1=reports/preds/m1.csv \
  --pred M2=/kaggle/working/m2.csv --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out_metrics /kaggle/working/metrics_m2.json --out_table /kaggle/working/e1_m2.csv
```

## 2. Что забрать из Output

- `checkpoints/m2_run1/best/` (веса + токенизатор), `run_meta.json` (время, усечение, eval val_real)
- `m2_truncation.json` (фактические доли усечения по сплитам — в `docs/data_notes.md`)
- `m2.csv` (pair_id, p_stale, type_pred), `metrics_m2.json`, `e1_m2.csv`
- Скопировать `run_meta.json` + `m2_truncation.json` в `reports/` репозитория (PR без весов).

## 3. Частые сбои

| Симптом | Причина | Что делать |
|---|---|---|
| DDP висит на старте | PCIe без NVLink | `export NCCL_P2P_DISABLE=1` перед запуском |
| loss NaN в FP16 (ModernBERT, не UniXcoder) | GeGLU/SDPA в FP16 | снизить lr или отключить unpadding; для FD-21 (UniXcoder) не ожидается |
| CUDA OOM | батч велик | снизить `per_device_batch_size` 8→4 в `config/train.yaml` |
| PR-AUC val_real ~0.5 | перепутаны сплиты/веса | проверить `split` и `weight`, выбор эпохи только по val_real |
| `dataset_v1.parquet` на 27143 строк | старый файл без gold-патча | перезалить актуальный (27203) из `data/processed/` |

## 4. Актуальность на 20.10.2026 (сверено)

- HF-веса доступны, лицензии Apache-2.0: `microsoft/unixcoder-base`,
  `answerdotai/ModernBERT-base` (нужен `transformers>=4.48.0`),
  `Alibaba-NLP/gte-modernbert-base`, `ibm-granite/granite-embedding-small-english-r2`.
- Локальное усечение @1024 (unixcoder): train 1.75%, val 3.76%, test_real 1.69%,
  test_synth 2.64%, gold 2.50% — ниже оценки 4.4% из workflow §3.
- `reports/data_stats.md` устарел (27143 строк, без gold-патча);
  канон — `reports/dataset_v1.manifest.json` (27203) + `data/processed/dataset_v1.parquet`.
