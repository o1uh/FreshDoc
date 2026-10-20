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

## 1. Ноутбук за 7 клеток

Клетка 1 — клон и зависимости:

```bash
%cd /kaggle/working
```
```python
# Приватный репозиторий (до релиза): токен из Add-ons -> Secrets -> GITHUB_TOKEN.
# Без токена клон упадёт на аутентификации; публичный URL — только запасной вариант.
import os
from kaggle_secrets import UserSecretsClient
try:
    _tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
    _repo = f"https://{_tok}@github.com/o1uh/FreshDoc.git"
    print("clone: private (token)")
except Exception:
    _repo = "https://github.com/o1uh/FreshDoc.git"
    print("clone: public fallback (repo must be public)")
```
```bash
!if [ ! -d FreshDoc ]; then git clone --quiet {_repo} > /tmp/clone.log 2>&1 || (echo "CLONE FAILED:"; sed -E 's#(https?://)[^@]+@#\1***@#g' /tmp/clone.log); fi
%cd /kaggle/working/FreshDoc
!git checkout feat/FD-21-m2-train
!git pull --ff-only origin feat/FD-21-m2-train || echo "PULL SKIPPED: убедись, что код свежий"
!pip install -q -r requirements.txt
```
```python
# NCCL на PCIe без NVLink (!export в subshell не сохраняется между клетками)
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
```
```bash
!nvidia-smi
```

Клетка 2 — проверка датасета (стоп при несовпадении):

```python
import pandas as pd
_p = "/kaggle/input/freshdoc-dataset/dataset_v1.parquet"
_df = pd.read_parquet(_p, columns=["split"])
assert len(_df) == 27203, f"dataset rows {len(_df)} != 27203 (старый файл без gold-патча?)"
_exp = {"train": 21997, "val": 559, "test_real": 2895, "test_synth": 1552, "gold": 200}
_got = _df["split"].value_counts().to_dict()
assert _got == _exp, f"split mismatch: {_got}"
print("dataset OK:", _got)
del _df
```
# Дальше — только при пройденной проверке, иначе сверить хэш с `reports/dataset_v1.manifest.json` и перезалить датасет.

Клетка 3 — веса и токенизатор (smoke, ~1 мин, FD-02):

```python
import torch
from transformers import AutoModel, AutoTokenizer
for m in ["microsoft/unixcoder-base"]:
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModel.from_pretrained(m).to("cuda")
    out = mod(**tok("def test(): pass", return_tensors="pt").to("cuda"))
    print(m, tuple(out.last_hidden_state.shape))
del mod, out, tok
import gc; gc.collect()
torch.cuda.empty_cache()
print("GPU freed")
```

Клетка 4 — пилот скорости (100 шагов, калибровка с/шаг):

```bash
!python scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_pilot --pilot_steps 100
# записать с/шаг -> пересчитать время эпохи: 688 шагов * с/шаг (ожидание 0.7-1.0 с/шаг)
```

Клетка 5 — тест на переобучение (шаг 2 FD-21, критерий приёмки):

```bash
!python scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_overfit --overfit_test
# ожидание: loss -> ~0, PR-AUC на тех же 200 -> ~1. Иначе стоп (маска/метки/обрезка/веса).
```

Клетка 6 — полный прогон (шаг 3 FD-21, DDP 2×T4):

```bash
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --output_dir /kaggle/working/checkpoints/m2_run1
# fallback без DDP (1 GPU): та же команда через python + --accum 4
# (обязательно: иначе eff_batch 16 вместо 32 = per_device 8 x accum 2 x 1 GPU)
```

Клетка 7 — усечение + предсказания + оценка (шаги 4–5 FD-21):

```bash
!python scripts/check_truncation.py --model microsoft/unixcoder-base --max_len 1024 \
  --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out /kaggle/working/m2_truncation.json
!python scripts/predict_m2.py --checkpoint /kaggle/working/checkpoints/m2_run1/best \
  --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out /kaggle/working/m2.csv
# reports/preds/ в .gitignore — после clone m0/m1.csv нет, регенерируем (~20 c, M1 ~1.2 c обучения)
!python scripts/baselines_m0_m1.py --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --out_m0 /kaggle/working/m0.csv --out_m1 /kaggle/working/m1.csv \
  --out_model /kaggle/working/m1_hgb.joblib --out_report /kaggle/working/m1_report.json
!python scripts/evaluate.py --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
  --pred M0=/kaggle/working/m0.csv --pred M1=/kaggle/working/m1.csv --pred M2=/kaggle/working/m2.csv \
  --out_metrics /kaggle/working/metrics_m2.json --out_table /kaggle/working/e1_m2.csv
# Альтернатива: положить m0.csv/m1.csv в Kaggle-датасет freshdoc-dataset (тогда regen не нужен).
# M0/M1 не должны пересчитываться на каждом прогоне без нужды — regen выше детерминирован (seed 42).
```

## 2. Что забрать из Output

- `checkpoints/m2_run1/best/` (веса + токенизатор), `run_meta.json` (время, усечение, eval val_real, состав train, история PR-AUC по эпохам)
- `m2_truncation.json` (фактические доли усечения по сплитам — в `docs/data_notes.md`)
- `m2.csv` (pair_id, p_stale, type_pred), `metrics_m2.json`, `e1_m2.csv`
- `reports/runs.csv` (строка прогона дописывается автоматически; формат принадлежит FD-24)
- `overfit_verdict.json` из overfit-прогона (артефакт приёма шага 2: pass/fail + PR-AUC)
- Скопировать `run_meta.json` + `m2_truncation.json` в `reports/` репозитория (PR без весов).

## 3. Частые сбои

| Симптом | Причина | Что делать |
|---|---|---|
| DDP висит на старте | PCIe без NVLink | `%env NCCL_P2P_DISABLE=1` или `os.environ["NCCL_P2P_DISABLE"]="1"` перед запуском |
| loss NaN в FP16 (ModernBERT, не UniXcoder) | GeGLU/SDPA в FP16 | снизить lr или отключить unpadding; для FD-21 (UniXcoder) не ожидается |
| CUDA OOM | батч велик | снизить `per_device_batch_size` 8→4 в `config/train.yaml` |
| PR-AUC val_real ~0.5 | перепутаны сплиты/веса | проверить `split` и `weight`, выбор эпохи только по val_real |
| `dataset_v1.parquet` на 27143 строк | старый файл без gold-патча | перезалить актуальный (27203) из `data/processed/` |

## 4. Актуальность на 20.10.2026 (сверено)

- HF-веса доступны, лицензии Apache-2.0: `microsoft/unixcoder-base`,
  `answerdotai/ModernBERT-base` (нужен `transformers>=4.48.0`),
  `Alibaba-NLP/gte-modernbert-base`, `ibm-granite/granite-embedding-small-english-r2`.
- Локальное усечение @1024 (unixcoder, doc_budget 384 — отклонение от R10/192,
решение команды 20.10.2026): сторона A — train 2.34%, val 5.55%, test_real 2.00%,
test_synth 3.09%, gold 3.00%; сторона B (docstring) — train 9.48%, val 9.48%,
test_real 4.11%, test_synth 10.50%, gold 4.00%. Хвост docstring режется у ~1/10 строк
(было ~1/5 при капе 192) — учесть в записке (кап по решению команды, влияние не измерено).
- `reports/data_stats.md` актуален (27203 строк, с gold-патчем; регенерируется
скриптом `scripts/refresh_data_stats.py`, свежесть стережет `tests/test_data_stats.py`);
канон — `reports/dataset_v1.manifest.json` + `data/processed/dataset_v1.parquet`.
