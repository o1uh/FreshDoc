# Инструкция по запуску обучения в GPU-окружении (Kaggle / Colab)

> Короткая версия для FD-21 — `docs/kaggle_fd21_guide.md` (6 клеток, пилот,
> overfit-тест, полный прогон, усечение, предсказания, оценка).
> Этот файл — полная версия с вариантами Colab и обменом промежуточными данными.

---

## Вариант 1. Запуск в Kaggle Notebooks (Рекомендуемый)

### 1. Настройка сессии
1. Создать блокнот: `+ Create` → `New Notebook`.
2. В панели настроек `Settings` и панели справа соответственно:
   - **Accelerator**: `GPU T4 x2`.
   - **Internet**: `On` (обязательно для скачивания пакетов и весов).

### 2. Клонирование репозитория и установка зависимостей
Выполнить в первой ячейке блокнота:

```bash
# 1. Переход в рабочую директорию
%cd /kaggle/working

# 2. Клонирование репозитория (для приватного репозитория использовать Personal Access Token)
# Токен можно безопасно добавить через Add-ons -> Secrets -> Name: GITHUB_TOKEN
import os
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("GITHUB_TOKEN")
    repo_url = f"https://{token}@github.com/o1uh/FreshDoc.git"
except Exception:
    # Запасной вариант при публичном репозитории или прямом вводе
    repo_url = "https://github.com/o1uh/FreshDoc.git"

!git clone {repo_url}
%cd /kaggle/working/FreshDoc

# 3. Переключение на целевую ветку экспериментов (при необходимости)
!git checkout feat/FD-21-m2-train
!git pull origin feat/FD-21-m2-train

# 4. Установка зависимостей проекта (нужны transformers>=4.48, accelerate>=0.34)
!pip install -q -r requirements.txt
!export NCCL_P2P_DISABLE=1  # DDP на PCIe без NVLink (иначе зависание allreduce)
```
### 2.1. Обмен промежуточными данными команды (FD-03 — FD-11)
Файловая система `/kaggle/working` изолирована для каждой сессии. Для передачи таблиц между задачами используется промежуточный приватный датасет Kaggle: `freshdoc-interim`.

#### 1. Права доступа
Создатель датасета добавляет всех участников команды через `Settings` -> `Sharing` с правами **Editor**.

#### 2. Список промежуточных артефактов в `freshdoc-interim`:
- `pairs_codocbench.parquet` (результат FD-04)
- `project_pools.parquet` (результат FD-04)
- `negatives_raw.parquet` (результат FD-05)
- `pairs_negatives.parquet` (результат FD-06)
- `functions_pool_v1.parquet` (результат FD-07)
- `synth_ast.parquet` (результат FD-08 / FD-09)
- `synth_llm.parquet` (результат FD-11)

#### 3. Использование в блокнотах:
- **Чтение:** подключить датасет через `+ Add Input` -> `freshdoc-interim`. Файлы доступны по пути:
  `/kaggle/input/freshdoc-interim/<имя_файла>.parquet`
- **Запись новой версии (через Kaggle API):**
  После генерации файла в `/kaggle/working/export/`:
  ```bash
  !kaggle datasets version -p /kaggle/working/export -m "Add <имя_файла> from FD-xx" --dir-mode zip
  ```
  *(Запасной вариант: скачать файл из панели Output и загрузить вручную через кнопку `New Version` на странице датасета).*


### 3. Подключение данных и пути к файлам
1. **Датасет:** Добавляется через интерфейс Kaggle: `+ Add Input` → выбор датасета `freshdoc-dataset` (формируется в задаче FD-12).
   - Путь к данным: `/kaggle/input/freshdoc-dataset/dataset_v1.parquet`
2. **Чекпоинты и логи:** Сохраняются строго в перезаписываемую директорию `/kaggle/working/`:
   - Чекпоинты: `/kaggle/working/FreshDoc/checkpoints/`
   - Отчеты: `/kaggle/working/FreshDoc/reports/`

### 4. Запуск скрипта обучения (FD-21: бейзлайн microsoft/unixcoder-base @1024)
Выполнить из директории `/kaggle/working/FreshDoc`:

```bash
# Пилот скорости 100 шагов (калибровка с/шаг, аналог FD-05)
!python scripts/train_m2.py \
    --config config/train.yaml \
    --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --output_dir /kaggle/working/checkpoints/m2_pilot --pilot_steps 100

# Тест на переобучение 200x20 (шаг 2 FD-21: loss -> 0, PR-AUC -> ~1)
!python scripts/train_m2.py \
    --config config/train.yaml \
    --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --output_dir /kaggle/working/checkpoints/m2_overfit --overfit_test

# Полный прогон 3 эпохи, 2xT4 DDP (шаг 3 FD-21; выбор эпохи по PR-AUC val_real)
!torchrun --nproc_per_node=2 scripts/train_m2.py \
    --config config/train.yaml \
    --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --output_dir /kaggle/working/checkpoints/m2_run1
```

### 5. Усечение, предсказания, оценка (шаги 4-5 FD-21)

```bash
!python scripts/check_truncation.py --model microsoft/unixcoder-base --max_len 1024 \
    --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --out /kaggle/working/m2_truncation.json
!python scripts/predict_m2.py --checkpoint /kaggle/working/checkpoints/m2_run1/best \
    --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --out /kaggle/working/m2.csv
!python scripts/evaluate.py --dataset /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --pred M0=reports/preds/m0.csv --pred M1=reports/preds/m1.csv --pred M2=/kaggle/working/m2.csv \
    --out_metrics /kaggle/working/metrics_m2.json --out_table /kaggle/working/e1_m2.csv
```

---

## Вариант 2. Запуск в Google Colab

### 1. Настройка сессии
1. Открыть Google Colab: `Файл` → `Создать блокнот`.
2. В меню: `Среда выполнения` → `Сменить среду выполнения` → выбрать **T4 GPU**.

### 2. Монтирование диска, клонирование и зависимости
Выполнить в ячейке блокнота:

```bash
# 1. Монтирование Google Drive для персистентного хранения весов
from google.colab import drive
drive.mount('/content/drive')

# Создание папки под чекпоинты на Диске, чтобы они не удалились после сброса сессии
!mkdir -p /content/drive/MyDrive/FreshDoc_Artifacts/checkpoints

# 2. Клонирование репозитория
# Токен задается через панель Secrets (значок ключа слева) -> GITHUB_TOKEN
from google.colab import userdata
try:
    token = userdata.get('GITHUB_TOKEN')
    repo_url = f"https://{token}@github.com/o1uh/FreshDoc.git"
except Exception:
    repo_url = "https://github.com/o1uh/FreshDoc.git"

%cd /content
!git clone {repo_url}
%cd /content/FreshDoc

# 3. Переключение ветки и установка пакетов
!git checkout feat/FD-21-m2-train
!pip install -q -r requirements.txt
!export NCCL_P2P_DISABLE=1
```

### 3. Пути к данным и чекпоинтам
- **Данные:** `/content/drive/MyDrive/FreshDoc_Artifacts/data/dataset_v1.parquet`
- **Чекпоинты:** `/content/drive/MyDrive/FreshDoc_Artifacts/checkpoints/`

### 4. Запуск скрипта обучения (FD-21)
```bash
!python scripts/train_m2.py \
    --config config/train.yaml \
    --data_path /content/drive/MyDrive/FreshDoc_Artifacts/data/dataset_v1.parquet \
    --output_dir /content/drive/MyDrive/FreshDoc_Artifacts/checkpoints/m2_run1
# Усечение/предсказания/оценка — как в шаге 5 Kaggle-варианта (пути на Drive).
```

---

## 3. Верификация базовых энкодеров
Перед тяжелым запуском убедиться в корректности CUDA и весов
(нужны `transformers>=4.48.0` для ModernBERT/GTE, `accelerate>=0.34.0` для Trainer/DDP):
```python
import torch
from transformers import AutoModel, AutoTokenizer

for m in ["microsoft/unixcoder-base", "answerdotai/ModernBERT-base",
          "Alibaba-NLP/gte-modernbert-base",
          "ibm-granite/granite-embedding-small-english-r2"]:
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModel.from_pretrained(m).to("cuda")
    out = mod(**tok("def test(): pass", return_tensors="pt").to("cuda"))
    print(m, tuple(out.last_hidden_state.shape))
print("Тест GPU и моделей пройден успешно.")
```
Быстрый smoke-тест пайплайна — на `granite-embedding-small-english-r2` (48M).