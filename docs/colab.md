# Инструкция по запуску обучения в GPU-окружении (Kaggle / Colab)

Документ описывает процедуру развертывания вычислительной среды и запуска обучения моделей (M2, FD-21 / FD-24) для участников команды.

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
!git checkout main
!git pull origin main

# 4. Установка зависимостей проекта
!pip install -q -r requirements.txt
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

### 4. Запуск скрипта обучения
Выполнить из директории `/kaggle/working/FreshDoc`:

```bash
!python scripts/train.py \
    --config config/train.yaml \
    --data_path /kaggle/input/freshdoc-dataset/dataset_v1.parquet \
    --output_dir /kaggle/working/checkpoints/m2_run1
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
!git checkout main
!pip install -q -r requirements.txt
```

### 3. Пути к данным и чекпоинтам
- **Данные:** `/content/drive/MyDrive/FreshDoc_Artifacts/data/dataset_v1.parquet`
- **Чекпоинты:** `/content/drive/MyDrive/FreshDoc_Artifacts/checkpoints/`

### 4. Запуск скрипта обучения
```bash
!python scripts/train.py \
    --config config/train.yaml \
    --data_path /content/drive/MyDrive/FreshDoc_Artifacts/data/dataset_v1.parquet \
    --output_dir /content/drive/MyDrive/FreshDoc_Artifacts/checkpoints/m2_run1
```

---

## 3. Верификация базовых энкодеров
Перед тяжелым запуском убедиться в корректности CUDA и весов:
```python
import torch
from transformers import AutoModel, AutoTokenizer

for m in ["microsoft/codebert-base", "microsoft/unixcoder-base"]:
    tok = AutoTokenizer.from_pretrained(m)
    mod = AutoModel.from_pretrained(m).to("cuda")
    out = mod(**tok("def test(): pass", return_tensors="pt").to("cuda"))
    assert out.last_hidden_state.shape[-1] == 768
print("Тест GPU и моделей пройден успешно.")
```