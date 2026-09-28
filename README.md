# FreshDoc

Мониторинг актуальности документации по изменениям кода методами машинного обучения.

## Быстрый старт

### Требования
- Python 3.11
- Git

### Установка
1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/o1uh/FreshDoc
   cd FreshDoc
   ```
2. Создать и активировать окружение:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Запустить тесты:
   ```bash
   pytest
   ```

## Соглашения по разработке (R15)
- Ветки: `feat/FD-xx-описание`, `fix/FD-xx-описание`
- Коммиты: `<type>(<scope>): <описание>` с `Refs: FD-xx` в теле
- Слияние в `main` строго через PR с >= 1 approve и зеленым CI
