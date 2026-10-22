"""FD-26: self-contained батчи для изолированных субагентов (паттерн FD-22).

    python scripts/make_fd26_batches.py

Читает reports/fd26_shards/fd26_shard_*.csv (слепые, без меток),
пишет reports/fd26_batches/fd26_batchNN.prompt.txt (NN=00..05):
внутри — полный текст задания R9 + все кейсы батча.
Субагенту передаётся ТОЛЬКО текст батча, репозиторий/датасет ему не нужны.
Ответ субагента — один CSV-блок (err_id,label,stale_statement,why),
оркестратор сохраняет его байт-в-байт как fd26_batchNN.response.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

SHARDS_DIR = PROJECT_ROOT / "reports" / "fd26_shards"
OUT_DIR = PROJECT_ROOT / "reports" / "fd26_batches"

HEADER = """FD-26: слепая R9-разметка ошибок M2. Ты — изолированный разметчик.
Доступа к репозиторию, датасету, меткам, другим батчам и интернету нет.
Единственный вход — кейсы ниже. Решай только по diff+docstring.

Вопрос на кейс: «нужно ли обновить документацию из-за этого изменения?»
Шкала:
- A. Нужно обновить — хотя бы одно утверждение docstring стало НЕВЕРНЫМ
  (имя параметра, дефолт, форма результата, исключение, поведение).
- B. Желательно обновить — неверного нет, но добавилось важное, чего нет в docstring.
- C. Актуально — всё остаётся верным и полным.
- D. Не могу решить — нужен контекст вне diff/docstring (укажи какой в why). D < 10%.
ЖЁСТКОЕ ПРАВИЛО: нет ложного утверждения в docstring — НЕ A.
Пример: дефолт изменился, но docstring дефолт не называл → C, не A.
Как размечать: выпиши проверяемые утверждения docstring; найди каждому фрагмент диффа;
есть ложное → A; неверного нет, но важное новое → B; иначе → C.
Сомнение A/B → B; B/C → C (поясни в why).

Верни ОДИН CSV-блок (без пояснений вне блока), заголовок строго:
err_id,label,stale_statement,why
- label в A/B/C/D; строк столько же, в том же порядке, что кейсы;
- stale_statement — точная подстрока DOC_OLD, ставшая ложной (только для A, иначе пусто);
  цитируй посимвольно;
- why — одна фраза (для D — какого контекста не хватило).
Код вне диффа не додумывать.

КЕЙСЫ:
"""


def case_block(r: pd.Series) -> str:
    return (
        f"=== CASE err_id={r['err_id']} ===\n"
        f"PROJECT: {r['project']} | SYMBOL: {r['symbol']}\n"
        f"--- SIG_OLD ---\n{str(r['sig_old'] or '').strip()}\n"
        f"--- DIFF ---\n{str(r['diff_code'] or '').strip()}\n"
        f"--- DOC_OLD ---\n{str(r['doc_old'] or '').strip()}\n"
        f"=== END CASE ===\n"
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="FD-26: self-contained батчи для субагентов.")
    ap.add_argument("--shards_dir", default=str(SHARDS_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    shards = sorted(Path(args.shards_dir).glob("fd26_shard_*.csv"))
    if not shards:
        raise SystemExit(f"нет шардов в {args.shards_dir}: сначала select_fd26_errors.py")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(out_dir.glob("fd26_batch*.prompt.txt")):
        old.unlink()
    for i, sp in enumerate(shards):
        df = pd.read_csv(sp, dtype=str, keep_default_na=False)
        if len(df) == 0:
            raise SystemExit(f"{sp.name}: пустой шард")
        if df["err_id"].duplicated().any():
            raise SystemExit(f"{sp.name}: дубли err_id")
        body = HEADER + "\n".join(case_block(r) for _, r in df.iterrows())
        out = out_dir / f"fd26_batch{i:02d}.prompt.txt"
        out.write_text(body, encoding="utf-8")
        print(f"{out.name}: {len(df)} кейсов, {out.stat().st_size} bytes")
    print(f"батчей: {len(shards)} -> {out_dir}")


if __name__ == "__main__":
    main()
