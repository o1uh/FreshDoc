"""FD-39: точечная докачка CoCC python_ccset_raw по fileId (вежливо, с бэкоффом).

gdown --folder упирается в лимиты Drive ("many accesses") и падает на сотнях
файлов. Здесь: перемешанный порядок, пропуск существующих, повторы с бэкоффом.

Пример:
    .venv/Scripts/python.exe scripts/download_cocc.py --n 1500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gdown  # noqa: E402


def load_ids(path: Path) -> list[tuple[str, str]]:
    """Пары (fileId, имя) из ids-файла; пустые строки пропускаются."""
    pairs: list[tuple[str, str]] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split(None, 1)
        if len(parts) != 2:
            print(f"ids: пропуск битой строки без имени: {ln.strip()[:80]!r}")
            continue
        fid, name = parts
        pairs.append((fid.strip(), name.strip()))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: докачка CoCC python по fileId.")
    ap.add_argument("--ids", default="data/external/cocc_ids.txt")
    ap.add_argument("--out", default="data/external/cocc_python")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max_fail_streak", type=int, default=20,
                    help="выйти, если столько файлов подряд не скачались "
                         "(квота Drive исчерпана; повторный запуск продолжит)")
    args = ap.parse_args()

    pairs = load_ids(Path(args.ids))
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in out.glob("*.py")}
    print(f"Всего id: {len(pairs)}, уже на диске: {len(have)}, цель новых: {args.n}")

    ok, fail, skipped, streak = 0, [], 0, 0
    for fid, name in pairs:
        if ok >= args.n:
            break
        if streak >= args.max_fail_streak:
            print(f"Стоп: {streak} неудач подряд (квота Drive). "
                  f"Повторный запуск продолжит с пропуском существующих.")
            break
        dest = out / name
        if dest.exists():
            skipped += 1
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        done = False
        for att in range(args.attempts):
            try:
                gdown.download(url, str(dest), quiet=True)
                if dest.exists() and dest.stat().st_size > 0:
                    done = True
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: попытка {att + 1}: {str(e)[:120]}")
            time.sleep(30 * (att + 1))
        if done:
            ok += 1
            streak = 0
            if ok % 50 == 0:
                print(f"  скачано новых: {ok}")
        else:
            fail.append(name)
            streak += 1
            if dest.exists():
                dest.unlink(missing_ok=True)
        time.sleep(args.sleep)

    man = {"target_new": args.n, "downloaded_new": ok, "skipped_existing": skipped,
           "failed": fail, "seed": args.seed}
    (out / "download_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: новых {ok}, пропущено {skipped}, ошибок {len(fail)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
