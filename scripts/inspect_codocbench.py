import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

def main():
    base_dir = Path("data/external/codocbench/dataset")
    train_path = base_dir / "train.jsonl"
    test_path = base_dir / "test.jsonl"

    if not train_path.exists():
        print(f"Ошибка: {train_path} не найден", file=sys.stderr)
        sys.exit(1)

    records = []
    for p in [train_path, test_path]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

    print(f"Всего записей обработано: {len(records)}")

    v_lens = set(len(r.get("version_data", [])) for r in records)
    print(f"Количество версий в массиве version_data: {v_lens}")

    order_0_is_older = 0
    order_1_is_older = 0
    for r in records[:500]:
        vd = r.get("version_data", [])
        if len(vd) == 2:
            t0 = vd[0].get("commit_date_time", "")
            t1 = vd[1].get("commit_date_time", "")
            if t0 < t1:
                order_0_is_older += 1
            elif t1 < t0:
                order_1_is_older += 1

    print("\nПроверка хронологии commit_date_time:")
    print(f"  version_data[0] старше version_data[1]: {order_0_is_older} раз")
    print(f"  version_data[1] старше version_data[0]: {order_1_is_older} раз")

    if order_0_is_older > order_1_is_older:
        old_idx, new_idx = 0, 1
        print("-> Итог: version_data.0 — СТАРАЯ версия (до изменения), version_data.1 — НОВАЯ.")
    else:
        old_idx, new_idx = 1, 0
        print("-> Итог: version_data.1 — СТАРАЯ версия, version_data.0 — НОВАЯ.")

    lengths = []
    empty_code = 0
    for r in records:
        vd = r.get("version_data", [])
        if len(vd) > old_idx:
            code = vd[old_idx].get("code", "")
            if code:
                lengths.append(len(code.splitlines()))
            else:
                empty_code += 1

    print(f"\nЗаписей с пустым кодом: {empty_code}")
    print(f"Медианная длина функции (старая версия): {np.median(lengths):.1f} строк")
    print(f"Средняя длина функции: {np.mean(lengths):.1f} строк")
    print(f"Минимальная длина: {np.min(lengths)}, максимальная: {np.max(lengths)}")

if __name__ == "__main__":
    main()