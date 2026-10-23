"""FD-39 фаза 2: сборка Go-набора (full381) из пулов кандидатов.

    python scripts/make_go_silver.py --pool data/processed/go_gin400/candidates.jsonl:gin \
        --pool data/processed/go_cobra/candidates.jsonl:cobra \
        --out data/processed/go_full381.csv --n 381 --seed 42

Пулы: candidates.jsonl майнера (code_old/code_new БЕЗ godoc, doc_old/doc_new отдельно).
Отбор: все judgeable doc_changed (код+док правили вместе — кандидаты в stale)
+ добор code_only (код менялся, док нет — кандидаты в consistent) до n.
Метки прокси — ТОЛЬКО в sidecar (разметчикам не показывать, судьям недоступны).
Слепой CSV содержит id,file,symbol,repo,commit_*,sig_old,diff_code,doc_old,label,comment
(label/comment пустые). Разметка — scripts/make_go_batches.py + combine_go.py
с --input/--out_* на этот файл.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mining.go_pair import judgeable  # noqa: E402

OUT_COLS = ["id", "file", "symbol", "repo", "commit_old", "commit_new",
            "sig_old", "diff_code", "doc_old", "code_old", "code_new",
            "label", "comment"]


def load_pool(spec: str) -> list[dict]:
    # Windows-пути содержат ':' (C:\...), поэтому делим по ПОСЛЕДНЕМУ двоеточию.
    path, sep, repo = spec.rpartition(":")
    if not sep or not repo:
        raise SystemExit(f"pool без :repo: {spec}")
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["repo"] = repo
    print(f"пул {repo}: {len(rows)} кандидатов из {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: сборка Go silver-набора.")
    ap.add_argument("--pool", action="append", required=True,
                    help="candidates.jsonl:repo (повторить для каждого репо)")
    ap.add_argument("--out", default="data/processed/go_full381.csv")
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--n", type=int, default=381)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--must_include", default="",
                    help="CSV калибровочных 30 (те же колонки): его строки входят в набор "
                         "первыми — калибровочные 30 переиспользуются 1:1")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for spec in args.pool:
        all_rows.extend(load_pool(spec))
    def _key(r):
        return (r.get("repo", ""), r["file"], r["symbol"],
                r["commit_old"], r["commit_new"])

    def _matchkey(r):
        # с repo: тот же символ из разных репо — разные строки
        return (r.get("repo", ""), r["file"], r["symbol"],
                r["commit_old"], r["commit_new"])

    def _mustkey(r):
        # must-CSV (пилот) колонки repo может не иметь — матчим без неё,
        # но если repo в must-CSV указан, требуем совпадения
        return (r["file"], r["symbol"], r["commit_old"], r["commit_new"])

    must: list[dict] = []
    if args.must_include:
        import csv as _csv2
        with open(args.must_include, newline="", encoding="utf-8") as f:
            must_rows = list(_csv2.DictReader(f))
        by_pool: dict = {}
        for r in all_rows:
            by_pool.setdefault(_mustkey(r), []).append(r)
        n_missing = 0
        for m in must_rows:
            cands = by_pool.get(_mustkey(m), [])
            want_repo = (m.get("repo") or "").strip()
            if want_repo:
                cands = [c for c in cands if c.get("repo") == want_repo]
            if not cands:
                n_missing += 1
                continue
            if len(cands) > 1:
                print(f"ВНИМАНИЕ: {len(cands)} кандидатов пула на строку "
                      f"пилота {m.get('id', '?')} — беру первый")
            must.append(cands[0])
        if n_missing:
            print(f"ВНИМАНИЕ: {n_missing} строк пилота нет в пулах — пропускаю")
        print(f"must-include из пилота: {len(must)}/{len(must_rows)}")
        picked_ids = {id(m) for m in must}
        rest_pool = [r for r in all_rows if id(r) not in picked_ids]
    else:
        rest_pool = all_rows
    pool = judgeable(rest_pool)
    print(f"judgeable-пул: {len(pool)}/{len(all_rows)} (+{len(must)} must-include)")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pos = [c for c in pool if c["doc_changed"]]
    neg = [c for c in pool if not c["doc_changed"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    need = args.n - len(must)
    half = need // 2
    picked = must + pos[:half] + neg[:half]
    rest = pos[half:] + neg[half:]
    rng.shuffle(rest)
    picked += rest[: max(0, args.n - len(picked))]
    rng.shuffle(picked)
    if len(picked) < args.n:
        print(f"ВНИМАНИЕ: набрано {len(picked)} из {args.n} — пул исчерпан")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        for i, c in enumerate(picked):
            w.writerow({"id": f"go{i:03d}", "file": c["file"], "symbol": c["symbol"],
                        "repo": c.get("repo", ""), "commit_old": c["commit_old"],
                        "commit_new": c["commit_new"], "sig_old": c["sig_old"],
                        "diff_code": c["diff_code"], "doc_old": c["doc_old"],
                        "code_old": c.get("code_old", ""), "code_new": c.get("code_new", ""),
                        "label": "", "comment": ""})
    side = Path(args.sidecar) if args.sidecar else out.with_name(out.stem + "_sidecar.json")
    side.write_text(json.dumps(
        [{"id": f"go{i:03d}", "repo": c.get("repo", ""),
          "doc_changed_proxy": c["doc_changed"], "sig_changed": c["sig_changed"]}
         for i, c in enumerate(picked)], ensure_ascii=False, indent=2), encoding="utf-8")
    man = {"n": len(picked), "seed": args.seed,
           "doc_changed_proxy": sum(1 for c in picked if c["doc_changed"]),
           "by_repo": {r: sum(1 for c in picked if c.get("repo") == r)
                       for r in sorted({c.get("repo", "") for c in picked})},
           "pools": args.pool}
    mp = Path(args.manifest) if args.manifest else out.with_name(out.stem + "_manifest.json")
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"silver: {out} ({len(picked)}), sidecar: {side}, manifest: {mp}")
    print(f"баланс прокси: {man['doc_changed_proxy']}/{len(picked)} по репо {man['by_repo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
