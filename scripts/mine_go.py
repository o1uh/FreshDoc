"""FD-39 фаза 1: драйвер Go-майнинга (git-история gin/cobra -> кандидаты + пилот 30).

Пример:
    .venv/Scripts/python.exe scripts/mine_go.py --repo ../gin-pilot \
        --commits 100 --out_dir data/processed/go_gin

Шаги:
1. список коммитов (новые первыми), берём N, идём хронологически;
2. на коммите: файлы .go (без _test.go и vendor/) через git ls-tree,
   содержимое через git show в temp-каталог, парсинг бинарником gominer;
3. pairing соседних коммитов (src/mining/go_pair);
4. отчёт валидации (counts, покрытие доками, стабильность имён);
5. пилот 30 (стратифицированно doc_changed 15/15) + sidecar с прокси-меткой.

Go-бинарник: scripts/gominer (go build) или `go run ./scripts/gominer`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mining.go_pair import judgeable, pair_maps, select_pilot, stability, with_doc  # noqa: E402


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout


def list_commits(repo: str, n: int) -> list[tuple[str, str]]:
    out = _git(repo, "log", "--no-merges", "-n", str(n), "--format=%H|%cI")
    rows = []
    for line in out.splitlines():
        if "|" in line:
            sha, date = line.split("|", 1)
            rows.append((sha.strip(), date.strip()))
    return list(reversed(rows))  # хронологически


def gominer_cmd(gominer: str, tmpdir: str) -> list[str]:
    """Команда запуска gominer. Строка 'go run ...' парсится через shlex
    (пути с пробелами в кавычках не рвутся); иначе — готовый путь к бинарнику."""
    if gominer.startswith("go "):
        return [*shlex.split(gominer), "-dir", tmpdir]
    return [gominer, "-dir", tmpdir]


def snapshot_funcs(repo: str, sha: str, gominer: str) -> dict[tuple[str, str], dict]:
    files = [
        ln for ln in
        _git(repo, "ls-tree", "-r", "--name-only", sha).splitlines()
        if ln.endswith(".go") and not ln.endswith("_test.go") and "vendor/" not in ln
    ]
    tmp = Path(tempfile.mkdtemp(prefix="gominer_"))
    try:
        for f in files:
            try:
                content = subprocess.run(
                    ["git", "-C", repo, "show", f"{sha}:{f}"],
                    capture_output=True, timeout=120,
                ).stdout
            except subprocess.TimeoutExpired:
                continue
            if not content:
                continue
            p = tmp / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        cmd = gominer_cmd(gominer, str(tmp))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"gominer: {r.stderr.strip()[:300]}")
        funcs = json.loads(r.stdout)
        out: dict[tuple[str, str], dict] = {}
        for x in funcs:
            if not isinstance(x, dict) or not x.get("file") or not x.get("symbol"):
                print(f"gominer: пропуск записи без file/symbol: {str(x)[:120]!r}")
                continue
            out[(x["file"], x["symbol"])] = x
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: Go-майнинг кандидатов.")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commits", type=int, default=100)
    ap.add_argument("--gominer", default="go run ./scripts/gominer")
    ap.add_argument("--out_dir", default="data/processed/go_gin")
    ap.add_argument("--pilot_n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    commits = list_commits(args.repo, args.commits)
    print(f"Коммитов: {len(commits)} ({commits[0][0][:8]}..{commits[-1][0][:8]})")
    snaps: list[dict] = []
    for i, (sha, date) in enumerate(commits):
        m = snapshot_funcs(args.repo, sha, args.gominer)
        cov = sum(1 for v in m.values() if (v.get("doc") or "").strip()) / max(len(m), 1)
        print(f"  [{i + 1}/{len(commits)}] {sha[:8]} {date[:10]} funcs={len(m)} doc_cov={cov:.2f}")
        snaps.append(m)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_cands: list[dict] = []
    stab_lines: list[str] = []
    for i in range(1, len(snaps)):
        c = pair_maps(snaps[i - 1], snaps[i], commits[i - 1][0], commits[i][0])
        all_cands.extend(c)
        st = stability(set(snaps[i - 1]), set(snaps[i]))
        stab_lines.append(
            f"{commits[i - 1][0][:8]}->{commits[i][0][:8]}: kept={st['kept']:.3f} "
            f"({st['n_old']}->{st['n_new']}), candidates={len(c)}"
        )
    by = {"doc_changed": sum(1 for c in all_cands if c["doc_changed"]),
          "code_only": sum(1 for c in all_cands if not c["doc_changed"])}
    print(f"Кандидатов: {len(all_cands)} {by}")

    (out / "candidates.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_cands), encoding="utf-8")
    report = {
        "repo": args.repo,
        "commits": len(commits),
        "sha_from": commits[0][0], "sha_to": commits[-1][0],
        "candidates": len(all_cands), **by,
        "stability": stab_lines[-5:],
    }
    (out / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    pilot_pool = judgeable(all_cands)
    print(f"Пилот-пул (непустые doc_old и diff): {len(pilot_pool)}/{len(all_cands)}")
    pilot = select_pilot(pilot_pool, n=args.pilot_n, seed=args.seed)
    with open(out / "pilot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "file", "symbol", "commit_old", "commit_new",
            "sig_old", "diff_code", "doc_old", "label", "comment"])
        w.writeheader()
        for i, c in enumerate(pilot):
            w.writerow({"id": f"go{i:03d}", "file": c["file"], "symbol": c["symbol"],
                        "commit_old": c["commit_old"], "commit_new": c["commit_new"],
                        "sig_old": c["sig_old"], "diff_code": c["diff_code"],
                        "doc_old": c["doc_old"], "label": "", "comment": ""})
    (out / "pilot_sidecar.json").write_text(json.dumps(
        [{"id": f"go{i:03d}", "doc_changed_proxy": c["doc_changed"],
          "sig_changed": c["sig_changed"]} for i, c in enumerate(pilot)],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Пилот: {out / 'pilot.csv'} ({len(pilot)}), sidecar записан")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
