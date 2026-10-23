"""FD-39 фаза 0: сборка внешних parquet (HebCup + JIT) в схеме пар.

Пример:
    .venv/Scripts/python.exe scripts/build_external.py \
        --hebcup_zip data/external/hebcup/Dataset.zip \
        --jit_dir data/external/jit_raw \
        --out data/processed/external_java.parquet

Выход: parquet со схемой R1 (drift_type/sig_type=unknown, weight=0.0,
split in ext_hebcup/ext_jit) + JSON-манифест (объёмы, хэши входов).
Без фильтров R1, без обучения.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.external_adapter import (
    PAIR_COLUMNS,
    cocc_code_has_docstring_block,
    cocc_record_to_pair,
    go_record_to_pair,
    hebcup_record_to_pair,
    jit_record_to_pair,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_many(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8") + b"\x00")
        h.update(p.read_bytes())
    return h.hexdigest()


def _read_hebcup(zip_path: Path, member: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read(member).decode("utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _read_jit(jit_dir: Path, types: list[str], split: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for t in types:
        p = jit_dir / t / f"{split}.json"
        if not p.exists():
            print(f"JIT: нет файла {p} — пропускаю")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = list(data.values())
        for r in data:
            r = dict(r)
            r.setdefault("comment_type", t)
            recs.append(r)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: сборка external parquet.")
    ap.add_argument("--hebcup_zip", default="data/external/hebcup/Dataset.zip")
    ap.add_argument("--hebcup_member", default="Dataset/test_clean.jsonl")
    ap.add_argument("--no_hebcup", action="store_true")
    ap.add_argument("--jit_dir", default="data/external/jit_raw")
    ap.add_argument("--jit_types", default="Param")
    ap.add_argument("--jit_split", default="test")
    ap.add_argument("--no_jit", action="store_true")
    ap.add_argument("--cocc_dir", default="data/external/cocc_python")
    ap.add_argument("--no_cocc", action="store_true")
    ap.add_argument("--go_csv", default="")
    ap.add_argument("--go_sidecar", default="")
    ap.add_argument("--go_consensus", default="",
                    help="CSV id,label (A/B/C/D) после арбитража споров: метки Go "
                         "заменяются консенсусом (label_source=go_silver_consensus)")
    ap.add_argument("--out", default="data/processed/external_java.parquet")
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    if not args.no_hebcup:
        zp = Path(args.hebcup_zip)
        hb = _read_hebcup(zp, args.hebcup_member)
        print(f"HebCup {args.hebcup_member}: {len(hb)} строк")
        rows.extend(hebcup_record_to_pair(r) for r in hb)
        h = hashlib.sha256(zp.read_bytes()).hexdigest()
        inputs["hebcup_zip_sha256"] = h
        inputs["hebcup_member"] = args.hebcup_member
        inputs["hebcup_rows"] = str(len(hb))
    if not args.no_jit:
        jd = Path(args.jit_dir)
        types = [t.strip() for t in args.jit_types.split(",") if t.strip()]
        jr = _read_jit(jd, types, args.jit_split)
        print(f"JIT {types}/{args.jit_split}: {len(jr)} строк")
        rows.extend(jit_record_to_pair(r) for r in jr)
        inputs["jit_types"] = ",".join(types)
        inputs["jit_split"] = args.jit_split
        inputs["jit_rows"] = str(len(jr))
        jit_files = [jd / t / f"{args.jit_split}.json" for t in types]
        inputs["jit_sha256"] = _sha256_many([p for p in jit_files if p.exists()])

    if not args.no_cocc:
        cd = Path(args.cocc_dir)
        n_cocc, n_skip = 0, 0
        cocc_files: list[Path] = []
        if cd.exists():
            cocc_files = sorted(cd.glob("*.py"))
            for p in cocc_files:
                try:
                    row = cocc_record_to_pair(
                        p.name, p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    row = None
                if row is None:
                    n_skip += 1
                else:
                    rows.append(row)
                    n_cocc += 1
        print(f"CoCC python: {n_cocc} строк, пропущено {n_skip}")
        inputs["cocc_rows"] = str(n_cocc)
        inputs["cocc_skipped"] = str(n_skip)
        if cocc_files:
            inputs["cocc_sha256"] = _sha256_many(cocc_files)
        n_docstring = sum(1 for r in rows if r.get("split") == "ext_python2"
                          and cocc_code_has_docstring_block(r.get("code_old"), r.get("code_new")))
        inputs["cocc_docstring_blocks"] = str(n_docstring)
        print(f"CoCC: строк с docstring-блоками \"\"\" в коде: {n_docstring} "
              f"(docstring не режем, оставлена в code_old/code_new как есть)")
    if args.go_csv:
        import csv as _csv
        gp = Path(args.go_csv)
        sp = Path(args.go_sidecar) if args.go_sidecar else gp.with_name(gp.stem + "_sidecar.json")
        proxy = {r["id"]: int(bool(r.get("doc_changed_proxy")))
                 for r in json.loads(sp.read_text(encoding="utf-8"))}
        consensus: dict[str, str] = {}
        if args.go_consensus:
            with open(args.go_consensus, newline="", encoding="utf-8") as f:
                for r in _csv.DictReader(f):
                    lab = (r.get("label") or "").strip()
                    rid = (r.get("id") or "").strip()
                    if lab in ("A", "B", "C", "D") and rid:
                        consensus[rid] = lab
            n_d = sum(1 for v in consensus.values() if v == "D")
            print(f"Go консенсус: {len(consensus)} меток (D/не могу решить: {n_d} — "
                  f"эти строки исключаются из сборки, а не откатываются к прокси)")
        bmap = {"A": 1, "B": 1, "C": 0}  # D намеренно нет: не консенсус, а пропуск
        n_go = n_cons = n_skip_d = 0
        with open(gp, newline="", encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                if not (r.get("diff_code") or "").strip():
                    continue
                lab = consensus.get(r["id"])
                if lab == "D":
                    n_skip_d += 1
                    continue
                if lab in bmap:
                    row = go_record_to_pair(r, bmap[lab])
                    row["label_source"] = "go_silver_consensus"
                    n_cons += 1
                else:
                    row = go_record_to_pair(r, proxy.get(r["id"], 0))
                rows.append(row)
                n_go += 1
        print(f"Go silver: {n_go} строк из {gp.name} (консенсус: {n_cons}, "
              f"пропущено D: {n_skip_d})")
        inputs["go_rows"] = str(n_go)
        inputs["go_consensus"] = str(n_cons)
        inputs["go_skipped_D"] = str(n_skip_d)
        inputs["go_csv"] = str(gp)
        inputs["go_csv_sha256"] = _sha256_file(gp)
        inputs["go_sidecar_sha256"] = _sha256_file(sp)
        if args.go_consensus:
            inputs["go_consensus_sha256"] = _sha256_file(Path(args.go_consensus))
    df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    before = len(df)
    df = df.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
    print(f"Строк: {before} -> {len(df)} после дедупа pair_id")
    print(df.groupby(["split", "label_stale"]).size().to_dict())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    by = df.groupby(["split", "label_stale"]).size()
    by_str = {f"{s}/{int(lbl)}": int(n) for (s, lbl), n in by.items()}
    manifest = {
        "out": str(out),
        "rows": int(len(df)),
        "inputs": inputs,
        "by_split_label": by_str,
    }
    mp = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parquet: {out} ({len(df)} строк); манифест: {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
