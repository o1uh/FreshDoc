"""FD-12: сборка dataset_v1 и сплиты (R1, R8).

    python scripts/build_dataset.py

Входы (финалы): pairs_codocbench (2997) + pairs_negatives (2997)
  + synth_ast_v2 (29515) + synth_llm_main700 (822).
Исключены: synth_ast v1, synth_llm_main300, synth_llm 120, pilot,
  unlabeled (label -1, только gold/replay), p3 (не пары, аугментация doc).

Сплиты R1: реальные test_pool -> test_real; реальные train_pool ->
  train/val по проекту (sha256(project)%100: <80 train, иначе val);
  синтетика -> train/test_synth по function_hash (<90 train).
  Val — только реальные строки; синтетика val-проектов исключается
  полностью (R8: проект не в нескольких сплитах); test_synth — только
  функции без реальных строк (иначе sig_old/doc_old совпадают с train).
sig_type пересчитывается для всех строк; вес llm_p1 = 0.5 в сборке.
Чистая логика — в src/data/dataset.py, покрыта tests/test_build_dataset.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.dataset import (
    REAL_SOURCES,
    apply_weights,
    assert_known_projects,
    assign_splits,
    compute_val_projects,
    dedup_by_pair_id,
    drop_leaked_test_synth,
    drop_val_project_synthetic,
    hash_overlap,
    projects_of,
    recompute_sig_column,
)

IN_CODOC = PROJECT_ROOT / "data" / "interim" / "pairs_codocbench.parquet"
IN_NEG = PROJECT_ROOT / "data" / "interim" / "pairs_negatives.parquet"
IN_AST = PROJECT_ROOT / "data" / "interim" / "synth_ast_v2.parquet"
IN_LLM = PROJECT_ROOT / "data" / "interim" / "synth_llm_main700.parquet"
IN_POOLS = PROJECT_ROOT / "data" / "interim" / "project_pools.parquet"
OUT_DATA = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_STATS = PROJECT_ROOT / "reports" / "data_stats.md"
OUT_MANIFEST = PROJECT_ROOT / "reports" / "dataset_v1.manifest.json"

CORE_COLS = ["pair_id", "source", "project", "commit_sha", "symbol",
             "function_hash", "sig_old", "code_old", "code_new", "diff_code",
             "doc_old", "label_stale", "drift_type", "sig_type",
             "label_source", "weight", "n_src_lines"]
LEAK_COLS = ["_leak_doc_new", "_leak_diff_docstring", "_leak_commit_message"]
AUX_COLS = ["_aux_src_old"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_to_root(p: Path) -> str:
    """Относительный путь от корня репозитория (переносимый манифест)."""
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="FD-12: сборка dataset_v1 (R1, R8).")
    ap.add_argument("--in_codocbench", default=str(IN_CODOC))
    ap.add_argument("--in_negatives", default=str(IN_NEG))
    ap.add_argument("--in_synth_ast", default=str(IN_AST))
    ap.add_argument("--in_synth_llm", default=str(IN_LLM))
    ap.add_argument("--in_pools", default=str(IN_POOLS))
    ap.add_argument("--out", default=str(OUT_DATA))
    ap.add_argument("--out_stats", default=str(OUT_STATS))
    ap.add_argument("--out_manifest", default=str(OUT_MANIFEST))
    args = ap.parse_args()

    in_files = {
        "pairs_codocbench": Path(args.in_codocbench),
        "pairs_negatives": Path(args.in_negatives),
        "synth_ast_v2": Path(args.in_synth_ast),
        "synth_llm_main700": Path(args.in_synth_llm),
        "project_pools": Path(args.in_pools),
    }
    for name, p in in_files.items():
        if not p.exists():
            raise FileNotFoundError(f"Вход {name} отсутствует: {p}")
    file_hashes = {n: sha256_file(p) for n, p in in_files.items()}
    df_cod = pd.read_parquet(in_files["pairs_codocbench"])
    df_neg = pd.read_parquet(in_files["pairs_negatives"])
    df_ast = pd.read_parquet(in_files["synth_ast_v2"])
    df_llm = pd.read_parquet(in_files["synth_llm_main700"])
    df_pools = pd.read_parquet(in_files["project_pools"])
    file_rows = {"pairs_codocbench": len(df_cod), "pairs_negatives": len(df_neg),
                 "synth_ast_v2": len(df_ast), "synth_llm_main700": len(df_llm)}
    print(f"Входы: codocbench {len(df_cod)}, negatives {len(df_neg)}, "
          f"ast_v2 {len(df_ast)}, llm {len(df_llm)}")

    # 1. Склейка в порядке приоритета (реальные первыми) + дедуп pair_id.
    for df, name in ((df_cod, "codocbench"), (df_neg, "history_neg"),
                     (df_ast, "synth_ast"), (df_llm, "synth_llm")):
        extra = [c for c in df.columns if c in ("pilot_id", "main_id")]
        if extra:
            print(f"ВНИМАНИЕ: {name} содержит вне-схемные колонки {extra} — отбрасываются")
            df.drop(columns=extra, inplace=True)
    df_all, before, dedup_removed, dup_rows = dedup_by_pair_id([df_cod, df_neg, df_ast, df_llm])
    print(f"Склеено: {before}; дедуп pair_id: снято {dedup_removed} (дублей-строк {dup_rows})")
    assert df_all["pair_id"].is_unique, "pair_id не уникален после дедупа!"

    # 2. Пулы и val-проекты (только по пулам, без данных).
    pool_of = dict(zip(df_pools["project"].astype(str), df_pools["pool"].astype(str)))
    val_projects = compute_val_projects(pool_of)
    assert_known_projects(df_all["project"].astype(str).unique(), pool_of)
    synth_projects = set(df_all.loc[df_all["source"].isin(("synth_ast", "synth_llm")), "project"])
    bad_pool = sorted(p for p in synth_projects if pool_of[str(p)] != "train_pool")
    print(f"Проектов в синтетике: {len(synth_projects)}, не-train: {bad_pool}")
    assert not bad_pool, f"Синтетика из test-пула: {bad_pool}"
    print(f"Вал-проектов: {len(val_projects)}: {sorted(val_projects)}")

    # 3. Полное исключение синтетики val-проектов (R8).
    df_all, n_valsynth = drop_val_project_synthetic(df_all, val_projects)
    print(f"Синтетики val-проектов снято: {n_valsynth}")

    # 4. Пересчёт sig_type для всех строк (R8).
    df_all, sig_ok, sig_mismatch = recompute_sig_column(df_all)
    print(f"sig_type пересчитан: ok {sig_ok}/{len(df_all)}, расхождений со stored {sig_mismatch}")
    if sig_mismatch:
        print("ВНИМАНИЕ: расхождения sig_type — использован пересчитанный (канон R8)")

    # 5. Сплиты R1.
    df_all["split"] = assign_splits(df_all, pool_of, val_projects).values
    print(f"Сплиты назначены: {df_all['split'].value_counts().to_dict()}")
    val_sources = set(df_all.loc[df_all["split"] == "val", "source"].unique())
    assert val_sources <= set(REAL_SOURCES), f"В val попала синтетика: {val_sources}"
    assert not (projects_of(df_all, "test_synth") & val_projects), "val-проекты в test_synth"
    outside = set(df_all.loc[df_all["split"] != "test_real", "project"].astype(str).unique())
    assert all(pool_of[p] == "train_pool" for p in outside), "test_pool вне test_real"
    print(f"Контроль train/val по function_hash: пересечение {len(hash_overlap(df_all, 'train', 'val'))}")
    assert not hash_overlap(df_all, "train", "val"), "Утечка train->val!"
    print(f"Контроль train/test_real по function_hash: пересечение {len(hash_overlap(df_all, 'train', 'test_real'))}")
    assert not hash_overlap(df_all, "train", "test_real"), "Утечка train->test_real!"

    # 6. test_synth — только невиденные функции: исключение мутаций функций,
    # встречающихся в реальных строках (иначе sig_old/doc_old совпадают с train).
    df_all, n_tsleak = drop_leaked_test_synth(df_all)
    print(f"Мутаций виденных функций снято из test_synth: {n_tsleak}")
    print(f"Контроль train/test_synth по function_hash: пересечение {len(hash_overlap(df_all, 'train', 'test_synth'))}")
    assert not hash_overlap(df_all, "train", "test_synth"), "Утечка train->test_synth!"

    # 6. Веса R1 (+llm_p1=0.5).
    df_all, w_fix = apply_weights(df_all)
    print(f"Веса приведены к R1: исправлено {w_fix} строк")

    # 7. _leak/_aux и лёгкие проверки R1.
    for c in LEAK_COLS + AUX_COLS:
        if c not in df_all.columns:
            df_all[c] = ""
        else:
            df_all[c] = df_all[c].fillna("")
    errs = []
    if (df_all["doc_old"].fillna("").str.len() == 0).any():
        errs.append("пустой doc_old")
    if not set(df_all["label_stale"].unique()) <= {0, 1}:
        errs.append(f"label_stale вне {{0,1}}: {sorted(df_all['label_stale'].unique())}")
    bad_name = df_all["symbol"].fillna("").str.split(".").str[-1].str.startswith("_")
    if bad_name.any():
        errs.append(f"имя с _: {int(bad_name.sum())}")
    bad_doc = df_all["doc_old"].fillna("").str.len() < 40
    if bad_doc.any():
        errs.append(f"doc<40: {int(bad_doc.sum())}")
    bad_len = ~df_all["n_src_lines"].between(4, 300)
    if bad_len.any():
        errs.append(f"n_src_lines вне [4,300]: {int(bad_len.sum())}")
    m_pos = (df_all["label_source"] == "ast") & (df_all["label_stale"] == 1)
    if (df_all.loc[m_pos, "drift_type"] == "none").any():
        errs.append("ast pos с drift none")
    m_neg = (df_all["label_source"] == "ast") & (df_all["label_stale"] == 0)
    if (df_all.loc[m_neg, "drift_type"] != "none").any():
        errs.append("ast neg с drift != none")
    if errs:
        raise RuntimeError("Проверки R1 не пройдены: " + "; ".join(errs))
    print("Проверки R1 пройдены (фильтры, метки, согласованность ast)")

    # 8. Порядок колонок и сохранение.
    out_cols = CORE_COLS + ["split"] + LEAK_COLS + AUX_COLS
    df_all = df_all[out_cols]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    print(f"Сохранено: {out_path} ({len(df_all)} строк)")

    # 9. Таблица split x source x label x drift.
    grp = (df_all.groupby(["split", "source", "label_stale", "drift_type"])
           .size().reset_index(name="n").sort_values(["split", "source", "label_stale", "drift_type"]))
    # 10. Манифест (версионируется в reports/).
    try:
        import pandas as _pd
        pandas_ver = _pd.__version__
    except Exception:
        pandas_ver = "?"
    try:
        import pyarrow as _pa
        pa_ver = _pa.__version__
    except Exception:
        pa_ver = "?"
    manifest = {
        "dataset": "dataset_v1",
        "built_by": "scripts/build_dataset.py (FD-12)",
        "inputs": {n: {"path": _rel_to_root(p), "sha256": file_hashes[n], "rows": file_rows.get(n, len(df_pools) if n == "project_pools" else None)}
                   for n, p in in_files.items()},
        "params": {
            "dedup": "pair_id drop_duplicates keep=first, приоритет codocbench>history_neg>synth_ast>synth_llm",
            "dedup_removed": int(dedup_removed),
            "val_isolation": f"val — только реальные строки; синтетика val-проектов исключена полностью ({n_valsynth} строк); test_synth очищен от функций реальных строк ({n_tsleak} строк)",
            "sig_type": "пересчитан для всех строк (derive_drift_type по code_old/code_new)",
            "sig_mismatch_stored": int(sig_mismatch),
            "split_real": "test_pool->test_real; train_pool: sha256(project)%100<80 train иначе val",
            "split_synth": "sha256(function_hash)%100<90 train иначе test_synth (единая карта ast+llm)",
            "weights": {"real_coupled": 1.0, "real_strong_neg": 1.0, "real_weak_neg": 0.5,
                        "ast": 1.0, "llm_p1": 0.5, "llm_p2": 0.5,
                        "note": "llm_p1 0.5 — решение команды 13.10/15.10 (аудит 56-60% шума)"},
            "excluded": ["synth_ast v1 (5575, подмножество пула v2)",
                         "synth_llm_main300 (449, 332 pair overlap с main700)",
                         "synth_llm (120, legacy)", "synth_llm_pilot (22)",
                         "unlabeled_sigchange (label -1, только gold/replay)",
                         "p3_main700 (не пары, аугментация doc)"],
        },
        "val_projects": sorted(val_projects),
        "counts": grp.to_dict(orient="records"),
        "totals": {"rows": int(len(df_all)),
                   "by_split": df_all["split"].value_counts().to_dict(),
                   "by_source": df_all["source"].value_counts().to_dict()},
        "versions": {"python": platform.python_version(), "pandas": pandas_ver, "pyarrow": pa_ver},
    }
    manifest["inputs"]["project_pools"]["rows"] = int(len(df_pools))
    man_path = Path(args.out_manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Манифест: {man_path}")

    # 11. reports/data_stats.md
    lines = ["# dataset_v1: статистика сборки (FD-12)", "",
             f"- Строк: {len(df_all)} (входы: codocbench {len(df_cod)}, negatives {len(df_neg)}, "
             f"ast_v2 {len(df_ast)}, llm {len(df_llm)}; дедуп pair_id −{dedup_removed}, "
             f"синтетика val-проектов −{n_valsynth}, виденные функции из test_synth −{n_tsleak})",
             f"- Сплиты: {df_all['split'].value_counts().to_dict()}",
             f"- Вал-проекты ({len(val_projects)}): {', '.join(sorted(val_projects))}",
             "- Val — только реальные строки; val-проекты отсутствуют в train/test_synth; "
             "test_synth — только функции без реальных строк; "
             "пересечения по function_hash train/val, train/test_real, train/test_synth отсутствуют (assert в сборке)",
             f"- sig_type пересчитан, расхождений со stored: {sig_mismatch}",
             "- Веса: real_coupled 1.0, strong 1.0, weak 0.5, ast 1.0, llm_p1 0.5*, llm_p2 0.5 (*решение команды)",
             "", "## split × source × label × drift_type", "",
             "| split | source | label | drift_type | n |", "|---|---|---|---|---|"]
    for _, r in grp.iterrows():
        lines.append(f"| {r['split']} | {r['source']} | {r['label_stale']} | {r['drift_type']} | {r['n']} |")
    lines += ["", f"Манифест: `{man_path.name}` (хэши входов, параметры, версии).",
              f"Файл: `{out_path.name}` ({len(df_all)} строк)."]
    sp = Path(args.out_stats)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Статистика: {sp}")
    print(df_all["split"].value_counts().to_string())
    print(grp.to_string(index=False))


if __name__ == "__main__":
    main()
