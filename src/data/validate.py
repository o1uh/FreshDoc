"""FD-13: автоматические проверки качества dataset_v1 (R8).

Чистая логика без I/O: каждая проверка принимает DataFrame и возвращает
CheckResult. Используется скриптом scripts/validate_dataset.py (локальный
прогон на полном dataset_v1) и покрыта tests/test_validate.py на фиктивных
данных (этот же набор гоняется в CI).

Проверки R8:
- pair_id уникален;
- обязательные колонки схемы R1 на месте;
- doc_old не пуст;
- label_stale в {0, 1};
- единые фильтры R1 (имя без _, docstring >= 40, 4-300 строк);
- синтетика только в train/test_synth;
- одна функция синтетики (function_hash) не в нескольких сплитах;
- реальный проект не в нескольких сплитах;
- ни одна функция из train не встречается в val/test_real/test_synth/gold
  (включая дубликаты кода между проектами — вендоринг);
- drift_type синтетики согласован с меткой;
- вход модели (реальный срез select_model_inputs) не пересекается
  с _leak_*/_aux_* колонками таблицы;
- label_source из допустимого словаря R1;
- веса соответствуют label_source, неизвестных источников нет;
- sig_type из допустимого набора;
- (при переданных пулах) синтетика только из проектов train-пула.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.dataset import REAL_SOURCES, SYNTH_SOURCES, WEIGHTS

#: Колонки входа модели (инвариант из раздела 0 workflow.md).
MODEL_INPUT_COLS = ["sig_old", "diff_code", "doc_old"]

REQUIRED_COLS = [
    "pair_id", "source", "project", "commit_sha", "symbol",
    "function_hash", "sig_old", "code_old", "code_new", "diff_code",
    "doc_old", "label_stale", "drift_type", "sig_type",
    "label_source", "weight", "n_src_lines", "split",
]

ALLOWED_SPLITS = ("train", "val", "test_real", "test_synth", "gold")
ALLOWED_DRIFT = (
    "none", "param_renamed", "param_removed", "param_added",
    "default_changed", "return_changed", "exception_changed",
    "behavior_changed", "unknown",
)
ALLOWED_SIG = (
    "none", "param_renamed", "param_removed", "param_added",
    "default_changed", "return_changed", "exception_changed",
)
REAL_POS_DRIFT = set(ALLOWED_DRIFT) - {"none", "behavior_changed"}

#: Допустимые значения label_source (R1 + data_notes: gold и unlabeled —
#: известные категории апстрима, хотя в сборке dataset_v1 их нет).
ALLOWED_LABEL_SOURCES = (
    "real_coupled", "real_strong_neg", "real_weak_neg",
    "ast", "llm_p1", "llm_p2", "gold_human", "real_unlabeled",
)

#: Ожидаемые веса по label_source: карта сборки + 0.0 для gold (R1).
#: У real_unlabeled веса нет — такие строки обязаны отсутствовать в сборке
#: (build_dataset.py их исключает), поэтому weights-контроль их не пропускает.
EXPECTED_WEIGHTS = {**WEIGHTS, "gold_human": 0.0}


def select_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Срез таблицы ко входу модели — единственный разрешённый путь
    от датасета к признакам (инвариант раздела 0 workflow.md).

    Модельный код обязан брать признаки только через эту функцию, тогда
    check_model_inputs_isolated гарантирует отсутствие утечек.
    """
    return df[MODEL_INPUT_COLS]


@dataclass
class CheckResult:
    """Итог одной проверки."""

    name: str
    ok: bool
    detail: str
    n_fail: int = 0


def _missing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Колонки, отсутствующие в таблице (защита от KeyError)."""
    return [c for c in cols if c not in df.columns]


def _no_cols(name: str, missing: list[str]) -> CheckResult:
    return CheckResult(name=name, ok=False,
                       detail=f"нет колонок: {missing}", n_fail=len(missing))


def check_pair_id_unique(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["pair_id"])):
        return _no_cols("pair_id_unique", m)
    n_dup = int(df.duplicated(subset=["pair_id"], keep=False).sum())
    return CheckResult(
        name="pair_id_unique",
        ok=n_dup == 0,
        detail=f"дублей-строк pair_id: {n_dup}",
        n_fail=n_dup,
    )


def check_required_columns(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return CheckResult(
        name="required_columns",
        ok=not missing,
        detail="все колонки R1 на месте" if not missing else f"нет колонок: {missing}",
        n_fail=len(missing),
    )


def check_doc_old_nonempty(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["doc_old"])):
        return _no_cols("doc_old_nonempty", m)
    n = int((df["doc_old"].fillna("").str.len() == 0).sum())
    return CheckResult(
        name="doc_old_nonempty",
        ok=n == 0,
        detail=f"строк с пустым doc_old: {n}",
        n_fail=n,
    )


def check_label_values(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["label_stale"])):
        return _no_cols("label_values", m)
    bad = df.loc[~df["label_stale"].isin([0, 1])]
    return CheckResult(
        name="label_values",
        ok=len(bad) == 0,
        detail=(
            "label_stale только 0/1"
            if len(bad) == 0
            else f"вне {{0,1}}: {len(bad)} строк, значения {sorted(bad['label_stale'].unique().tolist())}"
        ),
        n_fail=len(bad),
    )


def check_filters_uniform(df: pd.DataFrame) -> CheckResult:
    """Единые фильтры R1 для всех источников."""
    if (m := _missing(df, ["symbol", "doc_old", "n_src_lines"])):
        return _no_cols("filters_uniform", m)
    last = df["symbol"].fillna("").str.split(".").str[-1]
    bad_name = last.str.startswith("_")
    bad_doc = df["doc_old"].fillna("").str.len() < 40
    bad_len = ~df["n_src_lines"].between(4, 300)
    n = int((bad_name | bad_doc | bad_len).sum())
    return CheckResult(
        name="filters_uniform",
        ok=n == 0,
        detail=(
            f"нарушений фильтров R1: {n} "
            f"(имя с _: {int(bad_name.sum())}, "
            f"doc<40: {int(bad_doc.sum())}, "
            f"строк вне [4,300]: {int(bad_len.sum())})"
        ),
        n_fail=n,
    )


def check_synth_split_allowed(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["source", "split"])):
        return _no_cols("synth_split_allowed", m)
    m = df["source"].isin(SYNTH_SOURCES) & ~df["split"].isin(["train", "test_synth"])
    bad = df.loc[m]
    return CheckResult(
        name="synth_split_allowed",
        ok=len(bad) == 0,
        detail=(
            "синтетика только в train/test_synth"
            if len(bad) == 0
            else f"синтетика вне train/test_synth: {len(bad)} строк, сплиты {sorted(bad['split'].unique().tolist())}"
        ),
        n_fail=len(bad),
    )


def check_synth_function_single_split(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["source", "function_hash", "split"])):
        return _no_cols("synth_function_single_split", m)
    syn = df.loc[df["source"].isin(SYNTH_SOURCES)]
    if len(syn) == 0:
        return CheckResult(name="synth_function_single_split", ok=True,
                           detail="синтетики нет — проверять нечего")
    n_multi = int(syn.groupby("function_hash")["split"].nunique().gt(1).sum())
    return CheckResult(
        name="synth_function_single_split",
        ok=n_multi == 0,
        detail=f"function_hash синтетики в нескольких сплитах: {n_multi}",
        n_fail=n_multi,
    )


def check_real_project_single_split(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["source", "project", "split"])):
        return _no_cols("real_project_single_split", m)
    real = df.loc[df["source"].isin(REAL_SOURCES)]
    if len(real) == 0:
        return CheckResult(name="real_project_single_split", ok=True,
                           detail="реальных строк нет — проверять нечего")
    multi = real.groupby("project")["split"].nunique()
    bad = sorted(multi[multi > 1].index.astype(str).tolist())
    return CheckResult(
        name="real_project_single_split",
        ok=not bad,
        detail=(
            f"реальных проектов: {real['project'].nunique()}, все в одном сплите"
            if not bad
            else f"проекты в нескольких сплитах ({len(bad)}): {bad[:10]}"
        ),
        n_fail=len(bad),
    )


def check_train_hash_isolation(df: pd.DataFrame) -> CheckResult:
    """Функция из train не встречается в других сплитах (R8).

    Ловит и дубликаты кода между проектами (вендоринг): сравнение идёт по
    function_hash независимо от проекта.
    """
    if (m := _missing(df, ["split", "function_hash"])):
        return _no_cols("train_hash_isolation", m)
    if "train" not in set(df["split"].unique()):
        return CheckResult(name="train_hash_isolation", ok=True,
                           detail="сплита train нет — проверять нечего")
    train_hashes = set(df.loc[df["split"] == "train", "function_hash"].astype(str))
    parts = []
    total = 0
    for other in ("val", "test_real", "test_synth", "gold"):
        sub = df.loc[df["split"] == other]
        if len(sub) == 0:
            continue
        overlap = train_hashes & set(sub["function_hash"].astype(str))
        parts.append(f"{other}: {len(overlap)}")
        total += len(overlap)
    return CheckResult(
        name="train_hash_isolation",
        ok=total == 0,
        detail=f"пересечений train по function_hash: {total} ({', '.join(parts)})",
        n_fail=total,
    )


def check_drift_label_consistency(df: pd.DataFrame) -> CheckResult:
    """drift_type согласован с меткой (R8, R1, R7)."""
    if (m := _missing(df, ["source", "label_stale", "drift_type", "label_source"])):
        return _no_cols("drift_label_consistency", m)
    problems: list[str] = []
    total = 0

    m = (df["source"] == "synth_ast") & (df["label_stale"] == 1)
    bad = df.loc[m & (df["drift_type"] == "none")]
    if len(bad):
        problems.append(f"ast pos с drift none: {len(bad)}")
        total += len(bad)
    m = (df["source"] == "synth_ast") & (df["label_stale"] == 0)
    bad = df.loc[m & (df["drift_type"] != "none")]
    if len(bad):
        problems.append(f"ast neg с drift != none: {len(bad)}")
        total += len(bad)
    m = (df["label_source"] == "llm_p1") | (
        (df["source"] == "synth_llm") & (df["label_stale"] == 1)
    )
    bad = df.loc[m & ((df["label_stale"] != 1) | (df["drift_type"] != "behavior_changed"))]
    if len(bad):
        problems.append(f"llm_p1 не (1/behavior_changed): {len(bad)}")
        total += len(bad)
    m = (df["label_source"] == "llm_p2") | (
        (df["source"] == "synth_llm") & (df["label_stale"] == 0)
    )
    bad = df.loc[m & ((df["label_stale"] != 0) | (df["drift_type"] != "none"))]
    if len(bad):
        problems.append(f"llm_p2 не (0/none): {len(bad)}")
        total += len(bad)
    m = df["source"] == "history_neg"
    bad = df.loc[m & ((df["label_stale"] != 0) | (df["drift_type"] != "none"))]
    if len(bad):
        problems.append(f"history_neg не (0/none): {len(bad)}")
        total += len(bad)
    m = df["source"] == "codocbench"
    bad = df.loc[m & ((df["label_stale"] != 1) | (~df["drift_type"].isin(REAL_POS_DRIFT)))]
    if len(bad):
        problems.append(f"codocbench не (1/известный тип или unknown): {len(bad)}")
        total += len(bad)

    return CheckResult(
        name="drift_label_consistency",
        ok=total == 0,
        detail="типы дрейфа согласованы с метками" if total == 0 else "; ".join(problems),
        n_fail=total,
    )


def check_model_inputs_isolated(df: pd.DataFrame) -> CheckResult:
    """Объявленный вход модели не пересекается с _leak_*/_aux_* таблицы.

    Проверяется не константа сама по себе, а результат реального среза
    select_model_inputs(df) против leak-колонок, выведенных из таблицы:
    если в контракт MODEL_INPUT_COLS попадёт утечка, срез её вернёт
    и проверка покраснеет.
    """
    if (m := _missing(df, MODEL_INPUT_COLS)):
        return _no_cols("model_inputs_isolated", m)
    leak_cols = {c for c in df.columns if c.startswith(("_leak_", "_aux_"))}
    selected = select_model_inputs(df)
    bad = [c for c in selected.columns if c in leak_cols]
    ok = not bad
    return CheckResult(
        name="model_inputs_isolated",
        ok=ok,
        detail=(
            f"вход модели {list(selected.columns)}; "
            f"_leak_/_aux_ в таблице: {len(leak_cols)}, во входе: {len(bad)}"
            if ok else
            f"утечка во входе модели: {bad} (лечится только правкой контракта)"
        ),
        n_fail=len(bad),
    )


def check_label_source_valid(df: pd.DataFrame) -> CheckResult:
    """label_source — из допустимого словаря R1 (дефект 3)."""
    if (m := _missing(df, ["label_source"])):
        return _no_cols("label_source_valid", m)
    values = set(df["label_source"].astype(str).unique())
    bad = sorted(values - set(ALLOWED_LABEL_SOURCES))
    return CheckResult(
        name="label_source_valid",
        ok=not bad,
        detail=(
            f"label_source: {len(values)} значений, все из словаря R1"
            if not bad
            else f"недопустимый label_source: {bad}"
        ),
        n_fail=len(bad),
    )


def check_weights_consistent(df: pd.DataFrame) -> CheckResult:
    """Веса соответствуют label_source; неизвестных источников нет (дефект 2)."""
    if (m := _missing(df, ["label_source", "weight"])):
        return _no_cols("weights_consistent", m)
    total = 0
    problems: list[str] = []
    values = set(df["label_source"].astype(str).unique())
    unknown = sorted(values - set(EXPECTED_WEIGHTS))
    if unknown:
        n_unk = int(df["label_source"].astype(str).isin(unknown).sum())
        problems.append(f"источник без заданного веса ({n_unk} строк): {unknown}")
        total += n_unk
    for label_source, w in EXPECTED_WEIGHTS.items():
        m = df["label_source"] == label_source
        if m.any():
            bad = int((m & (df["weight"] != w)).sum())
            if bad:
                problems.append(f"{label_source}: {bad} строк не с весом {w}")
                total += bad
    return CheckResult(
        name="weights_consistent",
        ok=total == 0,
        detail="веса по R1" if total == 0 else "; ".join(problems),
        n_fail=total,
    )


def check_sig_type_valid(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["sig_type"])):
        return _no_cols("sig_type_valid", m)
    bad = df.loc[~df["sig_type"].isin(ALLOWED_SIG)]
    return CheckResult(
        name="sig_type_valid",
        ok=len(bad) == 0,
        detail=(
            "sig_type из допустимого набора"
            if len(bad) == 0
            else f"недопустимый sig_type: {len(bad)} строк, значения {sorted(bad['sig_type'].unique().tolist())}"
        ),
        n_fail=len(bad),
    )


def check_splits_valid(df: pd.DataFrame) -> CheckResult:
    if (m := _missing(df, ["split"])):
        return _no_cols("splits_valid", m)
    bad = df.loc[~df["split"].isin(ALLOWED_SPLITS)]
    return CheckResult(
        name="splits_valid",
        ok=len(bad) == 0,
        detail=(
            f"сплиты: {df['split'].value_counts().to_dict()}"
            if len(bad) == 0
            else f"недопустимый split: {len(bad)} строк, значения {sorted(bad['split'].unique().tolist())}"
        ),
        n_fail=len(bad),
    )


def check_synth_only_train_pool(
    df: pd.DataFrame, pool_of: dict[str, str]
) -> CheckResult:
    """Синтетика — только из функций проектов train-пула (раздел 0)."""
    if (m := _missing(df, ["source", "project"])):
        return _no_cols("synth_only_train_pool", m)
    syn_projects = set(df.loc[df["source"].isin(SYNTH_SOURCES), "project"].astype(str))
    bad = sorted(p for p in syn_projects if pool_of.get(str(p)) != "train_pool")
    unknown = sorted(p for p in syn_projects if str(p) not in pool_of)
    return CheckResult(
        name="synth_only_train_pool",
        ok=not bad,
        detail=(
            f"проектов в синтетике: {len(syn_projects)}, все из train-пула"
            if not bad
            else f"синтетика не из train-пула ({len(bad)}): {bad[:10]}; без пула: {unknown[:10]}"
        ),
        n_fail=len(bad),
    )


def run_all_checks(
    df: pd.DataFrame, pool_of: dict[str, str] | None = None
) -> list[CheckResult]:
    """Все проверки R8 по порядку. Без pool_of пропуск pool-проверки."""
    results = [
        check_pair_id_unique(df),
        check_required_columns(df),
        check_splits_valid(df),
        check_doc_old_nonempty(df),
        check_label_values(df),
        check_label_source_valid(df),
        check_filters_uniform(df),
        check_synth_split_allowed(df),
        check_synth_function_single_split(df),
        check_real_project_single_split(df),
        check_train_hash_isolation(df),
        check_drift_label_consistency(df),
        check_model_inputs_isolated(df),
        check_weights_consistent(df),
        check_sig_type_valid(df),
    ]
    if pool_of is not None:
        results.append(check_synth_only_train_pool(df, pool_of))
    return results


def all_ok(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results)
