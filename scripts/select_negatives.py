"""FD-06: отбор отрицательных пар и выравнивание (R5).

    python scripts/select_negatives.py [--seed 42]

Строго по R5: добор недостающего strong типом weak, дефицит фиксируется.
Strong из synth_ast НЕ подмешивается (источник фальсифицировать нельзя).
pair_id для history_neg расширенный: hash(source:project:symbol:commit:
code_old + code_new + doc_old) — иначе вендоринг-дубли с разным диффом
схлопывались и коллизии cand/unlabeled (см. src/data/negatives.py).
Канон графиков — PNG через matplotlib (в requirements); без него — SVG.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.negatives import (
    apply_r1_filters,
    assign_bin,
    build_r1_rows,
    ks_stat,
    select_balanced,
)

IN_NEG = PROJECT_ROOT / "data" / "interim" / "negatives_raw.parquet"
IN_POS = PROJECT_ROOT / "data" / "interim" / "pairs_codocbench.parquet"
IN_POOLS = PROJECT_ROOT / "data" / "interim" / "project_pools.parquet"
OUT_NEG = PROJECT_ROOT / "data" / "interim" / "pairs_negatives.parquet"
OUT_UN = PROJECT_ROOT / "data" / "interim" / "unlabeled_sigchange.parquet"
OUT_DEF = PROJECT_ROOT / "reports" / "tables" / "neg_deficit.csv"
OUT_FIG = PROJECT_ROOT / "reports" / "figures" / "neg_length_before_after.png"
OUT_JSON = PROJECT_ROOT / "reports" / "tables" / "neg_select.json"
OUT_DROPPED = PROJECT_ROOT / "reports" / "tables" / "neg_dropped_pair_ids.txt"

DEFICIT_LIMIT = 0.2


def _plot_svg(pos, cand_all, selected, path: Path) -> None:
    """Запасной вариант без matplotlib: групповые столбцы (stdlib-SVG)."""
    order = ["4-8", "9-15", "16-40", "41-300"]
    panels = [
        ("До: положительные vs все кандидаты", cand_all),
        ("После: положительные vs отобрано", selected),
    ]
    width, height, pad = 560, 300, 40
    bw = (width - 2 * pad) // (len(order) * 3)
    top = 20
    plot_h = height - pad - top
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" font-family="sans-serif" font-size="11">'
    ]
    for pi, (title, frame) in enumerate(panels):
        x0 = pad + pi * (width // 2)
        parts.append(f'<text x="{x0}" y="14">{title}</text>')
        p = pos["bin"].value_counts(normalize=True).reindex(order)
        q = frame["bin"].value_counts(normalize=True).reindex(order)
        vmax = max(float(p.max()), float(q.max()), 0.01)
        for i, b in enumerate(order):
            for j, v in enumerate((float(p[b]), float(q[b]))):
                bh = (v / vmax) * plot_h
                x = x0 + i * 3 * bw + j * bw
                y = top + plot_h - bh
                color = "#6B7C9B" if j == 0 else "#C17C60"
                parts.append(
                    f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw - 2}" '
                    f'height="{bh:.0f}" fill="{color}"/>'
                )
                parts.append(f'<text x="{x + 2:.0f}" y="{y - 3:.0f}">'
                             f"{v:.2f}</text>")
            parts.append(f'<text x="{x0 + i * 3 * bw + 4:.0f}" '
                         f'y="{height - 8}">{b}</text>')
    parts.append(
        f'<text x="{pad}" y="{height - 22}">'
        "синий — positive, оранжевый — negative</text>"
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_before_after(pos, cand_all, selected, path: Path) -> None:
    """Гистограммы долей до/после. Формат — по суффиксу пути."""
    path = Path(path)
    if path.suffix.lower() == ".svg":
        _plot_svg(pos, cand_all, selected, path)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Muted note gamma (FD-33): paper #FAF9F7, axes #FFFFFF, ink #2B2B2B,
    # positive slate #6B7C9B / negative terra #C17C60, grid #E5E0D8.
    PAPER = "#FAF9F7"
    CARD = "#FFFFFF"
    INK = "#2B2B2B"
    MUTED = "#6B7280"
    SLATE = "#6B7C9B"
    TERRA = "#C17C60"
    BORDER = "#E5E0D8"

    order = ["4-8", "9-15", "16-40", "41-300"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.patch.set_facecolor(PAPER)
    for ax, frame, title in (
        (axes[0], cand_all, "До: положительные vs все кандидаты"),
        (axes[1], selected, "После: положительные vs отобрано"),
    ):
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        p = pos["bin"].value_counts(normalize=True).reindex(order)
        q = frame["bin"].value_counts(normalize=True).reindex(order)
        x = range(len(order))
        ax.bar([i - 0.2 for i in x], p.values, width=0.4, label="positive",
               color=SLATE, edgecolor=CARD, zorder=3)
        ax.bar([i + 0.2 for i in x], q.values, width=0.4, label="negative",
               color=TERRA, edgecolor=CARD, zorder=3)
        ax.set_xticks(list(x), order)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        ax.set_ylabel("доля", fontsize=11, color=INK)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.7, color=BORDER, zorder=0)
        leg = ax.legend(fontsize=10, frameon=True, facecolor=CARD, edgecolor=BORDER)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FD-06: отбор отрицательных пар (R5)."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_neg", default=str(IN_NEG))
    parser.add_argument("--in_pos", default=str(IN_POS))
    parser.add_argument("--in_pools", default=str(IN_POOLS))
    parser.add_argument("--out_neg", default=str(OUT_NEG))
    parser.add_argument("--out_un", default=str(OUT_UN))
    parser.add_argument("--out_def", default=str(OUT_DEF))
    parser.add_argument("--out_fig", default=str(OUT_FIG))
    parser.add_argument("--out_json", default=str(OUT_JSON))
    parser.add_argument("--out_dropped", default=str(OUT_DROPPED))
    args = parser.parse_args()

    df_neg_raw = pd.read_parquet(args.in_neg)
    df_pos = pd.read_parquet(args.in_pos)
    df_pools = pd.read_parquet(args.in_pools)
    pool_of = dict(zip(df_pools["project"].astype(str),
                       df_pools["pool"].astype(str)))

    df_filt, filt_stats = apply_r1_filters(df_neg_raw)
    df_cand, df_un, role_stats, dropped_ids = build_r1_rows(df_filt, pool_of)

    df_pos = df_pos.copy()
    df_pos["bin"] = df_pos["n_src_lines"].map(assign_bin)
    df_cand["bin"] = df_cand["n_src_lines"].map(assign_bin)

    ks_before = ks_stat(df_pos["n_src_lines"], df_cand["n_src_lines"])
    selected, deficit = select_balanced(df_pos, df_cand, seed=args.seed)
    ks_after = ks_stat(df_pos["n_src_lines"], selected["n_src_lines"])
    ks_by_pool = {}
    for pool in sorted(df_pos["split_pool"].unique()):
        a = df_pos[df_pos["split_pool"] == pool]["n_src_lines"]
        b = selected[selected["split_pool"] == pool]["n_src_lines"]
        ks_by_pool[str(pool)] = round(ks_stat(a, b), 4) if len(b) else None

    assert selected["pair_id"].is_unique, "Дубликаты pair_id в отборе!"
    assert df_un["pair_id"].is_unique, "Дубликаты pair_id в unlabeled!"
    assert not (
        set(selected["pair_id"].astype(str)) & set(df_un["pair_id"].astype(str))
    ), "pair_id одновременно в отборе и unlabeled (нарушение FD-13)!"
    assert set(selected["split_pool"]) <= set(df_pos["split_pool"]), \
        "Неизвестный пул в отборе"

    out_neg = selected.drop(columns=["bin"])
    out_path = Path(args.out_neg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_neg.to_parquet(out_path, index=False)
    out_un_path = Path(args.out_un)
    out_un_path.parent.mkdir(parents=True, exist_ok=True)
    df_un.to_parquet(out_un_path, index=False)
    deficit.to_csv(args.out_def, index=False)
    plot_before_after(df_pos, df_cand, selected, Path(args.out_fig))
    dropped_path = Path(args.out_dropped)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.write_text("\n".join(dropped_ids) + "\n", encoding="utf-8")

    max_deficit = float(deficit["deficit_rate"].max()) if len(deficit) else 0.0
    over_limit = bool((deficit["deficit_rate"] > DEFICIT_LIMIT).any())
    summary = {
        "seed": args.seed,
        "pair_id_scheme": "history_neg: sha256(source:project:symbol:commit:"
        "code_old + code_new + doc_old)[:16]; "
        "полные дубликаты схлопываются, варианты с разным диффом/doc "
        "сохраняются",
        "filters": filt_stats,
        "roles": role_stats,
        "pair_id_dedup_dropped_rows": int(role_stats.get(
            "pair_id_dedup_dropped", 0)),
        "pair_id_dedup_dropped_unique": int(role_stats.get(
            "pair_id_dedup_dropped_unique", len(dropped_ids))),
        "cross_pair_id_collisions": int(role_stats.get(
            "cross_pair_id_collisions", 0)),
        "selected_total": int(len(selected)),
        "selected_by_pool": {
            str(k): int(v)
            for k, v in selected["split_pool"].value_counts().items()
        },
        "selected_strong": int(
            (selected["label_source"] == "real_strong_neg").sum()),
        "selected_weak": int(
            (selected["label_source"] == "real_weak_neg").sum()),
        "strong_target_total": int(deficit["strong_target"].sum()),
        "strong_deficit_total": int(deficit["strong_deficit"].sum()),
        "unlabeled_total": int(len(df_un)),
        "ks_before": round(ks_before, 4),
        "ks_after": round(ks_after, 4),
        "ks_by_pool": ks_by_pool,
        "max_deficit_rate": max_deficit,
        "deficit_over_20": over_limit,
    }
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Фильтры R1: {filt_stats}")
    print(f"Роли: {role_stats}")
    print(f"Дроп pair_id-дубликатов: строк {summary['pair_id_dedup_dropped_rows']}, "
          f"уникальных {summary['pair_id_dedup_dropped_unique']} "
          f"-> {args.out_dropped} "
          f"(файл — уникальные id; кросс-коллизий cand/un: "
          f"{summary['cross_pair_id_collisions']})")
    print(f"Отобрано: {len(out_neg)} "
          f"(strong {summary['selected_strong']} из "
          f"{summary['strong_target_total']})")
    print(f"KS до/после: {ks_before:.4f} / {ks_after:.4f}; "
          f"по пулам: {ks_by_pool}")
    print(f"Макс. дефицит итога: {max_deficit:.2%} "
          f"(флаг >20%: {over_limit})")
    print(deficit.to_string(index=False))
    if over_limit:
        print("ВНИМАНИЕ: дефицит >20% в ячейке — вернуться в FD-05!")
    print(f"Сохранено: {out_path} ({len(out_neg)}), "
          f"{out_un_path} ({len(df_un)})")


if __name__ == "__main__":
    main()
