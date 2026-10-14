"""FD-06: отбор отрицательных пар и выравнивание (R5).

    python scripts/select_negatives.py [--seed 42]

Строго по R5: добор недостающего strong типом weak, дефицит фиксируется.
Strong из synth_ast НЕ подмешивается (источник фальсифицировать нельзя).
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
OUT_JSON = PROJECT_ROOT / "reports" / "neg_select.json"


def _plot_svg(pos, cand_all, selected, path: Path) -> None:
    """Запасной вариант без matplotlib: групповые столбцы (stdlib-SVG)."""
    order = ["4-8", "9-15", "16-40", "41-300"]
    panels = [
        ("До: положительные vs все кандидаты", cand_all),
        ("После: положительные vs отобрано", selected),
    ]
    W, H, pad = 560, 300, 40
    bw = (W - 2 * pad) // (len(order) * 3)
    top = 20
    height = H - pad - top
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        'font-family="sans-serif" font-size="11">'
    ]
    for pi, (title, frame) in enumerate(panels):
        x0 = pad + pi * (W // 2)
        parts.append(f'<text x="{x0}" y="14">{title}</text>')
        p = pos["bin"].value_counts(normalize=True).reindex(order).fillna(0.0)
        q = frame["bin"].value_counts(normalize=True).reindex(order).fillna(0.0)
        vmax = max(float(p.max()), float(q.max()), 0.01)
        for i, b in enumerate(order):
            for j, v in enumerate((float(p[b]), float(q[b]))):
                bh = (v / vmax) * height
                x = x0 + i * 3 * bw + j * bw
                y = top + height - bh
                color = "#4a90d9" if j == 0 else "#e07a5f"
                parts.append(
                    f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw - 2}" '
                    f'height="{bh:.0f}" fill="{color}"/>'
                )
                parts.append(
                    f'<text x="{x + 2:.0f}" y="{y - 3:.0f}">{v:.2f}</text>'
                )
            parts.append(
                f'<text x="{x0 + i * 3 * bw + 4:.0f}" y="{H - 8}">{b}</text>'
            )
    parts.append(
        f'<text x="{pad}" y="{H - 22}">синий — positive, оранжевый — negative</text>'
    )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_before_after(pos, cand_all, selected, path: Path) -> None:
    """Гистограммы долей до/после (matplotlib; без него — SVG)."""
    path = Path(path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _plot_svg(pos, cand_all, selected, path.with_suffix(".svg"))
        return
    order = ["4-8", "9-15", "16-40", "41-300"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, frame, title in (
        (axes[0], cand_all, "До: положительные vs все кандидаты"),
        (axes[1], selected, "После: положительные vs отобрано"),
    ):
        p = pos["bin"].value_counts(normalize=True).reindex(order).fillna(0.0)
        q = frame["bin"].value_counts(normalize=True).reindex(order).fillna(0.0)
        x = range(len(order))
        ax.bar([i - 0.2 for i in x], p.values, width=0.4, label="positive")
        ax.bar([i + 0.2 for i in x], q.values, width=0.4, label="negative")
        ax.set_xticks(list(x), order)
        ax.set_title(title)
        ax.set_ylabel("доля")
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-06: отбор отрицательных пар (R5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in_neg", default=str(IN_NEG))
    parser.add_argument("--in_pos", default=str(IN_POS))
    parser.add_argument("--in_pools", default=str(IN_POOLS))
    parser.add_argument("--out_neg", default=str(OUT_NEG))
    parser.add_argument("--out_un", default=str(OUT_UN))
    parser.add_argument("--out_def", default=str(OUT_DEF))
    parser.add_argument("--out_fig", default=str(OUT_FIG))
    args = parser.parse_args()

    df_neg_raw = pd.read_parquet(args.in_neg)
    df_pos = pd.read_parquet(args.in_pos)
    df_pools = pd.read_parquet(args.in_pools)
    pool_of = dict(zip(df_pools["project"].astype(str), df_pools["pool"].astype(str)))

    df_filt, filt_stats = apply_r1_filters(df_neg_raw)
    df_cand, df_un, role_stats = build_r1_rows(df_filt, pool_of)

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
    assert set(selected["split_pool"]) <= set(df_pos["split_pool"]), "Неизвестный пул в отборе"

    out_neg = selected.drop(columns=["bin"])
    out_path = Path(args.out_neg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_neg.to_parquet(out_path, index=False)
    out_un_path = Path(args.out_un)
    out_un_path.parent.mkdir(parents=True, exist_ok=True)
    df_un.to_parquet(out_un_path, index=False)
    deficit.to_csv(args.out_def, index=False)
    plot_before_after(df_pos, df_cand, selected, Path(args.out_fig))

    summary = {
        "seed": args.seed,
        "filters": filt_stats,
        "roles": role_stats,
        "selected_total": int(len(selected)),
        "selected_by_pool": {
            str(k): int(v)
            for k, v in selected["split_pool"].value_counts().items()
        },
        "selected_strong": int((selected["label_source"] == "real_strong_neg").sum()),
        "selected_weak": int((selected["label_source"] == "real_weak_neg").sum()),
        "unlabeled_total": int(len(df_un)),
        "ks_before": round(ks_before, 4),
        "ks_after": round(ks_after, 4),
        "ks_by_pool": ks_by_pool,
        "max_deficit_rate": float(deficit["deficit_rate"].max()) if len(deficit) else 0.0,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"Фильтры R1: {filt_stats}")
    print(f"Роли: {role_stats}")
    print(
        f"Отобрано: {len(selected)} "
        f"(strong {summary['selected_strong']}, weak {summary['selected_weak']})"
    )
    print(f"KS до/после: {ks_before:.4f} / {ks_after:.4f}; по пулам: {ks_by_pool}")
    print(f"Макс. дефицит итога: {summary['max_deficit_rate']:.2%}")
    print(deficit.to_string(index=False))
    print(f"Сохранено: {out_path} ({len(out_neg)}), {out_un_path} ({len(df_un)})")


if __name__ == "__main__":
    main()
