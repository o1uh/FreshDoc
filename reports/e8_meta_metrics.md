# FD-15 Metadata Logistic Regression Metrics

Training set: 21997 rows (pos 10946, neg 11051)

## Coefficients

| Feature | Coefficient |
|---|---|
| `func_len` | -0.0004 |
| `param_cnt` | +0.0930 |
| `has_args` | +0.5037 |
| `has_returns` | -0.1260 |
| `has_raises` | -0.0981 |
| intercept | -0.3727 |

## ROC-AUC

| Split | Rows | Pos | Neg | ROC‑AUC |
|---|---|---|---|---|
| val | 559 | 211 | 348 | **0.6779** |
| test_real | 2895 | 1444 | 1451 | **0.5689** |

> **⚠️ ROC-AUC val = 0.6779 > 0.65.**
>
> Val-проекты — подмножество train-пула; модель выучивает project-specific паттерны. На test_real (другие проекты) AUC ≈ 0.57 — обходного пути через метаданные нет.

---

Generated automatically by FD‑15 pipeline.
