# Deep Research: открытые модели для парной классификации FreshDoc (merged)

**Дата:** 18.10.2026 (merged-ревизия: слеплено из трёх источников, все цифры — оценки)
**Источники:** merged из трёх: websearch Opencode через Muse Spark 1.3 Xhigh x2 + deep-research Antigravity через Gemini 3.8 flash High x1. (методология расчётов VRAM/времени взята из наиболее строгого из них).
**Задача:** Cross-Encoder DualHead: `p_stale` (BCEWithLogits) + `drift_type` 8 классов (CE, `ignore_index=-100`).
**Вход:** `[CLS] sig_old + diff_code [SEP] doc_old [SEP]`, full cross-attention.
**Железо:** Kaggle 2× Tesla T4 16 ГБ (Turing SM 7.5, PCIe, без NVLink), квота 30 ч/нед, сессия 9–12 ч. DDP — модель целиком на каждой карте, лимит 16 ГБ/карта.

> **Правило чтения этого файла.** Все VRAM/времена ниже — аналитические оценки, точность ±25–30%.
> Ни одна цифра не является замером. Перед фиксацией бюджета обязателен пилот 100 шагов (по аналогии FD-05):
> замерить с/шаг, пересчитать эпоху как `688 шагов × с/шаг`. Пилот решает, оценка — нет.

---

## 0. Жёсткие ограничения T4

| Фактор | Значение |
|---|---|
| Пик FP16 dense | 65 TFLOPS/карта (130 на 2), FP32 8.1, bandwidth ~320 ГБ/с, PCIe3 ~32 ГБ/с |
| `bfloat16` | Нет на кремнии. Эмуляция через FP32 → 5–10× медленнее. Только **FP16 mixed** |
| FlashAttention-2 | Нет на SM 7.5. Только `sdpa` (`math`/`mem_efficient`), xFormers-cutlass |
| DDP 2×T4 | Allreduce по PCIe: speedup реальный **1.6–1.7×**, не 2×. Для 1.5B+ падает к ~1.4× |
| Память | Полезный лимит **≤14.5–15.0 ГБ** из 16 (запас: CUDA-ctx ~1.2 ГБ + фрагментация ×1.15 + eval/лоадер) |
| Gradient Checkpointing | Память падаёт в ~3–3.5×, FLOPs растут на ~30% (throughput ×0.77) |

## 1. Профиль данных и усечение

Квантили (22k train, ~5.2k val/test; измерены на BPE типа RoBERTa; **токенизатор указать при замере**):
`p50≈340, p75≈560, p90≈870, p95≈1100, p99≈1950, max≈3800`.

| Окно | Усечение (оценка) | Смысл |
|---|---:|---|
| 512 | **28.6%** | Каждая 3-я пара битая: режется хвост диффа или `doc_old`. Для продакшна неприемлемо |
| 1024 | **4.4%** | Минимум для работы |
| 2048 | **~0.7–1.0%** | Целевая рабочая точка |
| 4096 | **0.0%** | Только если хвост p99 важен; ×2 время ради 1% |

Оговорка про Qwen: словарь 151k с токенами отступов упаковывает код плотнее RoBERTa примерно на треть
(оценка из ресёрчей: 25–38%, **проверить замером на FreshDoc**, не факт). Если подтвердится, эффективное покрытие Qwen@1024
ближе к 95%+, чем 4.4% потерь. В таблицу это не вписано — только как гипотеза.

## 2. Методология расчётов (взята из наиболее строгого внешнего ресёрча)

**Статика (на 1 GPU):**
- Full FT FP16-mixed AdamW: `16 байт/парам` (W2+G2+master4+m4+v4). 125M→2.00 ГБ; 149M→2.38; 395M→6.32; 0.49B→7.84 (поэтому Full FT 0.5B+ на 16 ГБ — впритык/риск).
- LoRA FP16: `2·P + ~0.2–0.4 ГБ` (адаптеры r=16–32 + опт). QLoRA NF4: `~0.5·P + ~0.6 ГБ`.

**Динамика:** без GC full-attention `O(N²)` материализован (верхняя оценка; SDPA-chunking даёт скидку ≤1.5–2×, в запас не кладём).
С GC храним входы слоёв + пик одного слоя. Пик карты: `(статика + активации) × 1.15 + 1.2 ГБ`.

**Время:** 22k при eff-батче 32 → **688 шагов/эпоха**. Реальный MFU SDPA на SM 7.5 — **25–35%** (35–45 TFLOPS sustained на 2 карты).
Отсюда: 149M@1024 (~29 TFLOP/шаг) → **0.7–1.0 с/шаг → 8–11 мин/эпоха**. Любая оценка «2–3 мин» требует ~100% пика и неверна.
Даём два числа: padded-worst (верхняя граница) и realistic (dynamic padding + bucketing + unpadding у ModernBERT, −15–25%).

## 3. Master-матрица (реалистичные диапазоны, не точечные замеры)

| Модель | Контекст | Режим | VRAM (оценка) | Эпоха 22k (оценка) | Вердикт |
|---|---|---|---|---|---|
| `microsoft/unixcoder-base` 125M | 1024 (поз. растяжка, не натив 512) | Full FT FP16 | ~4.5–6.0 ГБ | **~6–8 мин** | ⭐ Бейзлайн M2 v1 (FD-21). Кодовый, дешёвый, CLS |
| `answerdotai/ModernBERT-base` 149M | 8192 (раб. 1024/2048) | Full FT FP16 (+GC на 2048) | ~5.5–6.5 @1024; ~9–12 @2048 | **~8–11 мин @1024; ~13–18 мин @2048** | ⭐ Цель M2 v2 (FD-24). Единственный Full FT с 1% trunc |
| `answerdotai/ModernBERT-large` 395M | 8192 | Full FT+GC / LoRA | ~11–13 @1024 | ~15–24 мин | Резерв ёмкости, если base упрётся в head-2 |
| `microsoft/codebert-base` 125M | 512 (1024 — интерполяция) | Full FT | ~3.5–4.5 | ~2–4 мин | Устаревший: 28.6% trunc. Только как нижняя граница |
| `microsoft/graphcodebert-base` 125M | 512 | Full FT | как CodeBERT +10–15% | как CodeBERT | Не брать: кап + хрупкий dataflow на diff с `+/-` |
| `allenai/longformer-base-4096` 149M | 4096 sliding-512 | Full FT | линейная, ~12–15 @2048–4096 | ~13–25 мин | Запасной long-context, если ModernBERT не встанет. Нет code-pretrain, дальние связи только через CLS |
| `microsoft/codereviewer` 220M (T5) | 512 (ext 1024) | Full FT enc / LoRA | ~7–8.5 | ~12–16 мин | Diff-специалист (`<add>/<del>`). Только E8/feature-extractor, не основа |
| `Qwen2.5-Coder-0.5B` | 32k | LoRA FP16 (+GC); Full FT только с 8-bit Adam, риск OOM | ~4–6 LoRA @1024; ~10–12 Full @1024 | ~6–12 LoRA; ~16–22 Full | Causal-контроль. Видимость хуже энкодера |
| `Qwen2.5-Coder-1.5B` | 32k | LoRA FP16 / QLoRA (Full FT невозможен: статика 24.6 ГБ) | ~7–10 @1024 | **~30–45 мин** (3 эп 1.5–2 ч) | Опциональный E8 semantic-трек. Causal-штраф + 10× время base |
| `Qwen2.5-Coder-3B` | 32k | QLoRA +GC | ~12–15 @1024 | ~50–90 мин | Не брать: +0.5–1.5 п.п. за ×2 время |
| `Qwen2.5-Coder-7B` | 32k | QLoRA (программный INT4 на Turing) | ~10–12 | >60 мин/эп | ❌ Дисквалифицирован (квота) |
| `deepseek-coder-1.3B` | 16k | LoRA | ~6–8 @1024 | ~15–30 мин | Жёлтый, но лицензия хуже Qwen + слабее на NL-доках |
| `starcoder2-3b` | 16k SWA | QLoRA | ~12–15 @1024 | ~30–60+ мин | ❌ Медленнее Qwen-3B при gated-лицензии |
| `codegemma-2b` | 8k | QLoRA | ~7–10 @1024 | ~22–45 мин | Нет причин вместо Apache Qwen |
| `codet5p-220m` / `770m` | 512 | Full enc-only / LoRA | ~6–13 | ~4–20 мин | Кап + мёртвый декодер для классификации |
| `bge-m3` 568M | 8192 | Только LoRA (Full OOM) | vocab-налог 250k | ~30–60 мин | Fallback для мультиязычных доков, не code |
| MoE (`DeepSeek-V2-Lite`), ByT5 | — | — | OOM/×4 длина | >90 мин | ❌ |

Читать вердикты: ⭐ — в план FD-21/FD-24; жёлтые — только E8; ❌ — не брать.

## 4. Что взять из каждого исходника

- Из строгого внешнего ресёрча: методология §2, зонирование зелёная/жёлтая/красная, DDP-рецепты (`NCCL_P2P_DISABLE=1`, `find_unused_parameters=False`), quota-план, Longformer/BGE как fallback.
- Из нашего: CodeReviewer как diff-эксперт, DualHead-код ниже, анализ токенизаторов как гипотеза.
- Из второго внешнего ресёрча-дубля: ничего сверх — дублирует с худшими таймингами; оставлен для истории, цифры не использовать.

## 5. DualHead: пулинг, лосс, гейт (норматив для R10)

- Пулинг по умолчанию **`[CLS]`**. Masked Mean — только абляция E3 на FreshDoc (литературные +1–2% F1 не факт для нас).
- Декодеры (если E8): Last Non-Pad pooling. Causal-асимметрия (код не видит док вперёд) — причина не ставить декодер основой.
- Лосс — Masked Conditional: `L = BCE(stale) + λ·I(y_stale=1)·CE(drift)`, `ignore_index=-100` при `stale=0`.
  `λ ∈ {0.3, 0.6, 1.0}`, старт 0.6, выбор по `val`. Фиксированный λ запрещён.
- Гейт инференса: `drift_type` отдаётся только если `p_stale > τ` (τ из FD-25). Иначе `stale=0, drift=None`.
- Оценка: macro-F1 дрейфа — **только на `y_stale=1`**. На всей выборке — некорректно (голова не видела `none`).

```python
import torch, torch.nn as nn
class DualHeadCodeClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_drift_classes=8, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.stale_head = nn.Linear(hidden_size, 1)
        self.drift_head = nn.Linear(hidden_size, num_drift_classes)
        self.stale_criterion = nn.BCEWithLogitsLoss()
        self.drift_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self, input_ids, attention_mask, labels_stale=None, labels_drift=None, lambda_drift=0.6):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)  # mean; для CLS заменить на h[:,0]
        pooled = self.layer_norm(self.dropout(pooled))
        logit_s = self.stale_head(pooled).squeeze(-1)
        logit_d = self.drift_head(pooled)
        loss = None
        if labels_stale is not None:
            loss_s = self.stale_criterion(logit_s, labels_stale.float())
            if labels_drift is not None:
                md = labels_drift.clone(); md[labels_stale == 0] = -100
                loss = loss_s + lambda_drift * self.drift_criterion(logit_d, md)
            else: loss = loss_s
        return {"loss": loss, "logits_stale": logit_s, "logits_drift": logit_d}
```

## 6. Рекомендации

1. FD-21: строго `unixcoder-base @1024`, CLS, Full FT. Пилот 100 шагов + доля trunc + время — в приёмку.
2. FD-24: `ModernBERT-base @1024` (контроль домена) vs `@2048` (контроль контекста). Large и Qwen — только E8.
3. Не делать: `bf16` на T4, FA2 на Turing, 512-кап как основа, старт с 3B/7B, `AutoModelForSequenceClassification` для causal без `pad_token`/left-padding фикса.

## 7. Kaggle-рецепт

`transformers>=4.48, torch>=2.2, sdpa (не flash), fp16=True, bf16=False, GC on при ≥2048, per-device 8 + accum 2 → eff 32 (DDP), AdamW lr 2e-5 (головы 3e-5), wd 0.01, warmup 6–10%, clip 1.0, ckpt каждую эпоху + resume (сессии рвутся).`

## 8. Лицензии (проверено частично — сверить HF-страницу перед релизом)

MIT: CodeBERT/GraphCodeBERT. Apache-2.0 ungated: UniXcoder, ModernBERT base/large, Longformer, Qwen2.5-Coder 0.5/1.5 (3B — предварительно Apache-2.0; встречавшееся утверждение про Qwen Research для 3B требует проверки страницы модели). BSD-3: CodeT5p. MIT: BGE-M3.
Gated/ограниченные: StarCoder2-3B (OpenRAIL-M + accept + токен), CodeGemma-2B (Gemma Terms + accept), DeepSeek-Coder (DeepSeek License, не OSI).
