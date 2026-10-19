# Deep Research: открытые модели для парной классификации FreshDoc (merged & modernized)

**Дата актуализации:** 19.10.2026 (ревизия 2: интеграция реестра llm-checker, аудит точных спецификаций Hugging Face config/safetensors и deep-research)  
**Источники:** websearch Opencode (Muse Spark 1.3 Xhigh) + deep-research Antigravity (Gemini 3.8 Flash High) + аудит реестра `llm-checker` (55k+ артефактов HuggingFace/ModelScope/Ollama, бенчмарки BigCodeBench, EvalPlus, LiveBench) + прямая верификация `config.json` и метаданных `safetensors`.  
**Задача:** Cross-Encoder DualHead: `p_stale` (BCEWithLogits) + `drift_type` 8 классов (CE, `ignore_index=-100`).  
**Вход:** `[CLS] sig_old + diff_code [SEP] doc_old [SEP]`, full cross-attention.  
**Железо:** Kaggle 2× Tesla T4 16 ГБ (Turing SM 7.5, PCIe 3.0, без NVLink), квота 30 ч/нед, сессия 9–12 ч. DDP — модель целиком на каждой карте, лимит 16 ГБ/карта.  

> **Правило чтения этого файла.** Все значения VRAM и времени обучения — аналитические оценки sustained-производительности с точностью ±20–25%.  
> Окончательная калибровка скорости проводится пилотным замером на 100 шагах (по аналогии с FD-05): замерить `с/шаг`, пересчитать время эпохи как `688 шагов × с/шаг`.

---

## 0. Жёсткие аппаратные ограничения 2× Tesla T4

| Фактор | Спецификация T4 | Влияние на выбор модели и пайплайн |
|---|---|---|
| **Пик FP16 dense** | 65 TFLOPS/карта (130 TFLOPS на пару), FP32 8.1, пропускная способность памяти ~320 ГБ/с, шина PCIe 3.0 ~32 ГБ/с | Реальный sustained MFU для PyTorch SDPA составляет **25–35%** (35–45 sustained TFLOPS). Любые расчеты быстрее 0.7 с/шаг для 149M@1024 нереалистичны. |
| **`bfloat16`** | **Отсутствует аппаратно на Turing (SM 7.5)** | Эмуляция через FP32 дает замедление в 5–10 раз. Допустим только режим **FP16 mixed** (`torch.cuda.amp.autocast(dtype=torch.float16)`). |
| **FlashAttention-2** | **Не поддерживается на SM 7.5** (требует Ampere SM 8.0+) | Используется нативный PyTorch SDPA (`math` или `mem_efficient`) либо xFormers. |
| **DDP over PCIe 3.0** | Allreduce по шине PCIe без NVLink | Реальное ускорение от второй карты составляет **1.6–1.7×** (для моделей >1B падает до ~1.4× из-за накладных расходов синхронизации градиентов). |
| **Полезный VRAM** | **$\le 14.5–15.0$ ГБ** из 16 ГБ | Резерв: CUDA context ~1.2 ГБ + фрагментация PyTorch allocators ($\times 1.15$) + буферы DataLoader. Превышение 15.0 ГБ вызывает немедленный `CUDA out of memory`. |
| **Gradient Checkpointing** | Снижает VRAM в 3.0–3.5× ценой +30% FLOPs | Обязателен при контексте $\ge 2048$ для моделей от 149M и при любом контексте для декодеров. |

---

## 1. Профиль данных FreshDoc и штраф усечения контекста

Квантили длин токенов (22k train, 5.2k val/test; BPE-токенизация типа RoBERTa/ModernBERT):  
$p_{50} \approx 340,\; p_{75} \approx 560,\; p_{90} \approx 870,\; p_{95} \approx 1100,\; p_{99} \approx 1950,\; \max \approx 3800$.

| Размер окна | Доля усечения примеров | Практическое следствие для продукта |
|---|---:|---|
| **512 токенов** | **28.6%** ($p_{75}=560$) | **Неприемлемо для продакшна**: обрезается каждый 3-й пример. Теряется либо целевой docstring в конце конкатенации, либо ключевые фрагменты диффа. |
| **1024 токена** | **4.4%** | Минимальная рабочая точка для бейзлайна (UniXcoder с поз. растяжкой). |
| **2048 токенов** | **~0.7–1.0%** | **Целевая рабочая точка**: потери контекста пренебрежимо малы (<1%), покрывается 99.3% датасета. |
| **4096+ токенов** | **0.0%** | Избыточно: удвоение вычислительных затрат ради 0.7% примеров из крайнего хвоста. |

---

## 2. Методология расчёта памяти и времени (на 1 GPU T4 16 ГБ)

1. **Статическая память модели (FP16 Mixed AdamW):**  
   Требует **16 байт на параметр** ($W_{\text{fp16}} = 2$, $G_{\text{fp16}} = 2$, $W_{\text{master\_fp32}} = 4$, $m_{\text{fp32}} = 4$, $v_{\text{fp32}} = 4$):
   - 48M (`Granite-48M`): $0.048\text{B} \times 16 = \mathbf{0.77\text{ ГБ}}$
   - 125M (`UniXcoder`): $0.125\text{B} \times 16 = \mathbf{2.00\text{ ГБ}}$
   - 149M (`ModernBERT`, `GTE-ModernBERT`): $0.149\text{B} \times 16 = \mathbf{2.38\text{ ГБ}}$
   - 352M (`Granite-4.0-350m`): $0.352\text{B} \times 16 = \mathbf{5.63\text{ ГБ}}$ (Full FT возможен на 16 ГБ)
   - 494M (`Qwen2.5-Coder-0.5B`): $0.494\text{B} \times 16 = \mathbf{7.90\text{ ГБ}}$ (Full FT на грани OOM с активациями)
   - 1.54B (`Qwen2.5-Coder-1.5B`): $1.54\text{B} \times 16 = \mathbf{24.64\text{ ГБ}}$ (**Full FT физически невозможен на 16 ГБ**, только LoRA)

2. **Время обучения:**  
   Датасет 22 000 примеров при эффективном батче 32 (8/GPU × 2 GPU × accum 2) = **688 шагов на эпоху**.  
   При sustained MFU 25–35% (35–45 TFLOPS на 2× T4):
   - Модель 149M @ 1024 токенов: $\approx 29.3\text{ TFLOP/шаг} \to 0.7–1.0\text{ с/шаг} \to \mathbf{8–11\text{ минут/эпоха}}$.
   - Модель 149M @ 2048 токенов (с GC): $\approx 1.2–1.6\text{ с/шаг} \to \mathbf{14–18\text{ минут/эпоха}}$.
   - Модель 1.5B LoRA FP16 @ 1024 токена: $\approx 2.6–3.8\text{ с/шаг} \to \mathbf{30–45\text{ минут/эпоха}}$.

---

## 3. Актуализированная Master-матрица кандидатов

Модели проверены по прямым спецификациям Hugging Face (`config.json`, `safetensors`; у `unixcoder-base`, `codebert-base`, `codereviewer` веса на HF доступны только в формате `pytorch_model.bin`), реестру `llm-checker` и независимым бенчмаркам.

### 3.1. Основной трек: Cross-Encoder DualHead (Двунаправленные энкодеры)

| Модель | Параметры (точно) | Архитектура и токенизатор | Контекст (макс/цель) | Режим на 2× T4 | VRAM (пик на карту) | Время 1 эпохи (22k) | Лицензия | Вердикт и статус в проекте |
|---|---|---|---|---|---|---|---|---|
| **`Alibaba-NLP/gte-modernbert-base`** (и **`gte-reranker-modernbert-base`**) | **149 014 272** (149M; reranker: 149.6M) | `ModernBertModel` / `ModernBertForSequenceClassification` (RoPE, GeGLU, vocab 50k, safetensors) | 8192 / **2048** (0.7% trunc) | **Full FT FP16** (+GC на 2048) | ~5.5–6.5 ГБ @1024; ~9–12 ГБ @2048 | **~8–11 мин @1024; ~14–18 мин @2048** | **Apache-2.0** | ⭐ **ТОП-1 (Цель M2 v2 в FD-24)**. Полная совместимость с ModernBERT; парный претрейн на текстовых relevance-парах (GTE/mGTE), код — только COIR-оценка (79.3/80.0); bi-encoder, готовая classification head — у `gte-reranker-modernbert-base` |
| **`answerdotai/ModernBERT-base`** | 149 655 232 (149M) | `ModernBertForMaskedLM` (RoPE, GeGLU, vocab 50k) | 8192 / **2048** (0.7% trunc) | Full FT FP16 (+GC на 2048) | ~5.5–6.5 ГБ @1024; ~9–12 ГБ @2048 | ~8–11 мин @1024; ~14–18 мин @2048 | Apache-2.0 | ⭐ **ТОП-2 (Контроль M2 v2 в FD-24)**. Чистый Masked LM бейзлайн |
| **`microsoft/unixcoder-base`** | ~125M | `RobertaModel` (BPE, vocab 51k, `pytorch_model.bin`) | 1026 / **1024** (4.4% trunc) | Full FT FP16 | ~4.5–5.5 ГБ | **~6–8 мин** | Apache-2.0 | ⭐ **ТОП-3 (Бейзлайн M2 v1 в FD-21)**. Стабильный чисто кодовый бейзлайн, штатный CLS-пулинг |
| **`ibm-granite/granite-embedding-english-r2`** | 149 014 272 (149M) | `ModernBertModel` (RoPE, GeGLU, vocab 50k) | 8192 / **2048** | Full FT FP16 | ~5.5–6.5 ГБ | ~8–11 мин | Apache-2.0 | ⭐ **ТОП-4 (Резерв ModernBERT)**. Архитектура ModernBERT, обученная на open-source permissive парах и IBM-internal парах технических доменов (код — COIR-оценка 55.3; явный кодовый претрейн заявлен лишь у multilingual-r2) |
| **`jinaai/jina-embeddings-v2-base-code`** | 160 869 120 (161M) | `JinaBertForMaskedLM` (BERT + ALiBi, vocab 61k) | 8192 / **2048** | Full FT FP16 | ~5.5–6.5 ГБ | ~8–10 мин | Apache-2.0 | ⭐ **ТОП-5 (Независимый длинный fallback)**. Специализация на коде, позиционные смещения ALiBi, чистая лицензия Apache-2.0 |
| `jinaai/jina-embeddings-v2-base-en` | 137 368 320 (137M) | `JinaBertForMaskedLM` (BERT + ALiBi, vocab 30k) | 8192 / **2048** | Full FT FP16 | ~4.5–5.5 ГБ | ~7–9 мин | Apache-2.0 | Общий резерв ALiBi |
| `ibm-granite/granite-embedding-small-english-r2` | 47 662 464 (48M) | `ModernBertModel` (RoPE, hidden 384, layers 12) | 8192 / 2048 | Full FT FP16 | ~2.5–3.5 ГБ | **~2–3 мин** | Apache-2.0 | Скоростной smoke-тест DDP пайплайна |
| `nomic-ai/modernbert-embed-base` | 149 014 272 (149M) | `ModernBertModel` (RoPE, GeGLU) | 8192 / 2048 | Full FT FP16 | ~5.5–6.5 ГБ | ~8–11 мин | Apache-2.0 | Аналог GTE-ModernBERT |
| `ibm-granite/granite-embedding-311m-multilingual-r2` | 311 664 384 (312M) | `ModernBertModel` (vocab 262k) | **32 768** / 2048 | Full FT FP16 + GC | ~10–12 ГБ | ~15–20 мин | Apache-2.0 | Резерв сверхдлинного контекста |
| `answerdotai/ModernBERT-large` | 395M | `ModernBertModel` (RoPE, GeGLU) | 8192 / 2048 | Full FT FP16 + GC | ~11–13 ГБ | ~18–24 мин | Apache-2.0 | Резерв высокой емкости (слот 9) |
| `microsoft/codebert-base` | 125M | `RobertaModel` | 512 (**28.6% trunc**) | Full FT FP16 | ~3.5–4.5 ГБ | ~3–4 мин | MIT | Устаревший: критическое усечение контекста |
| `microsoft/codereviewer` | 220M | T5 Seq2Seq (Diff-tokens) | 512 (ext 1024) | Full FT (Enc) / LoRA | ~7.0–8.5 ГБ | ~12–16 мин | Apache-2.0 | Diff-специалист, только как feature-extractor |

---

### 3.2. Факультативный трек E8: Causal Decoder & Hybrid Models

> [!WARNING] Causal-декодеры имеют структурную асимметрию внимания: токены кода не видят токены документации в обратном направлении без снятия каузальной маски. Инференс и обучение в 3–5 раз медленнее энкодеров. Допустимы только в рамках факультатива E8.

| Модель | Параметры (точно) | Архитектура | Контекст (макс/цель) | Режим на 2× T4 | VRAM (пик) | Время 1 эпохи | Лицензия | Вердикт для E8 |
|---|---|---|---|---|---|---|---|---|
| **`ibm-granite/granite-4.0-350m`** | **352 379 904** (352M) | `GraniteMoeHybridForCausalLM` (dense attention-only, `num_local_experts: 0`) | 32 768 / **1024** | **Full FT FP16 + GC** | **~8.0–9.5 ГБ** (@1024) | **~9–12 мин** (@1024) | Apache-2.0* | ⭐ **Приоритетный Causal-кандидат для E8**. Легковесный, dense-архитектура, Full FT на T4 без риска OOM при рабочей длине 1024 (*файл LICENSE в репо 404, сверить перед релизом; при L=32k неизбежен OOM) |
| **`Qwen/Qwen2.5-Coder-1.5B`** | 1 543 714 304 (1.54B) | `Qwen2ForCausalLM` | 32 768 / **1024** | LoRA FP16 ($r=16$) | ~8.5–10.0 ГБ | **~30–45 мин** | Apache-2.0 | Базовый семантический декодер E8 (требует 1.5–2 часа GPU на 3 эпохи) |
| `01-ai/Yi-Coder-1.5B` | 1 476 495 360 (1.48B) | `LlamaForCausalLM` | 131 072 / **1024** | LoRA FP16 ($r=16$) | ~8.5–10.0 ГБ | ~30–45 мин | Apache-2.0 | Прямая альтернатива Qwen 1.5B (HumanEval 67.7% Chat / 41.5% Base — метрики GitHub README для Python-подмножества) |
| `infly/OpenCoder-1.5B-Instruct` | 1 905 650 880 (1.91B) | `LlamaForCausalLM` | 4096 / **1024** | LoRA FP16 ($r=16$) | ~8.5–10.0 ГБ | ~30–45 мин | ⚠️ INF Custom (revocable, PRC) | Топ-1 в BigCodeBench ($\le 1.5$B). Коммерция разрешена («manufacture... sell»), но суды Шанхая и право КНР. Жёлтый статус $\triangle$ |
| `Qwen/Qwen2.5-Coder-0.5B` | 494 032 768 (494M) | `Qwen2ForCausalLM` | 32 768 / **1024** | LoRA FP16 / Full FT 8-bit Adam | ~10–12 ГБ | ~16–22 мин | Apache-2.0 | Causal-контроль с риском OOM при Full FT |
| `BAAI/bge-code-v1` | 1 543 301 120 (1.54B) | `Qwen2Model` | 32 768 / **1024** | LoRA FP16 | ~8.5–10.0 ГБ | ~30–45 мин | Apache-2.0 | Избыточен по массе для эмбеддера на T4 |

---

### 3.3. Дисквалифицированные модели

| Модель | Параметры | Причина дисквалификации |
|---|---|---|
| `Qwen/Qwen2.5-Coder-3B` | 3.09B | ❌ **Лицензия `Qwen Research License`** (в отличие от 0.5B/1.5B/7B, размер 3B выпущен под несвободной лицензией) + двукратное время обучения по сравнению с 1.5B. |
| `jinaai/jina-embeddings-v3` | 572M | ❌ **Лицензия CC-BY-NC-4.0** (строго некоммерческая) + избыточный размер для энкодера на T4. |
| `Qwen/Qwen2.5-Coder-7B` | 7.61B | ❌ QLoRA INT4 на архитектуре Turing программная (нет нативных тензорных ядер INT4) $\to$ скорость инференса >60 мин/эпоха, исчерпание недельной квоты 30 ч. |
| `DeepSeek-Coder-V2-Lite` | 16B (2.4B act) | ❌ Архитектура Multi-Head Latent Attention (MLA) вызывает CUDA-сбои и OOM на драйверах Turing 16 ГБ без FlashAttention-2. |
| `microsoft/graphcodebert-base` | 125M | ❌ Ограничение 512 токенов (28.6% потерь) + поломка dataflow-графа на синтаксисе git-диффа (`+`/`-`). |
| `allenai/longformer-base-4096` | 149M | ❌ Sliding-window attention: токены docstring не взаимодействуют с токенами диффа через локальные окна без глобального внимания, что вырождает Cross-Encoder. |

---

## 4. Архитектурный норматив DualHead (согласовано с R10)

- **Пулинг по умолчанию (энкодеры):** `[CLS]`. Пулинг `Masked Mean` допустим только как абляция E3 (слот 9).
- **Пулинг для каузальных декодеров (факультатив E8):** `Last Non-Pad` (последний непаддинговый токен последовательности). В каузальной архитектуре нулевой токен не агрегирует последующий контекст из-за треугольной маски внимания; полный receptive field всей пары имеет только последний токен.
- **Критерий потерь:** Masked Conditional Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(z_{\text{stale}}, y_{\text{stale}}) + \lambda \cdot \mathbb{I}(y_{\text{stale}} = 1) \cdot \mathcal{L}_{\text{CE}}(z_{\text{drift}}, y_{\text{drift}})$$
  где для строк с $y_{\text{stale}} = 0$ метка дрейфа маскируется (`ignore_index = -100`), а $\lambda \in \{0.3, 0.6, 1.0\}$ подбирается по валидации (стартовое значение: $\lambda = 0.6$).
- **Гейт инференса:** Если $p_{\text{stale}} \le \tau$ (порог из FD-25), инференс возвращает `stale = 0, drift_type = None`, не опрашивая голову типов дрейфа.
- **Метрика дрейфа:** Macro-F1 головы типов дрейфа оценивается **строго на подвыборке истинного рассинхрона ($y_{\text{stale}} = 1$)**.

```python
import torch
import torch.nn as nn

class DualHeadCodeClassifier(nn.Module):
    def __init__(self, backbone, hidden_size=768, num_drift_classes=8, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.stale_head = nn.Linear(hidden_size, 1)
        self.drift_head = nn.Linear(hidden_size, num_drift_classes)
        self.stale_criterion = nn.BCEWithLogitsLoss()
        self.drift_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels_stale=None, labels_drift=None, lambda_drift=0.6):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Штатный CLS-пулинг первого токена
        cls_rep = outputs.last_hidden_state[:, 0, :]
        pooled = self.layer_norm(self.dropout(cls_rep))
        
        logit_s = self.stale_head(pooled).squeeze(-1)
        logit_d = self.drift_head(pooled)
        
        loss = None
        if labels_stale is not None:
            loss_s = self.stale_criterion(logit_s, labels_stale.float())
            if labels_drift is not None:
                mask_d = labels_drift.clone()
                mask_d[labels_stale == 0] = -100
                loss = loss_s + lambda_drift * self.drift_criterion(logit_d, mask_d)
            else:
                loss = loss_s
                
        return {"loss": loss, "logits_stale": logit_s, "logits_drift": logit_d}
```

---

## 5. Итоговый ранжированный ТОП моделей для FreshDoc

1. **ТОП-1 (Основная цель улучшения M2 v2, FD-24): `Alibaba-NLP/gte-modernbert-base` (149M)**  
   *Обоснование:* Нативный контекст до 8192 (0.7% усечения при $L=2048$), современная архитектура RoPE/GeGLU/unpadding, полная совместимость с `ModernBertForSequenceClassification`. Предобучен на текстовых relevance-парах (GTE/mGTE), валидирован на коде (COIR 79.3/80.0); bi-encoder архитектура, а готовая sequence classification head присутствует в чекпоинте `gte-reranker-modernbert-base`; целевой бэкбон под DualHead Cross-Encoder.
2. **ТОП-2 (Факториальный контроль M2 v2, FD-24): `answerdotai/ModernBERT-base` (149M)**  
   *Обоснование:* Необходим для чистого экспериментального разделения: сколько прироста дает снятие усечения контекста ($1024 \to 2048$), а сколько — парный претрейн весов GTE.
3. **ТОП-3 (Обязательный базовый бейзлайн M2 v1, FD-21): `microsoft/unixcoder-base` (125M)**  
   *Обоснование:* Чисто кодовый энкодер с рабочим окном 1024 (4.4% потерь), проверенная сходимость, 6–8 мин на эпоху.
4. **ТОП-4 (Резерв ModernBERT): `ibm-granite/granite-embedding-english-r2` (149M)**  
   *Обоснование:* Прямой бэкбон ModernBERT с 8192 контекстом, обученный на open-source permissive парах и IBM-internal парах технических доменов (код — COIR-оценка 55.3; явный кодовый претрейн заявлен лишь у multilingual-r2).
5. **ТОП-5 (Независимый длинный fallback без ModernBERT): `jinaai/jina-embeddings-v2-base-code` (161M)**  
   *Обоснование:* Архитектура BERT с ALiBi-позиционными смещениями на 8192 токена, специализированная на коде, с лицензией Apache-2.0. Страховка на случай непредвиденных проблем с рантаймом ModernBERT.
6. **ТОП-6 (Инфраструктурный контроль и smoke-тесты): `ibm-granite/granite-embedding-small-english-r2` (48M)**  
   *Обоснование:* 48M параметров, 8k контекст, эпоха за 2–3 минуты. Используется для верификации DDP и пайплайнов логов без расхода квоты.
7. **ТОП-7 (Causal-контроль E8): `ibm-granite/granite-4.0-350m` (352M)**  
   *Обоснование:* Единственный декодер, позволяющий провести честный Full FT FP16 на 16 ГБ T4 за 9–12 минут на эпоху при $L=1024$ без риска CUDA OOM (сконфигурирован с `num_local_experts: 0` как dense attention-only; файл LICENSE в репозитории на HF дает 404, перед релизом перепроверить).
8. **ТОП-8 (Causal семантический эталон E8): `Qwen/Qwen2.5-Coder-1.5B` / `01-ai/Yi-Coder-1.5B` (1.5B)**  
   *Обоснование:* Сильнейшие открытые Causal-модели в классе 1.5B с Apache-2.0 для оценки теоретического потолка генеративных архитектур в режиме LoRA (значения HumanEval 67.7%/41.5% у Yi-Coder взяты из GitHub README для multilingual Python-подмножества).

---

## 6. Kaggle DDP-рецепт для 2× Tesla T4

- **Библиотеки:** `transformers>=4.48.0` (критично для ModernBERT/GTE), `torch>=2.2.0`, `accelerate>=0.34.0`.
- **DDP флаги окружения:** `export NCCL_P2P_DISABLE=1` (на шине PCIe без NVLink предотвращает зависания peer-to-peer allreduce).
- **Связка памяти:** FP16 mixed (`fp16=True`, `bf16=False`), Gradient Checkpointing активен при $L \ge 2048$.
- **Батчинг:** `per_device_batch_size = 8`, `gradient_accumulation_steps = 2`, эффективный размер батча = 32.
- **Оптимизатор:** `AdamW`, $\text{lr} = 2\cdot 10^{-5}$ для бэкбона, $3\cdot 10^{-5}$ для голов классификатора, `weight_decay = 0.01`, линейный warmup 6–10%, `max_grad_norm = 1.0`.
- **Чекпоинты:** Сохранение на каждой эпохе с ротацией `save_total_limit = 2` для защиты от разрыва 12-часовой сессии Kaggle.

---

## 7. Верифицированный реестр лицензий

| Лицензия | Модели | Статус для проекта |
|---|---|---|
| **Apache-2.0 (чистая, ungated)** | `Alibaba-NLP/gte-modernbert-base`, `answerdotai/ModernBERT-base`, `microsoft/unixcoder-base`, `ibm-granite/granite-embedding-*`, `jinaai/jina-embeddings-v2-*`, `ibm-granite/granite-4.0-350m` (*файл LICENSE в репо 404, в метаданных указан apache-2.0), `01-ai/Yi-Coder-1.5B`, `Qwen/Qwen2.5-Coder-0.5B`, `Qwen/Qwen2.5-Coder-1.5B`, `BAAI/bge-code-v1` | ✅ **Полная свобода использования, модификации и встраивания в продукт** |
| **MIT** | `microsoft/codebert-base`, `microsoft/graphcodebert-base` (*на HF лицензия не указана в карточке, MIT подтвержден только по upstream-репозиторию CodeXGLUE) | ✅ Разрешено |
| **INF Custom (revocable, PRC)** | `infly/OpenCoder-1.5B-Instruct` | ⚠️ **Коммерция формально разрешена** («manufacture... sell... offer for sale»), но действуют строгие ограничения: право КНР, исключительная юрисдикция судов Шанхая. Жёлтый статус $\triangle$ |
| **Qwen Research License** | `Qwen/Qwen2.5-Coder-3B` | ⚠️ Только некоммерческие исследования. Не брать в продакшн-трек |
| **CC-BY-NC-4.0** | `jinaai/jina-embeddings-v3`, `jina-reranker-m0` | ❌ Некоммерческая лицензия |
| **DeepSeek License** | `deepseek-coder-1.3b` | ⚠️ Non-OSI, допустимо с оговорками |
