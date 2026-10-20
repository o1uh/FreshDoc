"""FD-21: двухголовая модель M2 (R10, rev.8 model_strategy).

Архитектура: backbone AutoModel + CLS-пулинг первого токена + LayerNorm +
Dropout(0.1) + две линейные головы:
- stale_head: 1 выход (BCEWithLogits)
- drift_head: 8 выходов (CE, ignore_index=-100)

Классы дрейфа (фиксированы): param_renamed, param_removed, param_added,
default_changed, return_changed, exception_changed, behavior_changed,
unknown (резерв без супервизии). Класса none в голове нет: синхронизацию
обслуживает гейт p_stale > tau.

Лосс: L = BCE(веса примеров) + 0.6 * CE(маска y_stale=1 и тип известен).
Строки y_stale=0 и drift unknown/none маскируются (-100).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.m2_data import DRIFT_CLASSES

LAMBDA_DRIFT = 0.6


class DualHeadCodeClassifier(nn.Module):
    def __init__(self, backbone, hidden_size: int = 768, num_drift_classes: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.stale_head = nn.Linear(hidden_size, 1)
        self.drift_head = nn.Linear(hidden_size, num_drift_classes)
        self.stale_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.drift_criterion = nn.CrossEntropyLoss(reduction="none",
                                                   ignore_index=-100)
        self.num_drift_classes = num_drift_classes

    def forward(self, input_ids=None, attention_mask=None, labels_stale=None,
                labels_drift=None, sample_weight=None, lambda_drift: float = LAMBDA_DRIFT):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        pooled = self.layer_norm(self.dropout(cls_rep))
        logit_s = self.stale_head(pooled).squeeze(-1)
        logit_d = self.drift_head(pooled)
        loss = None
        if labels_stale is not None:
            y = labels_stale.float()
            raw_s = self.stale_criterion(logit_s, y)
            if sample_weight is not None:
                w = sample_weight.float()
                loss_s = (raw_s * w).sum() / w.sum().clamp_min(1e-8)
            else:
                loss_s = raw_s.mean()
            loss = loss_s
            if labels_drift is not None:
                mask_d = labels_drift.clone()
                # маскируются строки без рассинхрона и с неизвестным типом
                # (unknown закодирован как -100 в drift_to_id, резерв без супервизии)
                invalid = (labels_stale == 0) | (labels_drift == -100)
                # CrossEntropy сам игнорит -100, но для взвешивания строим маску
                raw_d = self.drift_criterion(logit_d, mask_d)
                valid = (~invalid).float()
                if valid.sum() > 0:
                    if sample_weight is not None:
                        w = sample_weight.float() * valid
                        # raw_d на invalid позициях = 0 (ignore), вес 0
                        loss_d = (raw_d * w).sum() / w.sum().clamp_min(1e-8)
                    else:
                        loss_d = (raw_d * valid).sum() / valid.sum().clamp_min(1e-8)
                else:
                    # пустой батч для головы дрейфа: нулевой лосс, но с графом,
                    # иначе DDP с find_unused_parameters=False падает
                    # (drift_head не участвует в backward шага)
                    loss_d = logit_d.sum() * 0.0
                loss = loss + lambda_drift * loss_d
        return {"loss": loss, "logits_stale": logit_s, "logits_drift": logit_d}

    @torch.no_grad()
    def predict_gated(self, input_ids=None, attention_mask=None, threshold: float = 0.5):
        """Гейт инференса R10: тип отдаётся только если p_stale > tau."""
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        p = torch.sigmoid(out["logits_stale"])
        drift_id = out["logits_drift"].argmax(dim=-1)
        gated = torch.where(p > threshold, drift_id, torch.full_like(drift_id, -1))
        return {"p_stale": p, "drift_id": drift_id, "drift_gated": gated}
