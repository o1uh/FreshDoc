"""FD-21: тесты входа R10, DualHead-лосса и гейта (без сети, CPU)."""
from __future__ import annotations

import torch

from src.models.m2_data import (DRIFT_CLASSES, DRIFT_TO_ID, UNKNOWN_ID,
                                drift_to_id, encode_pair, is_sig_hunk,
                                n_special_tokens, prioritize_diff,
                                split_hunks)
from src.models.m2_model import DualHeadCodeClassifier


class _FakeTok:
    cls_token = "<s>"
    sep_token = "</s>"
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def __init__(self):
        self._v = {"<s>": 0, "<pad>": 1, "</s>": 2}

    def encode(self, text, add_special_tokens=False):
        # детерминированное псевдо-токенизирование по словам
        out = []
        for w in str(text).split():
            out.append(10 + (abs(hash(w)) % 1000))
        return out


class _FakeTokModern(_FakeTok):
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    cls_token_id = 101
    sep_token_id = 102


def test_drift_classes_fixed_order():
    assert DRIFT_CLASSES == ["param_renamed", "param_removed", "param_added",
                             "default_changed", "return_changed",
                             "exception_changed", "behavior_changed", "unknown"]
    assert drift_to_id("none") == -100
    assert drift_to_id("") == -100
    # unknown — резерв без супервизии: в обучение не идёт (маска -100),
    # индекс 7 используется только для декодирования argmax инференса
    assert drift_to_id("unknown") == -100
    assert DRIFT_TO_ID["unknown"] == UNKNOWN_ID == 7
    assert drift_to_id("param_added") == 2
    assert drift_to_id("behavior_changed") == 6


def test_sig_hunk_priority():
    diff = ("--- a/code.py\n+++ b/code.py\n"
            "@@ -1,3 +1,3 @@\n-    body = _get(url)\n+    body_tmp = _get(url)\n"
            "@@ -10,3 +10,3 @@\n-def fetch(url, timeout=10):\n+def fetch(endpoint, timeout=10):\n")
    hunks = split_hunks(diff)
    assert len(hunks) == 2
    assert is_sig_hunk(hunks[1]) and not is_sig_hunk(hunks[0])
    pr = prioritize_diff(diff)
    assert pr.index("def fetch(endpoint") < pr.index("body_tmp")


def test_encode_roberta_vs_modernbert_specials():
    tok_r, tok_m = _FakeTok(), _FakeTokModern()
    assert n_special_tokens(tok_r) == 4
    assert n_special_tokens(tok_m) == 3
    er = encode_pair(tok_r, "def f(a):", "-x\n+y", "Docstring here.", max_len=64)
    em = encode_pair(tok_m, "def f(a):", "-x\n+y", "Docstring here.", max_len=64)
    # RoBERTa: <s> A </s></s> B </s> ; ModernBERT: [CLS] A [SEP] B [SEP]
    assert er["input_ids"][0] == 0 and er["input_ids"].count(2) == 3
    assert em["input_ids"][0] == 101 and em["input_ids"].count(102) == 2
    assert not er["truncated"] and not em["truncated"]


def test_encode_truncation_flag_and_budget():
    tok = _FakeTok()
    long_diff = "\n".join(f"+line_{i} change here" for i in range(500))
    e = encode_pair(tok, "def f(a):", long_diff, "Docstring.", max_len=32)
    assert e["truncated"] is True
    assert len(e["input_ids"]) <= 32


def test_dualhead_loss_masking():
    torch.manual_seed(0)
    backbone = _FakeBackbone()
    model = DualHeadCodeClassifier(backbone, hidden_size=16,
                                   num_drift_classes=8)
    b = 4
    ids = torch.randint(0, 50, (b, 8))
    mask = torch.ones(b, 8, dtype=torch.long)
    stale = torch.tensor([1, 1, 0, 1])
    # R10: y=0 и unknown маскируются (-100); валидны только idx0 и idx3
    drift = torch.tensor([0, -100, 0, 2])
    out = model(input_ids=ids, attention_mask=mask, labels_stale=stale,
                labels_drift=drift)
    assert out["loss"] is not None and torch.isfinite(out["loss"])
    out["loss"].backward()
    # обе головы получили градиент (валидные строки есть)
    assert model.stale_head.weight.grad is not None
    assert model.drift_head.weight.grad is not None
    assert torch.isfinite(model.stale_head.weight.grad).all()
    assert torch.isfinite(model.drift_head.weight.grad).all()


def test_dualhead_all_masked_keeps_graph():
    """Пустой батч для головы дрейфа: лосс нулевой, но с графом (DDP не падает)."""
    torch.manual_seed(0)
    model = DualHeadCodeClassifier(_FakeBackbone(), hidden_size=16,
                                   num_drift_classes=8)
    ids = torch.randint(0, 50, (4, 8))
    mask = torch.ones(4, 8, dtype=torch.long)
    stale = torch.tensor([0, 0, 1, 1])
    drift = torch.tensor([-100, -100, -100, -100])
    out = model(input_ids=ids, attention_mask=mask, labels_stale=stale,
                labels_drift=drift)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    # градиент drift_head существует (нулевой путь через logit_d.sum()*0.0),
    # что держит DDP-редукцию даже при find_unused_parameters=True/False
    assert model.drift_head.weight.grad is not None


def test_dualhead_gated_inference():
    torch.manual_seed(1)
    backbone = _FakeBackbone()
    model = DualHeadCodeClassifier(backbone, hidden_size=16)
    model.eval()
    ids = torch.randint(0, 50, (3, 8))
    mask = torch.ones(3, 8, dtype=torch.long)
    out = model.predict_gated(input_ids=ids, attention_mask=mask, threshold=0.99)
    # при высоком пороге почти всё gated=-1 (синхронизация)
    assert out["drift_gated"].shape == (3,)
    assert set(out["drift_gated"].tolist()) <= set(list(range(8)) + [-1])


class _FakeBackbone(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        b, n = input_ids.shape
        hs = torch.randn(b, n, 16)
        return type("O", (), {"last_hidden_state": hs})()
