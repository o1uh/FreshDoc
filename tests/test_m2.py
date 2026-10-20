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
        # (hash() запрещён: соль PYTHONHASHSEED делает тесты flaky)
        import zlib
        out = []
        for w in str(text).split():
            out.append(10 + (zlib.crc32(w.encode("utf-8")) % 1000))
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
    # (структурно, а не count(): id токенов слов могут случайно совпасть со спецтокенами)
    assert er["input_ids"][0] == 0 and er["input_ids"][-1] == 2
    assert er["input_ids"][er["n_a_tok"] + 1:er["n_a_tok"] + 3] == [2, 2]
    assert em["input_ids"][0] == 101 and em["input_ids"][-1] == 102
    assert em["input_ids"][em["n_a_tok"] + 1] == 102
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


def test_m2_data_collator_picklable():
    """Регрессия блокера: коллатор обязан пережить pickle (num_workers>0)."""
    import pickle

    from scripts.train_m2 import M2DataCollator, m2_collator
    col = M2DataCollator(_FakeTok())
    col2 = pickle.loads(pickle.dumps(col))
    assert col2.pad_id == 1
    # фабрика-обёртка возвращает тот же picklable класс
    assert isinstance(m2_collator(_FakeTok()), M2DataCollator)
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1]),
         "labels_stale": torch.tensor(1), "labels_drift": torch.tensor(0),
         "sample_weight": torch.tensor(1.0)},
        {"input_ids": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1]),
         "labels_stale": torch.tensor(0), "labels_drift": torch.tensor(-100),
         "sample_weight": torch.tensor(0.5)},
    ]
    out = col2(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1, 2].item() == 1  # pad_id _FakeTok
    assert out["attention_mask"][1, 2].item() == 0
    assert out["labels_stale"].tolist() == [1, 0]


def test_is_sig_hunk_strict_def_start():
    # "def" внутри строки/комментария удалённой строки — не сигнатура (minor 7)
    assert not is_sig_hunk("@@ -1 +1 @@\n-    s = \"def foo\"\n+    y = 1")
    assert not is_sig_hunk("@@ -1 +1 @@\n-    # def helper\n+    pass")
    assert not is_sig_hunk("@@ -1 +1 @@\n-    body = _get(url)\n+    body_tmp = _get(url)")
    # настоящие сигнатурные правки — сигнатура
    assert is_sig_hunk("@@ -1 +1 @@\n-def f(a):\n+def f(b):")
    assert is_sig_hunk("@@ -1 +1 @@\n+async def f(a):\n+    pass")
    assert is_sig_hunk("@@ -1 +1 @@\n-async def f(a):\n+async def g(a):")


def test_encode_pair_doc_truncated_flag():
    # S1: обрезка B считается отдельно — раньше молча терялся хвост Returns/Raises
    tok = _FakeTok()
    short = encode_pair(tok, "def f(a):", "-x\n+y", "Short docstring here.",
                        max_len=64)
    assert short["doc_truncated"] is False
    long_doc = " ".join(f"word{i}" for i in range(500))
    long = encode_pair(tok, "def f(a):", "-x\n+y", long_doc, max_len=256)
    assert long["doc_truncated"] is True
    assert long["n_b_tok"] == 192  # кап B по R10
    assert len(long["input_ids"]) == 1 + long["n_a_tok"] + 2 + 192 + 1


def test_resolve_threshold_priority(tmp_path):
    # Hermetic: не зависит от репозиторного config/risk.yaml
    # (сломается на FD-25, когда туда ляжет рабочий порог).
    import scripts.predict_m2 as pm
    from pathlib import Path
    rp = tmp_path / "risk.yaml"
    rp.write_text("threshold: 0.8\n", encoding="utf-8")
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert pm.resolve_threshold(0.3, risk_path=rp) == 0.3
    assert pm.resolve_threshold(None, risk_path=rp) == 0.8
    assert pm.resolve_threshold(0.2, risk_path=rp) == 0.2
    assert pm.resolve_threshold(None, risk_path=empty) == 0.5  # пустой файл -> дефолт
    assert pm.resolve_threshold(None, risk_path=tmp_path / "nope.yaml") == 0.5


def test_predict_main_end_to_end_fake_backbone(tmp_path, monkeypatch, capsys):
    """Регрессия B1/2.3: полный проход predict_m2 (раньше падал NameError len(sub))."""
    import sys

    import pandas as pd
    from safetensors.torch import save_file

    import scripts.predict_m2 as pm
    from src.models.m2_model import DualHeadCodeClassifier

    torch.manual_seed(0)

    df = pd.DataFrame({
        "pair_id": ["p1", "p2", "p3"],
        "sig_old": ["def f(a):", "def g(x):", "def h():"],
        "diff_code": ["-a\n+b", "-x\n+y", "-m\n+n"],
        "doc_old": ["Doc one here.", "Doc two here.", "Doc three here."],
        "label_stale": [1, 0, 1],
        "drift_type": ["param_renamed", "none", "param_added"],
        "weight": [1.0, 0.5, 1.0],
        "split": ["test_real", "test_real", "gold"],
        "source": ["codocbench", "history_neg", "gold"],
    })
    ds_path = tmp_path / "mini.parquet"
    df.to_parquet(ds_path, index=False)

    backbone = _TinyBackbone(hidden=16)
    model = DualHeadCodeClassifier(backbone, hidden_size=16, num_drift_classes=8)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file(model.state_dict(), ckpt / "model.safetensors")

    monkeypatch.setattr(pm.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _FakeTok()))
    monkeypatch.setattr(pm, "AutoModel", _FakeAutoModelFactory)
    out_path = tmp_path / "m2.csv"
    monkeypatch.setattr(sys, "argv", ["predict_m2.py", "--checkpoint", str(ckpt),
                                     "--dataset", str(ds_path),
                                     "--out", str(out_path),
                                     "--batch_size", "2"])
    assert pm.main() == 0
    got = pd.read_csv(out_path)
    assert len(got) == 3
    assert set(got.columns) >= {"pair_id", "p_stale", "type_pred", "type_pred_gated"}
    assert ((got["p_stale"] >= 0.0) & (got["p_stale"] <= 1.0)).all()


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


def test_stale_zero_with_known_drift_still_masked():
    """y=0 с известным типом дрейфа (как 76 строк gold): голова дрейфа
    не учится — градиент нулевой, но граф жив (защита от 0.0*inf=NaN)."""
    torch.manual_seed(0)
    model = DualHeadCodeClassifier(_FakeBackbone(), hidden_size=16,
                                   num_drift_classes=8)
    ids = torch.randint(0, 50, (4, 8))
    mask = torch.ones(4, 8, dtype=torch.long)
    stale = torch.tensor([0, 0, 0, 0])
    drift = torch.tensor([0, 2, 5, 1])  # валидные id, но y=0 -> маска
    out = model(input_ids=ids, attention_mask=mask, labels_stale=stale,
                labels_drift=drift)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.drift_head.weight.grad is not None
    assert torch.allclose(model.drift_head.weight.grad,
                          torch.zeros_like(model.drift_head.weight.grad))


def test_training_args_warmup_steps_ratio():
    """transformers>=5.19: warmup_ratio удалён, warmup_steps<1 — доля."""
    from transformers import TrainingArguments
    import inspect
    assert "warmup_ratio" not in inspect.signature(TrainingArguments.__init__).parameters
    a = TrainingArguments(output_dir="checkpoints/t", warmup_steps=0.06,
                          report_to="none")
    # 6% от эпохи 688 шагов = 42 шага разогрева
    assert a.get_warmup_steps(688) == 42


def test_append_runs_csv_conforms_to_existing_header(tmp_path):
    # N2: писатель FD-21 не должен разъезжать CSV форматом FD-24.
    import csv

    from scripts.train_m2 import append_runs_csv
    p = tmp_path / "runs.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "model_name", "extra_fd24"])
        w.writeheader()
        w.writerow({"timestamp": "t0", "model_name": "m0", "extra_fd24": "x"})
    append_runs_csv(p, {"timestamp": "t1", "model_name": "m1", "seed": 42,
                        "eval_pr_auc": 0.9})
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert len(rows) == 2
    assert set(rows[1].keys()) == {"timestamp", "model_name", "extra_fd24"}
    assert rows[1] == {"timestamp": "t1", "model_name": "m1", "extra_fd24": ""}
    # нового файла: заголовок из строки
    p2 = tmp_path / "new.csv"
    append_runs_csv(p2, {"a": 1})
    assert list(csv.DictReader(p2.open(encoding="utf-8")))[0] == {"a": "1"}


def test_check_input_capacity():
    # Fail-fast: max_len сверх позиционной матрицы бэкбона (поймано дымовым
    # тестом: tiny-RoBERTa 512 + max_len=1024 -> index out of bounds).
    import pytest

    from src.models.m2_model import (backbone_position_capacity,
                                     check_input_capacity)

    class _B(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.embeddings = torch.nn.Module()
            if n is not None:
                self.embeddings.position_embeddings = torch.nn.Embedding(n, 4)

    assert backbone_position_capacity(_B(512)) == 512
    assert backbone_position_capacity(_B(1026)) == 1026
    assert backbone_position_capacity(_B(None)) is None  # RoPE/ALiBi -> пропуск
    check_input_capacity(_B(1026), 1024, "unixcoder")  # FD-21: ок
    with pytest.raises(ValueError):
        check_input_capacity(_B(512), 1024, "tiny")


def test_train_saves_guarded_by_is_main():
    """Регрессия S5/2.1: на rank 1 (is_main False) не должно быть ни записей,
    ни NameError. Все использования best_dir в main() обязаны лежать
    внутри `if is_main:` (иначе UnboundLocalError роняет torchrun-ноду)."""
    import ast
    from pathlib import Path

    from scripts import train_m2

    src = Path(train_m2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")

    class GuardVisit(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[ast.AST] = []
            self.loads_outside: list[int] = []
            self.assign_in_guard = False

        def _in_is_main(self) -> bool:
            return any(isinstance(n, ast.If)
                       and isinstance(n.test, ast.Name)
                       and n.test.id == "is_main"
                       for n in self.stack)

        def visit_Name(self, node: ast.Name):
            if node.id == "best_dir" and isinstance(node.ctx, ast.Load):
                if not self._in_is_main():
                    self.loads_outside.append(node.lineno)
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign):
            targets = [t.id for t in ast.walk(node)
                       if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)]
            if "best_dir" in targets and self._in_is_main():
                self.assign_in_guard = True
            self.generic_visit(node)

        def generic_visit(self, node):
            self.stack.append(node)
            super().generic_visit(node)
            self.stack.pop()

    v = GuardVisit()
    v.visit(main_fn)
    assert v.assign_in_guard, "best_dir должен создаваться внутри if is_main:"
    assert not v.loads_outside, \
        f"best_dir используется вне if is_main: (строки {v.loads_outside})"


class _FakeBackbone(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        b, n = input_ids.shape
        hs = torch.randn(b, n, 16)
        return type("O", (), {"last_hidden_state": hs})()


class _TinyBackbone(torch.nn.Module):
    """Стебовый бэкбон фиксированной ширины для сквозного теста инференса."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self._hidden = hidden

    def forward(self, input_ids=None, attention_mask=None):
        b, n = input_ids.shape
        hs = torch.randn(b, n, self._hidden)
        return type("O", (), {"last_hidden_state": hs})()


class _FakeAutoModelFactory:
    """Замена AutoModel.from_pretrained/from_config в тесте инференса."""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        return _TinyBackbone(hidden=16)

    @staticmethod
    def from_config(*args, **kwargs):
        return _TinyBackbone(hidden=16)
