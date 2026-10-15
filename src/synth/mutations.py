from __future__ import annotations

import ast
import copy
import random
import re
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Общие пулы (R6: разнообразие + одинаковые стили имён у pos/neg)
# ---------------------------------------------------------------------------

def _style_suffix_tmp(n: str) -> str:
    return f"{n}_tmp"


def _style_suffix_new(n: str) -> str:
    return f"{n}_new"


def _style_suffix_v2(n: str) -> str:
    return f"{n}_v2"


def _style_prefix_new(n: str) -> str:
    return f"new_{n}"


def _style_prefix_my(n: str) -> str:
    return f"my_{n}"


def _style_suffix_arg(n: str) -> str:
    return f"{n}_arg"


def _style_suffix_param(n: str) -> str:
    return f"{n}_param"


def _style_suffix_val(n: str) -> str:
    return f"{n}_val"


def _style_prefix_tmp(n: str) -> str:
    return f"tmp_{n}"


def _style_suffix_2(n: str) -> str:
    return f"{n}2"


# Ровно 10 стилей; используются и в param_renamed, и в rename_local (анти-шорткат).
NAME_STYLES: list[Callable[[str], str]] = [
    _style_suffix_tmp,
    _style_suffix_new,
    _style_suffix_v2,
    _style_prefix_new,
    _style_prefix_my,
    _style_suffix_arg,
    _style_suffix_param,
    _style_suffix_val,
    _style_prefix_tmp,
    _style_suffix_2,
]

# Синонимы для частых имён (если есть — предпочитаются стилям).
SYNONYMS: dict[str, list[str]] = {
    "path": ["file_path", "fpath"],
    "url": ["endpoint", "address"],
    "timeout": ["timeout_seconds", "timeout_ms"],
    "retries": ["max_retries", "retry_count"],
    "verbose": ["verbosity", "chatty"],
    "limit": ["max_limit", "cap"],
    "symbol": ["ticker", "sym"],
    "params": ["options", "kwargs_extra"],
    "data": ["payload", "content"],
    "value": ["val", "item_value"],
    "body": ["payload_body", "content_body"],
    "result": ["result_value", "out"],
    "response": ["reply", "resp"],
    "request": ["req", "call_request"],
    "market": ["market_info", "mkt"],
    "query": ["query_params", "q"],
}

# 12 вариантов новых параметров (имя, код дефолта).
NEW_PARAMS_12: list[tuple[str, str]] = [
    ("strict", "False"),
    ("timeout", "30.0"),
    ("encoding", "'utf-8'"),
    ("max_items", "100"),
    ("sep", "','"),
    ("limit_", "50"),
    ("verbose", "True"),
    ("retries", "5"),
    ("delay", "1.0"),
    ("mode", "'strict'"),
    ("count", "10"),
    ("recursive", "False"),
]

TEMP_NAMES = ["result_value", "tmp", "out", "res", "ret", "value_out", "final_result", "buf"]

EXC_REPLACE: dict[str, str] = {
    "ValueError": "TypeError",
    "TypeError": "ValueError",
    "KeyError": "LookupError",
    "LookupError": "KeyError",
    "RuntimeError": "ValueError",
    "OSError": "IOError",
    "IOError": "OSError",
}

EXC_MESSAGES = [
    "invalid value",
    "bad argument",
    "empty input not allowed",
    "out of range",
    "unsupported option",
    "malformed input",
]

STR_DEFAULT_REPLACE: dict[str, str] = {
    "utf-8": "latin-1",
    "strict": "lenient",
    "asc": "desc",
    "GET": "POST",
}

SKIP_NAMES = {"self", "cls"}


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _all_param_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    out: set[str] = set()
    a = fn.args
    for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
        out.add(arg.arg)
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    return out


def _regular_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Позиционные + kwonly (без self/cls, без *args/**kwargs)."""
    a = fn.args
    res = [x for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
           if x.arg not in SKIP_NAMES]
    return res


def _mentioned_as_word(name: str, doc: str) -> bool:
    return re.search(r"\b" + re.escape(name) + r"\b", doc or "") is not None


def _documented_as_param(name: str, doc: str) -> bool:
    # Параметр задокументирован как аргумент (а не просто упомянут в прозе):
    # Sphinx-стиль (:param [type] name:) или numpydoc/Google-стиль (name : ...).
    n = re.escape(name)
    d = doc or ""
    if re.search(r":(?:param|parameter|arg|argument|keyword|key)\s+(?:\S+\s+)?\[?" + n + r"\]?\s*:", d):
        return True
    return re.search(r"(?m)^\s*[-*]?\s*" + n + r"\s*[:(\[]", d) is not None


def _has_hidden_binding(fn, name: str) -> bool:
    # Привязки имени, которые плоский обход ast.walk не переименовывает:
    # плоское переименование их пропустило бы и сломало код (затенение).
    for n in ast.walk(fn):
        if isinstance(n, ast.ExceptHandler) and n.name == name:
            return True
        if isinstance(n, ast.MatchAs) and n.name == name:
            return True
        if isinstance(n, ast.MatchMapping) and n.rest == name:
            return True
        if isinstance(n, ast.MatchStar) and n.name == name:
            return True
        if isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            return True
        if isinstance(n, ast.alias) and (n.name == name or n.asname == name):
            return True
        if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n is not fn and n.name == name):
            return True
    return False


def _walk_with_parent(node, parent=None):
    yield parent, node
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_parent(child, node)


def _existing_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = _all_param_names(fn)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            names.add(n.id)
    return names


def _fresh_name(base: str, taken: set[str], rng: random.Random) -> Optional[str]:
    # Сначала синоним, потом случайный стиль (до 10 попыток).
    if base in SYNONYMS:
        for cand in SYNONYMS[base]:
            if cand not in taken and cand.isidentifier():
                return cand
    order = list(range(len(NAME_STYLES)))
    rng.shuffle(order)
    for i in order:
        cand = NAME_STYLES[i](base)
        if cand not in taken and cand.isidentifier():
            return cand
    return None


def _annotation_for_default(default_code: str) -> Optional[ast.expr]:
    try:
        v = ast.parse(default_code, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(v, ast.Constant):
        if isinstance(v.value, bool):
            return ast.Name(id="bool", ctx=ast.Load())
        if isinstance(v.value, int):
            return ast.Name(id="int", ctx=ast.Load())
        if isinstance(v.value, float):
            return ast.Name(id="float", ctx=ast.Load())
        if isinstance(v.value, str):
            return ast.Name(id="str", ctx=ast.Load())
    return None


# ---------------------------------------------------------------------------
# Положительные мутации: (fn_no_doc, doc_old, rng) -> Optional[new_fn_no_doc]
# ---------------------------------------------------------------------------

def mut_param_renamed(fn, doc_old: str, rng: random.Random):
    # Переименование делает док ложным, только если параметр задокументирован
    # как аргумент (упоминания в прозе недостаточно: проза остаётся верной — C,
    # а не A; замер v2r4: прозовых переименований было 25,2%).
    cands = [p for p in _regular_params(fn) if _documented_as_param(p.arg, doc_old)]
    if not cands:
        return None
    # Скрытые привязки (except-as, match, global/nonlocal, alias, def/class
    # с тем же именем) плоский обход не переименует — таких кандидатов нет.
    cands = [p for p in cands if not _has_hidden_binding(fn, p.arg)]
    if not cands:
        return None
    rng.shuffle(cands)
    taken = _existing_names(fn)
    for p in cands:
        new_name = _fresh_name(p.arg, taken, rng)
        if new_name is None:
            continue
        old = p.arg
        new_fn = copy.deepcopy(fn)
        for n in ast.walk(new_fn):
            if isinstance(n, ast.arg) and n.arg == old:
                n.arg = new_name
            elif isinstance(n, ast.Name) and n.id == old:
                n.id = new_name
        return new_fn
    return None


def mut_param_added(fn, doc_old: str, rng: random.Random):
    taken = _all_param_names(fn)
    bound = _bound_names(fn)
    doc_low = (doc_old or "").lower()
    cands: list[tuple[str, str, object]] = []
    for n, d in NEW_PARAMS_12:
        if n in taken or n in bound:
            continue
        if re.search(r"\b" + re.escape(n.lower()) + r"\b", doc_low) is not None:
            continue
        v = _literal_value(d)
        if v is _SENTINEL:
            continue
        cands.append((n, d, v))
    if not cands:
        return None
    order = list(cands)
    rng.shuffle(order)
    for name, default_code, value in order:
        try:
            default_node = ast.parse(default_code, mode="eval").body
        except SyntaxError:
            continue
        new_fn = copy.deepcopy(fn)
        matches = _body_const_matches(new_fn, value)
        if not matches:
            continue
        parent, target = matches[rng.randrange(len(matches))]
        if parent is None:
            continue
        new_name = ast.Name(id=name, ctx=ast.Load())
        ast.copy_location(new_name, target)
        if not _replace_node(parent, target, new_name):
            continue
        a = new_fn.args
        new_arg = ast.arg(arg=name, annotation=None)
        # Аннотацию ставим, только если у остальных параметров они есть.
        existing_annot = any(
            getattr(x, "annotation", None) is not None
            for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
        )
        if existing_annot:
            new_arg.annotation = _annotation_for_default(default_code)
        a.kwonlyargs.append(new_arg)
        a.kw_defaults.append(default_node)
        ast.fix_missing_locations(new_fn)
        return new_fn
    return None


def _literal_value(code: str):
    try:
        return ast.literal_eval(ast.parse(code, mode="eval").body)
    except Exception:  # noqa: BLE001 — не литерал, кандидат неприменим
        return _SENTINEL


def _const_matches(default_value, const_value) -> bool:
    # Строго по типу: bool не смешивается с int, int — с float.
    if isinstance(default_value, bool) or isinstance(const_value, bool):
        return (
            isinstance(default_value, bool)
            and isinstance(const_value, bool)
            and default_value == const_value
        )
    return type(default_value) is type(const_value) and default_value == const_value


def _bound_names(fn) -> set[str]:
    # Имена, связанные в функции: кандидат, затенённый локалом,
    # был бы мёртв (UnboundLocalError-риск) — таких не добавляем.
    bound = set(_all_param_names(fn))
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.MatchAs) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            bound.add(n.rest)
        elif isinstance(n, ast.MatchStar) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.alias) and n.asname:
            bound.add(n.asname)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
            bound.add(n.name)
    return bound


def _body_const_matches(fn, value) -> list:
    # Константы только из тела (дефолты сигнатуры и декораторы не трогаем:
    # подстановка туда дала бы NameError при определении).
    # Литеральные куски f-строк (родитель JoinedStr) и голые Expr пропускаем.
    out = []
    for stmt in fn.body:
        for parent, node in _walk_with_parent(stmt, None):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(parent, (ast.JoinedStr, ast.Expr)):
                continue
            if _const_matches(value, node.value):
                out.append((parent, node))
    return out


def _replace_node(parent, old, new) -> bool:
    for field, val in ast.iter_fields(parent):
        if val is old:
            setattr(parent, field, new)
            return True
        if isinstance(val, list):
            for i, item in enumerate(val):
                if item is old:
                    val[i] = new
                    return True
    return False


def mut_param_removed(fn, doc_old: str, rng: random.Random):
    # Удаление stales док, только если параметр задокументирован как аргумент
    # (упоминания в прозе недостаточно: удаление не делает утверждений ложными).
    cands = [p for p in _regular_params(fn) if _documented_as_param(p.arg, doc_old)]
    if not cands:
        return None
    cands = [p for p in cands if not _has_hidden_binding(fn, p.arg)]
    if not cands:
        return None
    with_def = [p for p in cands if _param_default_node(fn, p.arg) is not None]
    # Предпочтительно с дефолтом (~80%).
    pool = with_def if with_def and rng.random() < 0.8 else cands
    target = pool[rng.randrange(len(pool))]
    tname = target.arg
    # Параметр перезаписывается или удаляется в теле (цель присваивания) —
    # удаление породило бы некорректный код, мутация неприменима.
    for n in ast.walk(fn):
        if (isinstance(n, ast.Name) and n.id == tname
                and isinstance(n.ctx, (ast.Store, ast.Del))):
            return None
    def_node = _param_default_node(fn, tname)
    # Без дефолта чтения параметра нечем заменить: подстановка None ломает
    # код (напр. `re.match(rx, None)`), мутация неприменима. Чистое удаление
    # без чтений в теле (как абстрактные методы) разрешено.
    if def_node is None:
        for n in ast.walk(fn):
            if (isinstance(n, ast.Name) and n.id == tname
                    and isinstance(n.ctx, (ast.Load, ast.Del))):
                return None
    new_fn = copy.deepcopy(fn)
    if not _remove_param(new_fn, tname):
        return None
    # Все чтения параметра в теле -> дефолт либо None.
    repl = copy.deepcopy(def_node) if def_node is not None else ast.Constant(value=None)
    for n in ast.walk(new_fn):
        if (isinstance(n, ast.Name) and n.id == tname
                and isinstance(n.ctx, ast.Load)):
            # arg-узлы уже удалены; остались только Name-использования.
            n.id = "__fd_removed_placeholder__"
    # Заменяем плейсхолдеры выражением дефолта: проще через трансформер.
    # Узлы Store/Del заменять запрещено (иначе `None = ...`).
    # Подставленные None помечаются: ниже проверяется, что они не попали
    # в crash-позиции (иначе `None(predict)`, `None/1000` — так мейнтейнер
    # не пишет, мутация считается неприменимой).
    class _Rep(ast.NodeTransformer):
        def visit_Name(self, node):  # noqa: N802
            if node.id == "__fd_removed_placeholder__" and isinstance(node.ctx, ast.Load):
                new_node = copy.deepcopy(repl)
                new_node._fd_subst = True
                if isinstance(new_node, ast.Constant) and new_node.value is None:
                    new_node._fd_subst_none = True
                return new_node
            return node
    new_fn = _Rep().visit(new_fn)
    ast.fix_missing_locations(new_fn)
    for parent, node in _walk_with_parent(new_fn):
        if not getattr(node, "_fd_subst_none", False):
            continue
        if isinstance(parent, (ast.BinOp, ast.Await, ast.Starred, ast.YieldFrom)):
            return None
        if isinstance(parent, ast.Call) and parent.func is node:
            return None
        if isinstance(parent, (ast.Attribute, ast.Subscript)) and parent.value is node:
            return None
        if isinstance(parent, ast.UnaryOp) and not isinstance(parent.op, ast.Not):
            return None
        if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node:
            return None
        if isinstance(parent, ast.ExceptHandler) and parent.type is node:
            return None
    for parent, node in _walk_with_parent(new_fn):
        # Подставленный литерал в статически разрешимом условии
        # (`if None == 'true'`, `if not True:`) так мейнтейнер не пишет.
        if not getattr(node, "_fd_subst", False):
            continue
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(parent, ast.Compare):
            return None
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return None
    return new_fn


def _param_default_node(fn, name: str):
    a = fn.args
    combined = list(a.posonlyargs) + list(a.args)
    names = [x.arg for x in combined]
    if name in names:
        idx = names.index(name)
        n_pos = len(combined)
        n_def = len(a.defaults)
        d_idx = idx - (n_pos - n_def)
        if d_idx >= 0:
            return a.defaults[d_idx]
        return None
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg == name:
            return d
    return None


def _remove_param(fn, name: str) -> bool:
    a = fn.args
    for lst_name in ("posonlyargs", "args"):
        lst = getattr(a, lst_name)
        for i, x in enumerate(lst):
            if x.arg == name:
                #.defaults покрывают хвост posonlyargs+args.
                combined = list(a.posonlyargs) + list(a.args)
                gidx = [x.arg for x in combined].index(name)
                n_pos = len(combined)
                n_def = len(a.defaults)
                d_idx = gidx - (n_pos - n_def)
                del lst[i]
                if d_idx >= 0:
                    del a.defaults[d_idx]
                return True
    for i, x in enumerate(list(a.kwonlyargs)):
        if x.arg == name:
            del a.kwonlyargs[i]
            del a.kw_defaults[i]
            return True
    if a.vararg is not None and a.vararg.arg == name:
        a.vararg = None
        return True
    if a.kwarg is not None and a.kwarg.arg == name:
        a.kwarg = None
        return True
    return False


def mut_default_changed(fn, doc_old: str, rng: random.Random):
    lines = (doc_old or "").splitlines()
    cands: list[tuple[ast.arg, ast.Constant]] = []
    for p in _regular_params(fn):
        dnode = _param_default_node(fn, p.arg)
        if not isinstance(dnode, ast.Constant):
            continue
        # Дефолт задокументирован: имя словом + слово default в той же строке
        # или двух следующих (покрывает numpydoc-переносы). Упоминания значения
        # в прозе недостаточно (аналог примера 3 из R9 — C, не A).
        hit = False
        name_re = re.compile(r"\b" + re.escape(p.arg) + r"\b")
        for i, ln in enumerate(lines):
            if not name_re.search(ln):
                continue
            window = "\n".join(lines[i:i + 3])
            if "default" in window.lower():
                hit = True
                break
        if hit:
            cands.append((p, dnode))
    if not cands:
        return None
    # Сортируем для детерминизма при фиксированном сиде, затем shuffle.
    # Сортируем для детерминизма при фиксированном сиде, затем shuffle.
    cands.sort(key=lambda t: t[0].arg)
    rng.shuffle(cands)
    for p, dnode in cands:
        v = dnode.value
        new_fn = copy.deepcopy(fn)
        nd = _param_default_node(new_fn, p.arg)
        new_val = _mutate_const_default(v, rng)
        if new_val is _SENTINEL or new_val == v:
            continue
        nd.value = new_val
        return new_fn
    return None


_SENTINEL = object()


def _mutate_const_default(v, rng: random.Random):
    if isinstance(v, bool):
        return (not v)
    if isinstance(v, int) and not isinstance(v, bool):
        opts = [v + 1, v * 2, v * 10, v // 2, 0]
        opts = [x for x in opts if x != v]
        return rng.choice(opts) if opts else _SENTINEL
    if isinstance(v, float):
        opts = [v * 2, v / 2, v + 0.5, 0.0]
        opts = [x for x in opts if x != v]
        return rng.choice(opts) if opts else _SENTINEL
    if isinstance(v, str):
        if v in STR_DEFAULT_REPLACE:
            return STR_DEFAULT_REPLACE[v]
        ways = [v + "_v2", v.upper() if v != v.upper() else v.lower(), "x_" + v, v + "!"]
        return rng.choice(ways)
    return _SENTINEL


def _returns_nullable(returns: ast.expr) -> bool:
    if isinstance(returns, ast.Constant) and returns.value is None:
        return True
    if isinstance(returns, ast.Subscript):
        v = returns.value
        if isinstance(v, ast.Name) and v.id == "Optional":
            return True
        if isinstance(v, ast.Name) and v.id == "Union":
            sl = returns.slice
            elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
            if any(isinstance(e, ast.Constant) and e.value is None for e in elts):
                return True
    try:
        text = ast.unparse(returns)
    except Exception:  # noqa: BLE001
        return False
    t = text.replace(" ", "")
    return t.startswith("Optional[") or ("Union[" in t and "None" in t)


def _has_none_guard(fn, first: str) -> bool:
    # В теле уже есть идентичный гвард (if not first / if first is None
    # с телом return None) — вставка дубля сделала бы старый код мёртвым.
    for n in ast.walk(fn):
        if not isinstance(n, ast.If):
            continue
        test = n.test
        is_not_first = (
            isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name) and test.operand.id == first
        )
        is_none_cmp = (
            isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == first and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        )
        if not (is_not_first or is_none_cmp):
            continue
        if (len(n.body) == 1 and isinstance(n.body[0], ast.Return)
                and isinstance(n.body[0].value, ast.Constant)
                and n.body[0].value.value is None):
            return True
    return False


def mut_return_changed(fn, doc_old: str, rng: random.Random):
    if fn.returns is None:
        return None
    # Уже nullable (Optional/Union[..., None]) — иначе артефакт Optional[Optional[...]].
    if _returns_nullable(fn.returns):
        return None
    if "return" not in (doc_old or "").lower():
        return None
    params = [p.arg for p in _regular_params(fn)]
    if not params:
        return None
    first = params[0]
    # Такой гвард уже есть в теле — вставка дубля сделала бы старый код мёртвым.
    if _has_none_guard(fn, first):
        return None
    new_fn = copy.deepcopy(fn)
    old_ret = new_fn.returns
    new_fn.returns = ast.Subscript(
        value=ast.Name(id="Optional", ctx=ast.Load()),
        slice=copy.deepcopy(old_ret),
        ctx=ast.Load(),
    )
    # Всегда `is None`: truthiness-гард мёртв на объектах без __bool__/__len__
    # и падает на Series/DataFrame/ndarray (ambiguous truth value).
    guard = ast.parse(f"if {first} is None:\n    return None").body[0]
    new_fn.body.insert(0, guard)
    ast.fix_missing_locations(new_fn)
    return new_fn


def mut_exception_changed(fn, doc_old: str, rng: random.Random):
    # 1) замена типа raise из таблицы — только если старый тип
    # задокументирован (иначе замена ничего не делает ложным: C, не A).
    # Недокументированные типы проваливаются в ветку 2 (добавление проверки).
    doc_types = {t for t in EXC_REPLACE if _mentioned_as_word(t, doc_old)}
    matches: list[ast.Raise] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Raise) and n.exc is not None:
            t = _raise_type(n.exc)
            if t in doc_types:
                matches.append(n)
    if matches:
        new_fn = copy.deepcopy(fn)
        # Найти соответствующие узлы в копии по порядку обхода.
        copies = [n for n in ast.walk(new_fn)
                  if isinstance(n, ast.Raise) and n.exc is not None
                  and _raise_type(n.exc) in doc_types]
        if copies:
            target = copies[rng.randrange(len(copies))]
            _set_raise_type(target, EXC_REPLACE[_raise_type(target.exc)])
            return new_fn
    # 2) иначе — добавить проверку первого параметра.
    params = [p.arg for p in _regular_params(fn)]
    if not params:
        return None
    first = params[0]
    exc = rng.choice(["ValueError", "TypeError"])
    msg = rng.choice(EXC_MESSAGES)
    new_fn = copy.deepcopy(fn)
    guard = ast.parse(f"if {first} is None:\n    raise {exc}('{msg}')").body[0]
    new_fn.body.insert(0, guard)
    ast.fix_missing_locations(new_fn)
    return new_fn


def _raise_type(exc: ast.expr) -> Optional[str]:
    if isinstance(exc, ast.Name):
        return exc.id
    if isinstance(exc, ast.Call):
        f = exc.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
    if isinstance(exc, ast.Attribute):
        return exc.attr
    return None


def _set_raise_type(raise_node: ast.Raise, new_type: str) -> None:
    exc = raise_node.exc
    if isinstance(exc, ast.Name):
        exc.id = new_type
    elif isinstance(exc, ast.Call):
        f = exc.func
        if isinstance(f, ast.Name):
            f.id = new_type
        elif isinstance(f, ast.Attribute):
            f.attr = new_type


# ---------------------------------------------------------------------------
# Отрицательные мутации
# ---------------------------------------------------------------------------

def mut_rename_local(fn, doc_old: str, rng: random.Random):
    param_names = _all_param_names(fn)
    seen: list[str] = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            if n.id not in param_names and not n.id.startswith("_") and n.id not in seen:
                seen.append(n.id)
    if not seen:
        return None
    # Скрытые привязки (except-as, match, global/nonlocal) плоский обход
    # не переименует — такие локалы не трогаем.
    seen = [s for s in seen if not _has_hidden_binding(fn, s)]
    if not seen:
        return None
    rng.shuffle(seen)
    taken = _existing_names(fn)
    for old in seen:
        new_name = _fresh_name(old, taken, rng)
        if new_name is None:
            continue
        new_fn = copy.deepcopy(fn)
        for n in ast.walk(new_fn):
            if isinstance(n, ast.Name) and n.id == old:
                n.id = new_name
            elif isinstance(n, ast.arg) and n.arg == old:
                n.arg = new_name
        # Убедиться, что сигнатура не поменялась (локал не был параметром).
        return new_fn
    return None


class _ExtractTemp(ast.NodeTransformer):
    def __init__(self, tmp_name: str):
        self.tmp_name = tmp_name
        self.done = False
        # Стек вложенных def: сама целевая функция — глубина 1, её прямые
        # return — тоже 1; return вложенных функций (глубина ≥ 2) не трогаем.
        self._stack: list = []

    def visit_FunctionDef(self, node):  # noqa: N802
        self._stack.append(node)
        self.generic_visit(node)
        self._stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        self._stack.append(node)
        self.generic_visit(node)
        self._stack.pop()
        return node

    def visit_Return(self, node):  # noqa: N802
        # Только return целевой функции: вложенные def трогать нельзя.
        if self.done or len(self._stack) > 1 or node.value is None:
            return node
        self.done = True
        assign = ast.Assign(
            targets=[ast.Name(id=self.tmp_name, ctx=ast.Store())],
            value=node.value,
        )
        ret = ast.Return(value=ast.Name(id=self.tmp_name, ctx=ast.Load()))
        ast.copy_location(assign, node)
        ast.copy_location(ret, node)
        return [assign, ret]


def mut_extract_temp(fn, doc_old: str, rng: random.Random):
    has_ret = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))
    if not has_ret:
        return None
    taken = _existing_names(fn)
    cands = [t for t in TEMP_NAMES if t not in taken]
    if not cands:
        return None
    # Пробуем несколько имён (сид влияет на порядок).
    order = list(cands)
    rng.shuffle(order)
    for tmp in order:
        new_fn = copy.deepcopy(fn)
        tr = _ExtractTemp(tmp)
        new_fn = tr.visit(new_fn)
        if tr.done:
            ast.fix_missing_locations(new_fn)
            return new_fn
    return None


def _mutate_message(s: str, rng: random.Random) -> str:
    ways: list[Callable[[str], str]] = [
        lambda x: "Error: " + x,
        lambda x: x + ".",
        lambda x: x + "!",
        lambda x: x + " Please retry.",
        lambda x: x[0].upper() + x[1:] if x else x,
        lambda x: x.replace("bad", "invalid") if "bad" in x else x + " [updated]",
    ]
    order = list(range(len(ways)))
    rng.shuffle(order)
    for i in order:
        out = ways[i](s)
        if out != s:
            return out
    return s + "!"


_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _is_message_call(node: ast.Call) -> bool:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in ("print", "warn")
    if isinstance(f, ast.Attribute):
        return f.attr in _LOG_METHODS
    return False


def _message_constants(fn) -> list[ast.Constant]:
    # Только строки сообщений: внутри вызовов print/logging/warnings.warn
    # или внутри аргументов raise. Изолированные константы бизнес-логики
    # (условия, тернарные операторы, словари, возвраты) игнорируются.
    allowed: set[int] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and _is_message_call(n):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    allowed.add(id(sub))
        elif isinstance(n, ast.Raise):
            for sub in ast.walk(n):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    allowed.add(id(sub))
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) in allowed]


def mut_change_message(fn, doc_old: str, rng: random.Random):
    doc_low = (doc_old or "").lower()
    cands = [n for n in _message_constants(fn)
             if n.value.strip() and n.value.strip().lower() not in doc_low]
    if not cands:
        return None
    k = rng.randrange(len(cands))
    new_fn = copy.deepcopy(fn)
    copies = [n for n in _message_constants(new_fn)
              if n.value.strip() and n.value.strip().lower() not in doc_low]
    if len(copies) != len(cands):
        return None
    copies[k].value = _mutate_message(copies[k].value, rng)
    return new_fn


# Реестр мутаций: имя -> (функция, метка, ожидаемый drift).
MUTATIONS: dict[str, tuple[Callable, int, str]] = {
    "param_renamed": (mut_param_renamed, 1, "param_renamed"),
    "param_added": (mut_param_added, 1, "param_added"),
    "param_removed": (mut_param_removed, 1, "param_removed"),
    "default_changed": (mut_default_changed, 1, "default_changed"),
    "return_changed": (mut_return_changed, 1, "return_changed"),
    "exception_changed": (mut_exception_changed, 1, "exception_changed"),
    "rename_local": (mut_rename_local, 0, "none"),
    "extract_temp": (mut_extract_temp, 0, "none"),
    "change_message": (mut_change_message, 0, "none"),
}

POS_TYPES = [k for k, (_, lb, _) in MUTATIONS.items() if lb == 1]
NEG_TYPES = [k for k, (_, lb, _) in MUTATIONS.items() if lb == 0]
