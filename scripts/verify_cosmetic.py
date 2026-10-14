"""Synthetic verification of cosmetic_equal against R2.7.

R2.7 (strict): two docstring-free variants are cosmetic iff renaming
LOCAL VARIABLES to v0, v1, ... (order of first appearance) makes
ast.dump equal. Free names -- call targets, decorators, attributes,
parameters -- must NOT be renamed: changing them is a behavior change.

Usage:
    .venv/bin/python3.11 scripts/verify_cosmetic.py
Exit code 0 = all cases pass.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mining.miner import cosmetic_equal

# (description, old_src, new_src, expected_cosmetic)
CASES = [
    # --- positives: pure local renames (R2.7) ---
    ("one local renamed",
     "def f(items):\n    total = 0\n    for i in items:\n        total += i\n"
     "    return total\n",
     "def f(items):\n    acc = 0\n    for i in items:\n        acc += i\n"
     "    return acc\n", True),
    ("several locals renamed",
     "def f(a):\n    x = a + 1\n    y = x * 2\n    return y\n",
     "def f(a):\n    b = a + 1\n    c = b * 2\n    return c\n", True),
    ("for-loop target renamed",
     "def f(items):\n    out = []\n    for i in items:\n        out.append(i)\n"
     "    return out\n",
     "def f(items):\n    out = []\n    for item in items:\n"
     "        out.append(item)\n    return out\n", True),
    ("comprehension target renamed",
     "def f(xs):\n    return [i * 2 for i in xs]\n",
     "def f(xs):\n    return [x * 2 for x in xs]\n", True),
    ("with-as renamed",
     "def f(path):\n    with open(path) as fh:\n        return fh.read()\n",
     "def f(path):\n    with open(path) as handle:\n        return handle.read()\n",
     True),
    ("except-as renamed",
     "def f():\n    try:\n        return 1\n    except ValueError as e:\n"
     "        return 0\n",
     "def f():\n    try:\n        return 1\n    except ValueError as err:\n"
     "        return 0\n", True),
    # --- negatives: any other change is NOT cosmetic ---
    ("call target renamed (behavior change)",
     "def f(x):\n    return print(x)\n",
     "def f(x):\n    return log(x)\n", False),
    ("decorator renamed",
     "@foo\ndef f():\n    return 1\n",
     "@bar\ndef f():\n    return 1\n", False),
    ("attribute renamed",
     "def f(obj, k):\n    return obj.data[k]\n",
     "def f(obj, k):\n    return obj.store[k]\n", False),
    ("constant changed",
     "def f():\n    x = 1\n    return x\n",
     "def f():\n    x = 2\n    return x\n", False),
    ("operator changed",
     "def f(items):\n    t = 0\n    for i in items:\n        t += i\n    return t\n",
     "def f(items):\n    t = 0\n    for i in items:\n        t -= i\n    return t\n",
     False),
    ("string literal changed",
     "def f():\n    raise ValueError('a')\n",
     "def f():\n    raise ValueError('b')\n", False),
    ("statement added",
     "def f(x):\n    y = x + 1\n    return y\n",
     "def f(x):\n    y = x + 1\n    z = y * 2\n    return y\n", False),
    ("parameter renamed (not a local per R2.7)",
     "def f(value):\n    return value + 1\n",
     "def f(arg):\n    return arg + 1\n", False),
    ("local renamed AND call target renamed (mixed)",
     "def f(x):\n    t = print(x)\n    return t\n",
     "def f(x):\n    u = log(x)\n    return u\n", False),
]


def main() -> int:
    failures = 0
    for desc, old_src, new_src, expected in CASES:
        fn_old = ast.parse(old_src).body[0]
        fn_new = ast.parse(new_src).body[0]
        got = cosmetic_equal(fn_old, fn_new)
        ok = got == expected
        failures += not ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {desc}: cosmetic={got}, expected={expected}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} cases pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
