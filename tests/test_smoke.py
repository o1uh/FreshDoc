import ast


def test_python_version_support():
    code = "def foo(x: int) -> bool: return x > 0"
    tree = ast.parse(code)
    unparsed = ast.unparse(tree)
    assert "def foo" in unparsed


def test_imports():
    import numpy
    import pandas
    import pytest
    import sklearn
    import torch
    import transformers
    assert True