"""Tests for src.data.metadata_extractor (FD-15)."""
import unittest

from src.data.metadata_extractor import (
    extract_metadata,
    func_length,
    param_count,
    parse_function_ast,
    has_section,
)


class TestParseFunction(unittest.TestCase):
    def test_regular_function(self):
        code = "def foo(a, b):\n    return a + b\n"
        node = parse_function_ast(code)
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "foo")

    def test_async_function(self):
        code = "async def bar(x):\n    await x\n"
        node = parse_function_ast(code)
        self.assertIsNotNone(node)
        self.assertEqual(node.name, "bar")

    def test_empty_string(self):
        self.assertIsNone(parse_function_ast(""))

    def test_syntax_error(self):
        self.assertIsNone(parse_function_ast("def broken(:\n"))

    def test_no_function(self):
        self.assertIsNone(parse_function_ast("x = 1\ny = 2\n"))


class TestFuncLength(unittest.TestCase):
    def test_counts_non_empty(self):
        code = "def f():\n    a = 1\n\n    b = 2\n    return a + b\n"
        self.assertEqual(func_length(code), 4)  # def + 3 body

    def test_empty_code(self):
        self.assertEqual(func_length(""), 0)

    def test_broken_code(self):
        self.assertEqual(func_length("not a function"), 0)


class TestParamCount(unittest.TestCase):
    def test_no_params(self):
        node = parse_function_ast("def f():\n    pass\n")
        self.assertEqual(param_count(node), 0)

    def test_mixed_params(self):
        code = "def f(a, b, *args, kw=1, **kwargs):\n    pass\n"
        node = parse_function_ast(code)
        # a, b, *args, kw, **kwargs = 5
        self.assertEqual(param_count(node), 5)

    def test_self_counted(self):
        code = "def f(self, x):\n    pass\n"
        node = parse_function_ast(code)
        self.assertEqual(param_count(node), 2)


class TestHasSection(unittest.TestCase):
    def test_args_present(self):
        doc = "Do something.\n\nArgs:\n    x: value\n"
        self.assertTrue(has_section(doc, "Args"))

    def test_returns_present(self):
        doc = "Do something.\n\n    Returns:\n        int\n"
        self.assertTrue(has_section(doc, "Returns"))

    def test_missing_section(self):
        doc = "Simple docstring without sections."
        self.assertFalse(has_section(doc, "Args"))

    def test_empty_docstring(self):
        self.assertFalse(has_section("", "Args"))
        self.assertFalse(has_section(None, "Args"))

    def test_case_insensitive(self):
        doc = "Something.\n\n    args:\n        x: val\n"
        self.assertTrue(has_section(doc, "Args"))


class TestExtractMetadata(unittest.TestCase):
    def test_basic_function(self):
        code = (
            "def add(a, b=2):\n"
            '    """Adds two numbers.\n'
            "\n"
            "    Args:\n"
            "        a: first number\n"
            "        b: second number (default 2)\n"
            "    Returns:\n"
            "        sum of a and b\n"
            '    """\n'
            "    return a + b\n"
        )
        doc = (
            "Adds two numbers.\n"
            "\n"
            "    Args:\n"
            "        a: first number\n"
            "        b: second number (default 2)\n"
            "    Returns:\n"
            "        sum of a and b\n"
        )
        meta = extract_metadata(code, doc)
        self.assertGreater(meta["len"], 0)
        self.assertEqual(meta["params"], 2)
        self.assertTrue(meta["has_args"])
        self.assertTrue(meta["has_returns"])
        self.assertFalse(meta["has_raises"])

    def test_empty_code(self):
        meta = extract_metadata("", "some doc")
        self.assertEqual(meta["len"], 0)
        self.assertEqual(meta["params"], 0)
        self.assertFalse(meta["has_args"])

    def test_no_docstring(self):
        code = "def f(x):\n    return x\n"
        meta = extract_metadata(code, "")
        self.assertEqual(meta["params"], 1)
        self.assertFalse(meta["has_args"])
        self.assertFalse(meta["has_returns"])

    def test_function_with_raises(self):
        code = "def f():\n    raise ValueError\n"
        doc = "Do stuff.\n\nRaises:\n    ValueError: always\n"
        meta = extract_metadata(code, doc)
        self.assertTrue(meta["has_raises"])


if __name__ == "__main__":
    unittest.main()
