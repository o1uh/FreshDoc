import unittest
from src.data.metadata_extractor import extract_metadata

class TestMetadataExtractor(unittest.TestCase):
    def test_basic_function(self):
        code = """
def add(a, b=2):
    """Adds two numbers.

    Args:
        a: first number
        b: second number (default 2)
    Returns:
        sum of a and b
    """
        return a + b
"""
        doc = """Adds two numbers.

    Args:
        a: first number
        b: second number (default 2)
    Returns:
        sum of a and b
    """
        meta = extract_metadata(code, doc)
        self.assertEqual(meta["len"], 5)  # 5 non‑empty lines inside function
        self.assertEqual(meta["params"], 2)
        self.assertTrue(meta["has_args"])
        self.assertTrue(meta["has_returns"])
        self.assertFalse(meta["has_raises"])

if __name__ == "__main__":
    unittest.main()
