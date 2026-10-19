"""Tests for src.data.docsections (shared section detector, FD-15/M1)."""
import unittest

from src.data.docsections import doc_section_flags
from src.models.baselines import _doc_section_flags


class TestDocSections(unittest.TestCase):
    def test_google_style(self):
        self.assertEqual(
            doc_section_flags("Do.\n\nArgs:\n    x: v\n"), (1, 0, 0))

    def test_parameters_spelling_counts_as_args(self):
        self.assertEqual(
            doc_section_flags("Do.\n\nParameters:\n    x: v\n")[0], 1)

    def test_numpy_underline(self):
        self.assertEqual(
            doc_section_flags("Do.\n\nParameters\n----------\nx\n")[0], 1)
        self.assertEqual(
            doc_section_flags("Do.\n\nReturns\n-------\nint\n")[1], 1)

    def test_rst_markers(self):
        self.assertEqual(
            doc_section_flags("Do.\n\n:param x: v\n")[0], 1)
        self.assertEqual(
            doc_section_flags("Do.\n\n:returns: int\n")[1], 1)
        self.assertEqual(
            doc_section_flags("Do.\n\n:raises ValueError:\n")[2], 1)

    def test_prose_without_marker(self):
        self.assertEqual(doc_section_flags("Return the value."), (0, 0, 0))
        self.assertEqual(
            doc_section_flags("Handles network errors."), (0, 0, 0))

    def test_empty(self):
        self.assertEqual(doc_section_flags(""), (0, 0, 0))
        self.assertEqual(doc_section_flags(None), (0, 0, 0))

    def test_baselines_wrapper_matches(self):
        doc = "Do.\n\nParameters\n----------\nx\n\nRaises:\n    E\n"
        self.assertEqual(tuple(_doc_section_flags(doc)),
                         doc_section_flags(doc))


if __name__ == "__main__":
    unittest.main()
