"""Tests for src.data.token_analysis (FD-15)."""
import unittest

from src.data.token_analysis import (
    _added_tokens,
    document_frequencies,
    filter_shortcut_tokens,
    max_literal_gap,
    max_suffix_gap,
    top_suspicious,
    word_token_gaps,
)


class TestAddedTokens(unittest.TestCase):
    """Verify that only added diff lines are tokenised."""

    def test_only_added_lines(self):
        diff = (
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,3 +1,4 @@\n"
            " unchanged\n"
            "-removed_line\n"
            "+added_alpha beta\n"
            "+added_gamma\n"
        )
        tokens = _added_tokens(diff)
        self.assertEqual(tokens, ["added_alpha", "beta", "added_gamma"])

    def test_header_plus_not_included(self):
        diff = "+++ b/foo.py\n+real_token\n"
        tokens = _added_tokens(diff)
        self.assertEqual(tokens, ["real_token"])

    def test_empty_diff(self):
        self.assertEqual(_added_tokens(""), [])

    def test_no_added_lines(self):
        diff = "--- a/foo.py\n-only removed\n context\n"
        self.assertEqual(_added_tokens(diff), [])


class TestDocumentFrequencies(unittest.TestCase):
    """Per-document frequency: fraction of pairs containing a token."""

    def test_basic(self):
        pairs = [
            ("+alpha beta", 1),
            ("+alpha gamma", 1),
            ("+beta delta", 0),
            ("+delta epsilon", 0),
        ]
        freqs = document_frequencies(pairs)
        # alpha: in 2/2 pos, 0/2 neg
        self.assertAlmostEqual(freqs["alpha"][0], 1.0)
        self.assertAlmostEqual(freqs["alpha"][1], 0.0)
        # beta: 1/2 pos, 1/2 neg
        self.assertAlmostEqual(freqs["beta"][0], 0.5)
        self.assertAlmostEqual(freqs["beta"][1], 0.5)
        # delta: 0/2 pos, 2/2 neg
        self.assertAlmostEqual(freqs["delta"][0], 0.0)
        self.assertAlmostEqual(freqs["delta"][1], 1.0)

    def test_duplicate_token_in_pair_counted_once(self):
        """Same token twice in one pair should count as one document."""
        pairs = [
            ("+x x x", 1),
            ("+y", 0),
        ]
        freqs = document_frequencies(pairs)
        self.assertAlmostEqual(freqs["x"][0], 1.0)

    def test_empty_pairs(self):
        freqs = document_frequencies([])
        self.assertEqual(freqs, {})


class TestFilterShortcutTokens(unittest.TestCase):
    def test_gap_filter(self):
        freqs = {
            "shortcut": (0.9, 0.1),   # gap 0.8
            "borderline": (0.5, 0.2),  # gap 0.3 — NOT > 0.3
            "safe": (0.3, 0.3),        # gap 0.0
        }
        hits = filter_shortcut_tokens(freqs, gap=0.3)
        tokens = [t for t, _, _ in hits]
        self.assertIn("shortcut", tokens)
        self.assertNotIn("borderline", tokens)
        self.assertNotIn("safe", tokens)

    def test_sorted_by_gap_descending(self):
        freqs = {
            "a": (0.8, 0.1),
            "b": (0.9, 0.05),
        }
        hits = filter_shortcut_tokens(freqs, gap=0.3)
        self.assertEqual(hits[0][0], "b")  # gap 0.85 > 0.7

    def test_absolute_catches_negative_side(self):
        freqs = {"neg_marker": (0.05, 0.9)}  # gap -0.85
        self.assertEqual(len(filter_shortcut_tokens(freqs, gap=0.3)), 1)
        self.assertEqual(
            len(filter_shortcut_tokens(freqs, gap=0.3,
                                       absolute=False)), 0)


class TestWordTokenGaps(unittest.TestCase):
    def test_suffix_visible_in_word_tokens(self):
        pairs = [
            ("+verbose_tmp = 1", 1),
            ("+verbose_tmp = 2", 1),
            ("+plain = 1", 0),
            ("+other = 2", 0),
        ]
        freqs = word_token_gaps(pairs)
        gap, tok = max_suffix_gap(freqs)
        self.assertEqual(tok, "verbose_tmp")
        self.assertAlmostEqual(gap, 1.0)

    def test_no_suffix_tokens(self):
        pairs = [("+plain = 1", 1), ("+other = 2", 0)]
        gap, tok = max_suffix_gap(word_token_gaps(pairs))
        self.assertEqual(gap, 0.0)
        self.assertEqual(tok, "")

    def test_literal_gap_excludes_structural(self):
        pairs = [
            ("+if x is None:", 1),
            ("+if y is None:", 1),
            ("+z = 1", 0),
            ("+w = 2", 0),
        ]
        freqs = word_token_gaps(pairs)
        gap_all, _ = max_literal_gap(freqs, exclude_structural=False)
        self.assertGreater(gap_all, 0.5)  # 'none' structural
        gap, _ = max_literal_gap(freqs, exclude_structural=True)
        self.assertLess(gap, 0.3)


class TestTopSuspicious(unittest.TestCase):
    def test_returns_top_n(self):
        freqs = {f"t{i}": (i / 10, 0.0) for i in range(1, 11)}
        top = top_suspicious(freqs, n=3)
        self.assertEqual(len(top), 3)
        self.assertEqual(top[0][0], "t10")

    def test_by_abs_surfaces_negative_side(self):
        freqs = {"a": (0.9, 0.1), "b": (0.1, 0.9)}
        top = top_suspicious(freqs, n=2, by_abs=True)
        self.assertEqual({t for t, _, _, _ in top}, {"a", "b"})
        top_pos = top_suspicious(freqs, n=1)
        self.assertEqual(top_pos[0][0], "a")


if __name__ == "__main__":
    unittest.main()
