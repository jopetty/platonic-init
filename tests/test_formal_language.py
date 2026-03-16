from __future__ import annotations

import unittest

from platonic_init.formal_language import (
    generate_formal_language_lines,
    generate_k_dyck_exact_depth,
    generate_shuffle_dyck,
    generate_ww,
    is_valid_k_dyck,
    is_valid_shuffle_dyck,
    is_valid_ww,
)


class FormalLanguageTests(unittest.TestCase):
    def test_generate_k_dyck_produces_valid_string(self) -> None:
        import random

        rng = random.Random(0)
        tokens = generate_k_dyck_exact_depth(4, rng, k=3)
        self.assertTrue(is_valid_k_dyck(tokens, k=3))

    def test_generate_shuffle_dyck_produces_valid_string(self) -> None:
        import random

        rng = random.Random(1)
        tokens = generate_shuffle_dyck(3, rng, k=4)
        self.assertTrue(is_valid_shuffle_dyck(tokens, k=4))

    def test_generate_ww_duplicates_prefix(self) -> None:
        import random

        rng = random.Random(2)
        tokens = generate_ww(rng, alphabet_size=8, min_half_length=4, max_half_length=4)
        self.assertTrue(is_valid_ww(tokens))

    def test_generate_formal_language_lines_is_reproducible(self) -> None:
        first = generate_formal_language_lines(
            language="shuffle_dyck", n_samples=5, seed=7, max_depth=3, k=3
        )
        second = generate_formal_language_lines(
            language="shuffle_dyck", n_samples=5, seed=7, max_depth=3, k=3
        )
        self.assertEqual(first, second)

    def test_compact_single_dyck_uses_parentheses(self) -> None:
        lines = generate_formal_language_lines(
            language="dyck",
            n_samples=2,
            seed=3,
            max_depth=3,
            k=1,
            compact_single_dyck=True,
        )
        self.assertTrue(all(set(line) <= {"(", ")"} for line in lines))


if __name__ == "__main__":
    unittest.main()
