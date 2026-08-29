import unittest

import numpy as np

from src.pilot_v1.common import exact_mask_count
from src.pilot_v1.intervention import (
    layer_counts, layer_matched_random_order, mask_for_step, validate_schedule,
)


class PilotMaskTests(unittest.TestCase):
    def test_exact_llama_counts(self):
        n = 458_752
        self.assertEqual(exact_mask_count(n, 1), 458)
        self.assertEqual(exact_mask_count(n, 2), 917)
        self.assertEqual(exact_mask_count(n, 100), 45_875)

    def test_nested_schedule(self):
        order = np.arange(1000)[::-1].copy()
        rows = validate_schedule(order, 1000)
        self.assertEqual(len(rows), 101)
        self.assertEqual(rows[100]["count"], 100)
        for step in range(1, 101):
            self.assertTrue(set(mask_for_step(order, 1000, step - 1)).issubset(mask_for_step(order, 1000, step)))

    def test_layer_matched_random_every_step(self):
        layers, width = 4, 250
        primary = np.random.default_rng(2).permutation(layers * width)
        random_order = layer_matched_random_order(primary, layers, width, seed=19)
        self.assertFalse(np.array_equal(primary, random_order))
        for step in range(101):
            k = exact_mask_count(layers * width, step)
            self.assertTrue(np.array_equal(
                layer_counts(primary[:k], width, layers),
                layer_counts(random_order[:k], width, layers),
            ))


if __name__ == "__main__":
    unittest.main()
