from __future__ import annotations

import random
import unittest

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - local machines may not have torch.
    np = None
    torch = None

from src.common.seed import make_generator, set_seed


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SeedTest(unittest.TestCase):
    def test_set_seed_reproducible(self) -> None:
        set_seed(123)
        values_a = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

        set_seed(123)
        values_b = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

        self.assertEqual(values_a, values_b)

    def test_dataloader_generator_order_reproducible(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        loader_a = DataLoader(dataset, batch_size=2, shuffle=True, generator=make_generator(7))
        loader_b = DataLoader(dataset, batch_size=2, shuffle=True, generator=make_generator(7))

        order_a = [int(item) for batch in loader_a for item in batch[0]]
        order_b = [int(item) for batch in loader_b for item in batch[0]]

        self.assertEqual(order_a, order_b)


if __name__ == "__main__":
    unittest.main()
