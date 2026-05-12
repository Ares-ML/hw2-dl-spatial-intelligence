from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - local machines may not have torch.
    torch = None

from src.common.checkpoint import CheckpointManager, load_checkpoint, save_checkpoint


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CheckpointTest(unittest.TestCase):
    def test_save_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            save_checkpoint(path, model=model, optimizer=optimizer, epoch=3, metrics={"val/acc": 0.5})

            loaded_model = torch.nn.Linear(2, 1)
            checkpoint = load_checkpoint(path, model=loaded_model)

        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["metrics"]["val/acc"], 0.5)
        for a, b in zip(model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_checkpoint_manager_tracks_best(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = torch.nn.Linear(2, 1)
            manager = CheckpointManager(tmpdir, monitor="val/acc", mode="max")

            result1 = manager.save(model=model, epoch=1, metrics={"val/acc": 0.5})
            result2 = manager.save(model=model, epoch=2, metrics={"val/acc": 0.4})
            result3 = manager.save(model=model, epoch=3, metrics={"val/acc": 0.7})

            best_checkpoint = load_checkpoint(Path(tmpdir) / "best.pt")
            last_checkpoint = load_checkpoint(Path(tmpdir) / "last.pt")

        self.assertTrue(result1["is_best"])
        self.assertFalse(result2["is_best"])
        self.assertTrue(result3["is_best"])
        self.assertEqual(best_checkpoint["epoch"], 3)
        self.assertEqual(last_checkpoint["epoch"], 3)
        self.assertEqual(manager.best_score, 0.7)


if __name__ == "__main__":
    unittest.main()
