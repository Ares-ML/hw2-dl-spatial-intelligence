from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from src.common.logger import SwanLabLogger, get_git_commit, setup_logging
from src.common.swanlab_logger import finish, format_run_name, init_run, log_metrics, metric_name


class LoggerTest(unittest.TestCase):
    def test_setup_logging_does_not_duplicate_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "train.log"
            logger = setup_logging("hw2.test.logger", log_file=log_file, level=logging.INFO)
            handler_count = len(logger.handlers)
            logger_again = setup_logging("hw2.test.logger", log_file=log_file, level=logging.INFO)
            logger_again.info("hello")

            try:
                for handler in logger_again.handlers:
                    handler.flush()
                text = log_file.read_text(encoding="utf-8")
            finally:
                for handler in list(logger_again.handlers):
                    handler.close()
                    logger_again.removeHandler(handler)

        self.assertIs(logger, logger_again)
        self.assertEqual(handler_count, 2)
        self.assertEqual(text.count("hello"), 1)

    def test_git_commit_returns_string(self) -> None:
        self.assertIsInstance(get_git_commit(), str)

    def test_swanlab_disabled_is_noop(self) -> None:
        logger = SwanLabLogger(enabled=False)
        logger.log({"val/miou": 0.5}, step=1)
        logger.update_config({"seed": 42})
        logger.finish()
        self.assertFalse(logger.enabled)

    def test_swanlab_function_wrapper_disabled_is_noop(self) -> None:
        run_name = format_run_name(
            {"model": "resnet18", "init": "pretrained", "loss": "ce", "lr": 1e-4, "epochs": 50, "seed": 42}
        )
        self.assertEqual(run_name, "resnet18_pretrained_ce_lr1e-4_e50_s42")
        self.assertEqual(metric_name("task1", "train_loss"), "task1/train_loss")
        run = init_run(task="task1", run_name=run_name, mode="disabled")
        log_metrics({"train_loss": 1.0}, step=1, task="task1")
        self.assertFalse(run.enabled)
        self.assertFalse(finish().enabled)


if __name__ == "__main__":
    unittest.main()
