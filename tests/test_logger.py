from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

from src.common.logger import SwanLabLogger, get_git_commit, setup_logging


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


if __name__ == "__main__":
    unittest.main()
