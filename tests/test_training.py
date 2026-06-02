import csv
import tempfile
import unittest
from pathlib import Path

from src.training import EarlyStopping, HistoryLogger, resolve_generation_settings


class TrainingHelperTests(unittest.TestCase):
    def test_early_stopping_resets_after_improvement(self):
        stopper = EarlyStopping(patience=2)
        self.assertTrue(stopper.update(0, 3.0))
        self.assertFalse(stopper.update(10, 3.1))
        self.assertTrue(stopper.update(20, 2.9))
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update(30, 3.0))
        self.assertFalse(stopper.update(40, 3.1))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_step, 20)

    def test_history_logger_writes_csv_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'history.csv'
            history = HistoryLogger(path)
            history.append(0, 4.0, 4.2)
            with path.open(encoding='utf-8') as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [['step', 'train_loss', 'val_loss'], ['0', '4.0', '4.2']])

    def test_generation_presets_allow_overrides(self):
        self.assertEqual(resolve_generation_settings('precise'), (0.5, 15))
        self.assertEqual(resolve_generation_settings('creative', temperature=0.7), (0.7, 80))


if __name__ == '__main__':
    unittest.main()
