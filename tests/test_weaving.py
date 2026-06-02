import json
import tempfile
import unittest
from pathlib import Path

from src.weaving import (
    load_trace,
    normalize_weights,
    parse_assignment,
    validate_compatible_configs,
)


class WeavingHelperTests(unittest.TestCase):
    def test_parse_assignment_requires_name_value(self):
        self.assertEqual(parse_assignment('poetry=out/poetry.pt', '--model'), ('poetry', 'out/poetry.pt'))
        with self.assertRaises(ValueError):
            parse_assignment('poetry', '--model')

    def test_normalize_weights_defaults_to_equal(self):
        self.assertEqual(normalize_weights(['a', 'b']), [0.5, 0.5])
        self.assertEqual(normalize_weights(['a', 'b'], {'a': 2, 'b': 1}), [2 / 3, 1 / 3])

    def test_normalize_weights_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            normalize_weights(['a', 'a'])
        with self.assertRaises(ValueError):
            normalize_weights(['a'], {'b': 1})
        with self.assertRaises(ValueError):
            normalize_weights(['a'], {'a': 0})

    def test_validate_compatible_configs(self):
        base = {
            'tokenizer': 'byte',
            'vocab_size': 256,
            'block_size': 64,
            'n_layer': 2,
            'n_head': 2,
            'n_embd': 64,
        }
        validate_compatible_configs([base, dict(base)])
        mismatch = dict(base)
        mismatch['n_embd'] = 128
        with self.assertRaises(ValueError):
            validate_compatible_configs([base, mismatch])

    def test_load_trace_reads_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'trace.json'
            path.write_text(json.dumps([{'token_id': 1}]), encoding='utf-8')
            self.assertEqual(load_trace(path), [{'token_id': 1}])


if __name__ == '__main__':
    unittest.main()
