import tempfile
import unittest
from pathlib import Path

from src.data_prep import discover_files, prepare_dataset, read_manifest


class DatasetPreparationTests(unittest.TestCase):
    def test_prepares_supported_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'source'
            source.mkdir()
            (source / 'notes.md').write_text('# Notes\nhello loom', encoding='utf-8')
            (source / 'records.jsonl').write_text('{"text": "second document"}\n', encoding='utf-8')
            (source / 'ignored.bin').write_bytes(b'ignore me')

            manifest = prepare_dataset(source, root / 'prepared', 'demo')
            corpus = Path(manifest.output_file).read_text(encoding='utf-8')

            self.assertEqual(manifest.file_count, 2)
            self.assertIn('hello loom', corpus)
            self.assertIn('second document', corpus)
            self.assertEqual(read_manifest(root / 'prepared').sha256, manifest.sha256)

    def test_missing_source_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            discover_files('missing-dataset-folder')

    def test_prepares_a_single_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'notes.txt'
            source.write_text('one small file', encoding='utf-8')

            manifest = prepare_dataset(source, root / 'prepared', 'single')

            self.assertEqual(manifest.file_count, 1)
            self.assertIn('one small file', Path(manifest.output_file).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
