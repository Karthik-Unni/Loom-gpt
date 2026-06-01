import unittest

from src.tokenizer import ByteTokenizer, CharTokenizer, create_tokenizer


class TokenizerTests(unittest.TestCase):
    def test_byte_tokenizer_round_trip_supports_unicode(self):
        tokenizer = ByteTokenizer()
        text = 'LOOM works with cafe and नमस्ते'
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_character_tokenizer_remains_available(self):
        tokenizer = CharTokenizer('abc ')
        self.assertEqual(tokenizer.decode(tokenizer.encode('cab')), 'cab')
        self.assertIsInstance(create_tokenizer('char', 'abc'), CharTokenizer)


if __name__ == '__main__':
    unittest.main()
