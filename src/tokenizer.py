class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids)
    
    '''
from src.tokenizer import CharTokenizer
>> text = open('data/input.txt').read()
>> tok = CharTokenizer(text)
>> print('Vocab size:', tok.vocab_size)
>> print('All characters:', ''.join(sorted(tok.stoi.keys())))
>> print()
>> test = 'Hello, World!'
>> encoded = tok.encode(test)
>> decoded = tok.decode(encoded)
>> print(f'Input:   {test}')
>> print(f'Encoded: {encoded}')
>> print(f'Decoded: {decoded}')
>> print(f'Round-trip OK: {test == decoded}')
>> "
These are for testing the tokeniser using the dataset. 


The expected outcome will be :
Vocab size: 65
All characters:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz

Input:   Hello, World!
Encoded: [20, 43, 50, 50, 53, 6, 1, 35, 53, 56, 50, 42, 2]
Decoded: Hello, World!
Round-trip OK: True

    '''