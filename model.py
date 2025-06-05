import torch
from torch import nn
import torch.nn.functional as F


class ModelArgs:
    """
    A class to hold model arguments.
    """
    dim: int = 512
    max_len: int = 1024

def build_vocab(text: str):
    """
    Build a character-level vocabulary from the text.
    """
    encode = {}
    decode = {}
    text = sorted(list(set(text)))
    for char in text:
        if char not in encode: # And by extension not in decode
            encode[char] = len(encode) + 1
            decode[len(decode) + 1] = char
    encode["<start>"] = 0
    decode[0] = "<start>"
    assert len(encode) == len(decode), "Encoding and decoding dictionaries must have the same length."

    return encode, decode, len(encode)

def tokenize(text: str, encode: dict):
    """"
    Convert text to tokens using the vocabulary.
    """
    tokens = [0]  # Start token
    for char in text:
        tokens.append(encode[char])

    return torch.tensor(tokens, dtype=torch.long)

def detokenize(tokens: list, decode: dict):
    """
    Convert tokens back to text using the vocabulary.
    """
    char_list = []
    for token in tokens:
        char_list.append(decode[token.item()])

    return "".join(char_list)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.embedding(x)





if __name__ == "__main__":

    file = open("input.txt", "r", encoding="utf-8")
    text = file.read()
    encode_vocab, decode_vocab, vocab_size = build_vocab(text)
    print(detokenize(tokenize(text[:100], encode_vocab), decode_vocab))

