import torch
from torch import nn
import torch.nn.functional as F


class ModelArgs:
    """
    A class to hold model arguments.
    """
    dim: int = 128
    max_len: int = 1024
    block_size: int = 8
    batch_size: int = 4

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

def split_dataset(data: torch.tensor, train_ratio: float = 0.9):
    """
    Split the dataset into training and validation sets.
    """
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data

def get_batch(mode: str):
    """
    Get a batch of data for training or validation.
    """
    if mode == "train":
        data = train_data
    elif mode == "val":
        data = val_data
    else:
        raise ValueError("Mode must be 'train' or 'val'.")

    batch_size = ModelArgs.batch_size
    block_size = ModelArgs.block_size
    input_data = torch.empty((batch_size, block_size), dtype=torch.long)
    target_data = torch.empty((batch_size, block_size), dtype=torch.long)
    for b in range(batch_size):
        start = torch.randint(0, len(data) - block_size - 1, (1,)).item()
        seq = data[start:start + block_size + 1]
        input_data[b] = seq[:-1]
        target_data[b] = seq[1:]

    return input_data, target_data

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
    x = detokenize(tokenize(text[:100], encode_vocab), decode_vocab)

    train_data, val_data = split_dataset(tokenize(text[:100], encode_vocab))
    print(detokenize(train_data, decode_vocab))
    print("////////")
    print(detokenize(val_data, decode_vocab))
    x, y = get_batch("train")
    print(x, y)
    for i in range(x.shape[0]):
        print(detokenize(x[i], decode_vocab))
        print("////////")
    