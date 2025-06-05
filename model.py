import torch
from torch import nn
import torch.nn.functional as F


class ModelArgs:
    """
    A class to hold model arguments.
    """
    dim: int = 32
    max_len = 512
    block_size: int = 8
    batch_size: int = 4
    lr = 1e-3
    qkv_dim: int = 64
    eps = 1e-6
    vocab_size = 65
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

def build_vocab(text: str):
    """
    Build a character-level vocabulary from the text.
    """
    encode = {}
    decode = {}
    text = sorted(list(set(text)))
    print(text[0:10])
    for char in text:
        if char not in encode: # And by extension not in decode
            encode[char] = len(encode)
            decode[len(decode)] = char

    assert len(encode) == len(decode), "Encoding and decoding dictionaries must have the same length."

    return encode, decode, len(encode)

def tokenize(text: str, encode: dict):
    """"
    Convert text to tokens using the vocabulary.
    """
    tokens = []  
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

def split_dataset(data: torch.Tensor, train_ratio: float = 0.9):
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
    """
    Embedding layer for the model.
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.vocab_size = ModelArgs.vocab_size
        self.dim = ModelArgs.dim
        self.block_size = ModelArgs.block_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.position_embedding = nn.Embedding(self.block_size, self.dim)

    def forward(self, x: torch.Tensor):
        
        return self.token_embedding(x) + self.position_embedding(torch.arange(x.size(1), device=x.device))

class RMSNorm(nn.Module):
    """"
    Normalization layer
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.dim = ModelArgs.dim
        self.eps = ModelArgs.eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MHA(nn.Module):
    """
    Multi-head attention Layer
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.dim = ModelArgs.dim
        self.qkv_dim = ModelArgs.qkv_dim
        self.n_heads = ModelArgs.n_heads
        self.head_dim = self.qkv_dim // self.n_heads
        self.wq = nn.Linear(self.dim, self.qkv_dim, bias=False) # (H, D, C)
        self.wk = nn.Linear(self.dim, self.qkv_dim, bias=False) # (H, D, C)
        self.wv = nn.Linear(self.dim, self.qkv_dim, bias=False) # (H, D, C)
        self.wo = nn.Linear(self.qkv_dim, self.dim, bias=False) # (C, D)
        self.softmax_scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(ModelArgs.dropout)


    def forward(self, x: torch.Tensor, mask: bool = True):
        """
        Forward pass for the attention head.
        """
        batch_size, seq_len, _ = x.size() # (B, S, D) (not worrying about multiple heads yet)
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, C)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q,k.transpose(-2, -1)) * self.softmax_scale # (B, H, S, S)
        if mask:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            scores = scores.masked_fill(attention_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf")) 
        
        scores = self.dropout(F.softmax(scores, dim=-1))
        x = torch.matmul(scores, v) # (B, H, S, C)
        x = self.wo(x.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_heads * self.head_dim)) # (B, S, D)

        return x

class MLP(nn.Module):
    """
    FFN Layer
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.dim = ModelArgs.dim
        self.hidden_dim = ModelArgs.dim * 4
        self.w1 = nn.Linear(self.dim, self.hidden_dim)
        self.w2 = nn.Linear(self.hidden_dim, self.dim)
        self.w3 = nn.Linear(self.dim, self.hidden_dim)
        self.dropout = nn.Dropout(ModelArgs.dropout)

    def forward(self, x: torch.Tensor):
        """
        SwiGLU (im a deepseek enjoyer)
        """
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        return self.w2(x)

    

if __name__ == "__main__":

    file = open("input.txt", "r", encoding="utf-8")
    text = file.read()
    encode_vocab, decode_vocab, vocab_size = build_vocab(text)
    assert vocab_size == ModelArgs.vocab_size, "Vocabulary size mismatch."


    x = detokenize(tokenize(text[:100], encode_vocab), decode_vocab)

    train_data, val_data = split_dataset(tokenize(text[:100], encode_vocab))
    print(detokenize(train_data, decode_vocab))
    print("////////")
    print(detokenize(val_data, decode_vocab))
    x, y = get_batch("train")
    
    embed = Embedding(ModelArgs)
    print(embed(x).shape)  # Should print (batch_size, block_size, dim)
    norm = RMSNorm(ModelArgs)
    attention = MHA(ModelArgs)
    x = norm(embed(x))
    print(x.shape)
    print(x[0][0])
    x = attention(x)
    print(x[0][0])
    print(x.shape)