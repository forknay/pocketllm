import torch
from torch import nn
import torch.nn.functional as F


class ModelArgs:
    """
    A class to hold model arguments.
    """
    dim: int = 384
    max_len = 5000 # TO change but i fucked up my weights so its saved and i will not change it now (anw not using it until context extension)
    block_size: int = 256
    batch_size: int = 64
    lr = 3e-4
    qkv_dim: int = dim*2
    eps = 1e-6
    vocab_size = 65
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@torch.no_grad()
def loss_calculation():
    """
    Estimate the loss
    """
    model.eval()
    out = {'train': 0, 'val': 0}
    for split in ['train', 'val']:
        for i in range(10):
            x, y = get_batch(split)
            _, loss = model(x, y)
            out[split] += loss.item()
    model.train()
    return {k: v / 100 for k, v in out.items()}

class Embedding(nn.Module):
    """
    Embedding layer for the model.
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.vocab_size = ModelArgs.vocab_size
        self.dim = ModelArgs.dim
        self.max_len = ModelArgs.max_len
        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.position_embedding = nn.Embedding(self.max_len, self.dim)

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

class Block(nn.Module):
    """
    A single block of the transformer model.
    Contains MHA and MLP layers with normalization.
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.mha = MHA(ModelArgs)
        self.mlp = MLP(ModelArgs)
        self.attn_norm = RMSNorm(ModelArgs)
        self.ffn_norm = RMSNorm(ModelArgs)

    def forward(self, x: torch.Tensor):
        x = x + self.mha(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    """
    Transformer model with multiple layers of MHA and MLP (returns the logits)
    """
    def __init__(self, ModelArgs):
        super().__init__()
        self.dim = ModelArgs.dim
        self.n_layers = ModelArgs.n_layers
        self.embed = Embedding(ModelArgs)
        self.blocks = nn.ModuleList([Block(ModelArgs) for _ in range(self.n_layers)])
        self.end_norm = RMSNorm(ModelArgs)
        self.head = nn.Linear(ModelArgs.dim, ModelArgs.vocab_size)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        x = self.embed(x)  # (B, S, D)
        for l in range(self.n_layers):
            x = self.blocks[l](x) 
        x = self.end_norm(x) # Take last token juiced up with all the info
        logits = self.head(x)
        # Note the shape of logits is different depending on if we have a target/loss
        if target is not None:
            B, S, D = logits.shape
            logits = logits.view(B * S, D)  # Reshape to (B * S, D) (cross entropy is being difficult)
            loss = F.cross_entropy(logits, target.flatten()) 
        else:
            loss = None
        return logits, loss
    @torch.no_grad()
    def generate(self, x, max_length):
        """
        Generate text using the model.
        """
        self.eval()
        for _ in range(max_length):
            if x.size(1) >= ModelArgs.block_size:
                x = x[:, -ModelArgs.block_size:]  # Keep only the last max_len tokens
            logits, _ = self(x)
            logits = logits[:, -1, :]  # Get the last token's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, next_token), dim=1) # (B, T+1)
        self.train()
        return x


if __name__ == "__main__":
    print(ModelArgs.device)
    with open("input.txt", "r", encoding="utf-8") as file:
        text = file.read()

    encode_vocab, decode_vocab, vocab_size = build_vocab(text)
    assert vocab_size == ModelArgs.vocab_size, "Vocabulary size mismatch."
    train_data, val_data = split_dataset(tokenize(text, encode_vocab))

    # Training Loop
    model = Transformer(ModelArgs).to(device=ModelArgs.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ModelArgs.lr)
    
    import os
    if os.path.exists("model_weights.pth"):
        model.load_state_dict(torch.load("model_weights.pth", map_location=ModelArgs.device))
        print("Model weights loaded from 'model_weights.pth'.")
    """
    model.train()
    nb_iters = 1000
    for i in range(nb_iters):
        x, y = get_batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item(), "  i =", i)
    
    torch.save(model.state_dict(), "model_weights.pth")
    """
    loss_dict = loss_calculation()
    print(f"Train Loss: {loss_dict['train']}, Validation Loss: {loss_dict['val']}, Iterations: {2}")
        
    # Generate
    generated = model.generate(torch.zeros((1,1), dtype=torch.long, device=ModelArgs.device), max_length=500)[0]
    print(detokenize(generated, decode_vocab))