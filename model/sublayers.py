import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int = 8, d_model: int = 512):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0, "Embedding dimension must be divisible by the number of attention heads!"
        
        self.heads = heads
        self.d_model = d_model

        self.d_k = d_model // heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v):
        # Transpouse of K:: (bs, heads, seq_len, d_k) -> (bs, heads, d_k, seq_len)
        k_T = torch.einsum('bhld->bhdl', k) 
        
        # Q * K_T / sqrt(d_k): (bs, heads, seq_len, seq_len)
        logits = torch.einsum('bhij,bhjk->bhik', q, k_T) / math.sqrt(self.d_k)

        # Softmax: (bs, heads, seq_len, seq_len)
        probs = torch.softmax(logits, dim=-1)

        # logits * V: (bs, heads, seq_len, d_v)
        attentions = torch.einsum('bhij,bhjk->bhik', probs, v)

        return attentions

    def forward(self, x):
        bs, seq_len, _ = x.shape

        # (bs, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # (bs, heads, seq_len, 3 * d_k) --> 3 x (bs, heads, seq_len, d_k)
        qkv = qkv.reshape(bs, seq_len, self.heads, 3 * self.d_k).permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv, self.d_k, dim=-1)

        # (bs, heads, seq_len, d_v)
        attentions = self.scaled_dot_product(q, k, v)
        
        # (bs, heads, d_v * heads = d_model)
        attentions = attentions.permute(0, 2, 1, 3).reshape(bs, seq_len, self.d_model)
        out = self.o_proj(attentions)
        
        return out
    
class FFN(nn.Module):
    def __init__(self, d_model: int = 512, d_hidden: int = 2048):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    bs = 1
    seq_len = 10
    n_heads = 8
    d_model = 128
    d_hidden = 256

    x = torch.randn(bs, seq_len, d_model)

    mha = MultiHeadAttention(heads=n_heads, d_model=d_model)
    attentions = mha(x)

    ffn = FFN(d_model=d_model, d_hidden=d_hidden)

    print(ffn(attentions).shape)
