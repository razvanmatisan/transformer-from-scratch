import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,
            heads: int = 8,
            d_model: int = 128,
    ):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0, "Embedding dimension must be divisible by the number of attention heads!"
        
        self.heads = heads
        self.d_model = d_model

        self.d_k = d_model // heads
        self.d_v = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)

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

        # (bs, seq_len, d_model)
        q = self.q_proj(x) 
        k = self.k_proj(x)
        v = self.v_proj(x) 

        # (bs, heads, seq_len, d_k)
        q = q.reshape(bs, seq_len, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = k.reshape(bs, seq_len, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = v.reshape(bs, seq_len, self.heads, self.d_k).permute(0, 2, 1, 3)
        
        # (bs, heads, seq_len, d_v)
        attentions = self.scaled_dot_product(q, k, v)
        
        # (bs, heads, d_v * heads = d_model)
        attentions = attentions.permute(0, 2, 1, 3).reshape(bs, seq_len, self.d_model)
        out = self.o_proj(attentions)
        
        return out
    

if __name__ == "__main__":
    bs = 2
    seq_len = 10
    n_heads = 8
    d_model = 128

    MHA = MultiHeadAttention(n_head=n_heads, d_model=d_model)
    x = torch.randn(bs, seq_len, d_model)

    print(MHA(x).shape)
