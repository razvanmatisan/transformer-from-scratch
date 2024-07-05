import torch.nn as nn
import torch
from sublayers import MultiHeadAttention, FFN

class Decoder(nn.Module):
    def __init__(self, n_layers: int = 6, n_heads: int = 8, d_model: int = 512, d_hidden: int = 2048):
        super(Decoder, self).__init__()

        decoder_layers = [DecoderLayer(n_heads, d_model, d_hidden) for _ in range(n_layers)]
        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, target, key_source, value_source, target_mask, source_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(target, key_source, value_source, source_mask, target_mask)
            
        return x


class DecoderLayer(nn.Module):
    def __init__(self, heads: int = 8, d_model: int = 512, d_hidden: int = 2048, dropout_probability: int = 0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model

        self.masked_mha = MultiHeadAttention(heads=heads, d_model=d_model)
        self.mha = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = FFN(d_model=d_model, d_hidden=d_hidden)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, key, value, source_mask, target_mask):
        # Causal attention
        x = self.layer_norm1(x + self.dropout(self.mha(x, target_mask)))

        # Unmasked attention
        bs, seq_len, _ = x.shape
        qkv = self.mha.qkv_proj(x)
        qkv = qkv.reshape(bs, seq_len, self.mha.heads, 3 * self.mha.d_k).permute(0, 2, 1, 3)
        query, _, _ = torch.split(qkv, self.mha.d_k, dim=-1)

        x = self.mha.scaled_dot_product(query, key, value, source_mask)
        x = x.permute(0, 2, 1, 3).reshape(bs, seq_len, self.d_model)
        x = self.mha.o_proj(x)

        x = self.layer_norm2(x + self.dropout(x))

        # FFN
        x = self.layer_norm3(x + self.dropout(self.ffn(x)))

        return x
