import torch.nn as nn
from sublayers import MultiHeadAttention, FFN

class Encoder(nn.Module):
    def __init__(self, n_layers: int = 6, n_heads: int = 8, d_model: int = 512, d_hidden: int = 2048, dropout_probability: int = 0.1):
        super(Encoder, self).__init__()

        encoder_layers = [EncoderLayer(n_heads, d_model, d_hidden, dropout_probability) for _ in range(n_layers)]
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, x, mask=None):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        return x


class EncoderLayer(nn.Module):
    def __init__(self, heads: int = 8, d_model: int = 512, d_hidden: int = 2048, dropout_probability: int = 0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(heads=heads, d_model=d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, d_hidden=d_hidden)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_probability)


    def forward(self, x, mask=None):
        x = self.layer_norm1(x + self.dropout(self.mha(x, mask)))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))

        return x
