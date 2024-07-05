import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 source_vocab_size: int = 5000,
                 target_vocab_size: int = 5000,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 n_heads: int = 8,
                 d_model: int = 512,
                 d_hidden: int = 2048,
                 max_seq_len: int = 5000,
                 pad_source_idx: int = 0, # pad token id
                 pad_target_idx: int = 0, # pad token id
                 is_positional_encodings: bool = True
                 ):
        super(Transformer, self).__init__()

        self.pad_source_idx = pad_source_idx
        self.pad_target_idx = pad_target_idx

        self.source_embeddings = nn.Embedding(source_vocab_size, d_model)
        self.target_embeddings = nn.Embedding(target_vocab_size, d_model)
        
        self.positional_encodings = PositionalEncoding(max_seq_len=max_seq_len, d_model=d_model) if is_positional_encodings else None
        
        self.encoder = Encoder(
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_hidden=d_hidden,
        )

        self.decoder = Decoder(
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_hidden=d_hidden,
        )

        self.fc_out = nn.Linear(d_model, target_vocab_size)


    def create_source_mask(self, source):
        bs, seq_len = source.shape
        mask = (source != self.pad_source_idx).unsqueeze(1).unsqueeze(2).expand(
            bs, 1, 1, seq_len
        ).float()

        # (bs, 1, 1, seq_len)
        return mask

    def create_target_mask(self, target):
        bs, seq_len = target.shape
        causal_mask = torch.tril(torch.ones((seq_len, seq_len))).expand(
            bs, 1, seq_len, seq_len
        ).int()

        padding_mask = (target != self.pad_target_idx).unsqueeze(1).unsqueeze(2).expand(
            bs, 1, seq_len, seq_len
        ).int()

        mask = (causal_mask & padding_mask).float()

        return mask

    def forward(self, source, target):
        ### Encoder
        # Get encoder mask (self-attention)
        source_mask = self.create_source_mask(source)

        # Get input embeddings
        source = self.source_embeddings(source)
        
        # Add positional encodings
        if self.positional_encodings is not None:
            source = self.positional_encodings(source)

        # Get the output of the encoder
        source = self.encoder(source, source_mask)

        # Get key, value from last encoder layer
        bs, seq_len, _ = source.shape

        last_encoder_layer = self.encoder.encoder_layers[-1]
        qkv = last_encoder_layer.mha.qkv_proj(source)
        qkv = qkv.reshape(bs, seq_len, last_encoder_layer.mha.heads, 3 * last_encoder_layer.mha.d_k).permute(0, 2, 1, 3)
        _, key_source, value_source = torch.split(qkv, last_encoder_layer.mha.d_k, dim=-1)
        
        ### Decoder
        # Get decoder mask (masked self-attention)
        target_mask = self.create_target_mask(target)

        # Get output embeddings
        target = self.target_embeddings(target)

        # Add positional encodings
        if self.positional_encodings is not None:
            target = self.positional_encodings(target)

        # Get the output of the decoder        
        out = self.decoder(target, key_source, value_source, target_mask, source_mask)

        ### Linear layer + Softmax
        out = self.fc_out(out)
        out = torch.softmax(out, dim=-1)

        return out
