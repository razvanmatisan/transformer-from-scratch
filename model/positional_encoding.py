import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int = 5000, d_model: int = 512):
        super(PositionalEncoding, self).__init__()        
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.positional_encodings = self.get_sinusoidal_positional_encodings()
    
    def get_sinusoidal_positional_encodings(self):
        positional_encodings = torch.zeros(self.max_seq_len, self.d_model)
        
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        positional_encodings[:, 0::2] = torch.sin(position * division_term)
        positional_encodings[:, 1::2] = torch.cos(position * division_term)

        # for allowing batch operations
        positional_encodings = positional_encodings.unsqueeze(0)

        return positional_encodings

    def forward(self, x):
        x = x + self.positional_encodings[:, :x.shape[1], :]

        return x
