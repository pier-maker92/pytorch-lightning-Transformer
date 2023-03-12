import torch.nn as nn
from .enc_dec_layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model, n, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(h, d_model, d_ff, dropout) for _ in range(n)])

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x