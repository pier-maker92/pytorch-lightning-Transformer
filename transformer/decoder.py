import torch.nn as nn
from .enc_dec_layers import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(h, d_model, d_ff, dropout) for _ in range(n)])
        self.linear = nn.Linear(d_model,tgt_vocab_size)

    def forward(self, x_dec, x_enc, tgt_mask, src_padding_mask):
        for layer in self.layers:
            x_dec = layer(x_dec, x_enc, tgt_mask, src_padding_mask)

        # pass to LM head
        output = self.linear(x_dec)
        return output