import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .encoder import Encoder


class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len, dropout=.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout) # in_place=True

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)

        # standard Positional Embedding 
        position = torch.arange(0, max_len).unsqueeze(1) 

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x
    
class Transformer(nn.Module):
    """
    Full transformer, implemented in pytorch lightning.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, src_pad_token, tgt_pad_token, n_layers=6, n_heads=8, d_ff=2048, d_model=512, dropout=.1):
        super(Transformer, self).__init__()
        # embeddings 
        self.embeddings = nn.ModuleDict({
            'src': Embeddings(src_vocab_size, d_model),
            'tgt': Embeddings(tgt_vocab_size, d_model)
        })
        # positions
        self.positional_encoding = nn.ModuleDict({
            'src': PositionalEncoding(d_model, src_seq_len, dropout),
            'tgt': PositionalEncoding(d_model, tgt_seq_len, dropout)
        })

        # encoder
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
        # decoder
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout)

        # pad tokens
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token

    def forward(self, x_src, x_tgt):
        # create masks
        src_mask = self.make_src_mask(x_src)
        #src_trg_mask = self.make_pad_mask(x_tgt, x_src, trg_pad_token, src_pad_token)
        trg_mask = self.make_trg_mask(x_tgt)

        # Pick propers embeddings end encode position info
        x_src = self.positional_encoding['src'](self.embeddings['src'](x_src))
        x_tgt = self.positional_encoding['tgt'](self.embeddings['tgt'](x_tgt))
      
        # Feed the encoder
        x_src_encoded = self.encoder(x_src, src_mask)
        # Feed the decoder
        output = self.decoder(x_src_encoded, x_tgt, trg_mask, src_mask ) # src_trg_mask

        return output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_token).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(src.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(trg.device)
    

