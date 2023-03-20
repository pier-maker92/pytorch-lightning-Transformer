import torch.nn as nn
from .add_norm import AddNormalization
from .position_wise_feed_forward import FeedForward
from .multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.feed_forward = FeedForward(d_ff, d_model, dropout)
        self.add_norm_layers = nn.ModuleList([AddNormalization(d_model, dropout) for i in range(2)])
        
    def forward(self, x, padding_mask=None):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        
        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm_layers[0](multihead_output,x) #self.add_norm_layers[0]
 
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
 
        # Followed by another Add & Norm layer
        return self.add_norm_layers[1](feedforward_output,addnorm_output) #self.add_norm_layers[1]

class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(h, d_model)
        self.feed_forward = FeedForward(d_ff, d_model, dropout)
        self.add_norm_layers = nn.ModuleList([AddNormalization(d_model, dropout) for i in range(3)])
        #self.add_norm_layers = AddNormalization(d_model, dropout)

    def forward(self, x_dec, x_enc, tgt_mask, src_mask):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x_dec, x_dec, x_dec, tgt_mask)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm_layers[0](multihead_output,x_dec) #self.add_norm_layers[0]

        if x_enc is not None:
            # Multi-head attention encoder - decoder
            multihead_output = self.multihead_attention(addnorm_output, x_enc, x_enc, src_mask)

            # Followed by an Add & Norm layer
            addnorm_output = self.add_norm_layers[1](multihead_output,addnorm_output) #self.add_norm_layers[1]

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        
        # Followed by another Add & Norm layer
        return self.add_norm_layers[2](feedforward_output,addnorm_output) #self.add_norm_layers[2]