import torch.nn as nn

class AddNormalization(nn.Module):
    def __init__(self,features, dropout=.1):
        super(AddNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(features)  # Layer normalization layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        add = self.dropout(x) + y
        return self.layer_norm(add)