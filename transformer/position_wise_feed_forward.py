import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model, dropout):
        super(FeedForward, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(d_model, d_ff),     # First fully connected layer
            nn.GELU(), 
            nn.Dropout(dropout),  
            nn.Linear(d_ff, d_model)      # Second fully connected layer
        )
    def forward(self, x):
        return self.block(x)