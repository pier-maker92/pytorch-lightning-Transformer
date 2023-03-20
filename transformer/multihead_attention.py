import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        k_t = keys.transpose(2, 3)  # transpose
        scores = (queries @ k_t) / math.sqrt(d_k)  # scaled dot product

        # Apply mask to the attention scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e20)

        # Computing the weights by a softmax operation
        weights = F.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return torch.matmul(weights, values)

# Implementing the Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model%h == 0, f"Impossible to equally split embeddings for this value of h: {h}"
        self.attention = DotProductAttention()         # Scaled dot product attention
        self.n_heads = h                               # Number of attention heads to use
        self.d_k = d_model//h                          # Dimensionality of the linearly projected queries and keys
        self.d_v = d_model//h                          # Dimensionality of the linearly projected values (Let'assume the same value of queries and keys)
        self.d_model = d_model                         # Dimensionality of the model
        self.W_q = nn.Linear(d_model, self.d_k*h)      # Learned projection matrix for the queries
        self.W_k = nn.Linear(d_model, self.d_k*h)      # Learned projection matrix for the keys
        self.W_v = nn.Linear(d_model, self.d_v*h)      # Learned projection matrix for the values
        self.W_concat = nn.Linear(d_model, d_model)    # Learned projection matrix for the multi-head output
    
    def head_split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        tensor = tensor.view(batch_size, length, self.n_heads, self.d_k).transpose(1, 2)

        return tensor
    
    def head_concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        assert d_model == self.d_model, "Something foes wrong in splitting for the heads"
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        return tensor

    def forward(self, queries, keys, values, mask=None):
        # 1. dot product with weight matrices
        queries, keys, values = self.W_q(queries), self.W_k(keys), self.W_v(values)

        # 2. split tensor by number of heads
        queries, keys, values = self.head_split(queries), self.head_split(keys), self.head_split(values)

        # 3. do scale dot product to compute similarity
        out = self.attention(queries, keys, values, self.d_k, mask=mask)
        
        # 4. concat and pass to linear layer
        out = self.head_concat(out)
        out = self.W_concat(out)

        return out