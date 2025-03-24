from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers  
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()

        self.q_proj   = Linear(embed_dim, embed_dim)
        self.k_proj   = Linear(embed_dim, embed_dim)
        self.v_proj   = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """

        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]

        # Reshape q, k, v to (-1, embed_dim) to apply mytorch's linear layer
        query = query.reshape(-1, self.E)
        key   = key.reshape(-1, self.E)
        value = value.reshape(-1, self.E)
        
        # Project the input into query, key, and value
        q = self.q_proj.forward(query) # (N*L, embed_dim)
        k = self.k_proj.forward(key)   # (N*S, embed_dim)
        v = self.v_proj.forward(value) # (N*S, embed_dim)

        # Reshape q, k, v back to (N, L, embed_dim)
        q = q.reshape(self.N, self.L, self.E) # (N, L, embed_dim)   
        k = k.reshape(self.N, self.S, self.E) # (N, S, embed_dim)
        v = v.reshape(self.N, self.S, self.E) # (N, S, embed_dim)

        # Split the input into multiple heads
        q = self._split_heads(q) # (N, num_heads, L, embed_dim // num_heads)
        k = self._split_heads(k) # (N, num_heads, S, embed_dim // num_heads)
        v = self._split_heads(v) # (N, num_heads, S, embed_dim // num_heads)

        # Merge the masks
        mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply the attention mechanism
        attn_outputs = self.attention.forward(q, k, v, mask=mask) # (N, num_heads, L, embed_dim // num_heads)

        # Merge the attention outputs   
        attn_output = self._merge_heads(attn_outputs)             # (N, L, embed_dim)

        # Reshape attn_output to (N*L, embed_dim) to apply mytorch's linear layer
        attn_output = attn_output.reshape(-1, self.E)

        # Project the attention outputs
        output = self.out_proj.forward(attn_output)        # (N, L, embed_dim)

        # Reshape output back to (N, L, embed_dim)
        output = output.reshape(self.N, self.L, self.E)
        return output   

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """

        # Reshape d_output to (N*L, embed_dim) to backpropagate through mytorch's linear layer
        d_output = d_output.reshape(-1, self.E) # (N*L, embed_dim)

        # Backpropagate through the output projection   
        d_attn_output = self.out_proj.backward(d_output)  # (N*L, embed_dim) 

        # Reshape d_attn_output back to (N, L, embed_dim)
        d_attn_output = d_attn_output.reshape(self.N, self.L, self.E) # (N, L, embed_dim)   

        # Split the gradients into multiple heads
        d_attn_outputs = self._split_heads(d_attn_output)        # (N, num_heads, L, embed_dim // num_heads)

        # Backpropagate through the attention mechanism
        d_q, d_k, d_v = self.attention.backward(d_attn_outputs)  # (N, num_heads, L, embed_dim // num_heads)

        # Merge the gradients
        d_q = self._merge_heads(d_q) # (N, L, embed_dim)    
        d_k = self._merge_heads(d_k) # (N, S, embed_dim)
        d_v = self._merge_heads(d_v) # (N, S, embed_dim)

        # Reshape d_q, d_k, d_v back to (N*L, embed_dim) to backpropagate through mytorch's linear layer
        d_q = d_q.reshape(-1, self.E) # (N*L, embed_dim)
        d_k = d_k.reshape(-1, self.E) # (N*S, embed_dim)
        d_v = d_v.reshape(-1, self.E) # (N*S, embed_dim)

        # Backpropagate through the input projections   
        d_q = self.q_proj.backward(d_q) # (N, L, embed_dim)
        d_k = self.k_proj.backward(d_k) # (N, S, embed_dim)
        d_v = self.v_proj.backward(d_v) # (N, S, embed_dim)

        # Reshape d_q, d_k, d_v back to (N, L, embed_dim), (N, S, embed_dim), (N, S, embed_dim)
        d_q = d_q.reshape(self.N, self.L, self.E) # (N, L, embed_dim)
        d_k = d_k.reshape(self.N, self.S, self.E) # (N, S, embed_dim)
        d_v = d_v.reshape(self.N, self.S, self.E) # (N, S, embed_dim)

        return d_q, d_k, d_v

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        N = key_padding_mask.shape[0]
        L = attn_mask.shape[0]
        S = key_padding_mask.shape[1]
        H = self.num_heads
        
        # Expand key_padding_mask to (N, 1, 1, S) and broadcast to (N, H, L, S)
        key_mask = np.broadcast_to(key_padding_mask[:, np.newaxis, np.newaxis, :], (N, H, L, S))
        
        # Expand attn_mask to (1, 1, L, S) and broadcast to (N, H, L, S)
        attention_mask = np.broadcast_to(attn_mask[np.newaxis, np.newaxis, :, :], (N, H, L, S))
        
        # Combine masks using logical_or - if either mask is True, we want to mask that position
        combined_mask = np.logical_or(key_mask, attention_mask)
        
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.transpose(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        """
        Merge the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x