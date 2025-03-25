import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # DO NOT MODIFY
        self.softmax = Softmax(dim=-1)
        self.eps = 1e-4
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # Store inputs for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        # Get dimensions
        self.E  = Q.shape[-1] # embedding dimension
        self.Ev = V.shape[-1] # value dimension
        self.L  = Q.shape[-2] # target sequence length
        self.S  = K.shape[-2] # source sequence length
        
        # Calculate attention scores: (..., H, L, S)
        # (..., H, L, E) @ (..., H, E, S) -> (..., H, L, S)
        scaled_dot_product = Q @  np.transpose(K, (0, 1, 3, 2)) / np.sqrt(self.E)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if self.mask is not None:
            scaled_dot_product = scaled_dot_product + np.where(self.mask, -self.eps, 0)

        # Apply softmax along S dimension (..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)

        # Store softmax output for backward pass
        self.softmax_output = self.attention_scores
        
        # Calculate output: (..., H, L, Ev)
        # (..., H, L, S) @ (..., H, S, Ev) -> (..., H, L, Ev) 
        self.output = self.attention_scores @ V
        return self.output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # Calculate gradients for V: (..., H, S, Ev)
        # (..., H, L, S) @ (..., H, S, Ev) -> (..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.transpose(self.attention_scores, (0, 1, 3, 2)) @ d_output
        
        # Calculate gradients for attention scores
        # (..., H, L, Ev) @ (..., H, Ev, S) -> (..., H, L, S)
        d_attention_scores = d_output @ np.transpose(self.V, (0, 1, 3, 2))
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.E)
        
        # Calculate gradients for Q and K
        # (..., H, L, S) @ (..., H, S, E) -> (..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K 
        # (..., H, L, S) @ (..., H, L, E) -> (..., H, S, E)
        d_K = np.transpose(d_scaled_dot_product, (0, 1, 3, 2)) @ self.Q 
        
        return d_Q, d_K, d_V

