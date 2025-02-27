import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        self.softmax = Softmax(dim=-1)
        self.eps = 1e-100
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., Hq, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., Hq, L, S) or broadcastable shape
        :return: Output matrix of shape (N, ..., Hq, L, Ev)
        """
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        
        # Get dimensions
        self.E  = Q.shape[-1] # embedding dimension
        self.Ev = V.shape[-1] # value dimension
        self.L  = Q.shape[-2] # target sequence length
        self.S  = K.shape[-2] # source sequence length
        
        # Transpose K to swap last two dimensions
        K_t = np.transpose(K, (0, 1, 3, 2))  # (..., H, E, S)

        # Calculate attention scores: (..., Hq, L, S)
        attention_scores = Q @ K_t / np.sqrt(self.E) # (..., Hq, L, S)
        
        # Apply mask before softmax if provided
        attention_scores_masked = None
        if self.mask is not None:
            attention_scores_masked = attention_scores + np.where(self.mask, -self.eps, 0)
        else:
            attention_scores_masked = attention_scores

        # Apply softmax along S dimension (..., Hq, L, S)
        softmax = self.softmax.forward(attention_scores_masked)

        # Store softmax output for backward pass
        self.softmax_output = softmax
        
        # Calculate output: (..., Hq, L, Ev)
        self.output = self.softmax_output @ V
        return self.output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., Hq, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # Calculate gradients for V: (..., H, S, Ev)
        # (..., Hq, L, S) @ (..., Hq, L, Ev) -> (..., H, S, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.transpose(self.softmax_output, (0, 1, 3, 2)) @ d_output
        
        # Calculate gradients for attention scores
        V_t = np.transpose(self.V, (0, 1, 3, 2))  # (..., H, Ev, S)
        d_softmax = d_output @ V_t  # (..., Hq, L, S)
        d_attention_scores_masked = self.softmax.backward(d_softmax)
        
        # Apply mask to gradients if mask was used in forward pass
        if self.mask is not None:
            d_attention_scores = d_attention_scores_masked + np.where(self.mask, -self.eps, 0)
        else:
            d_attention_scores = d_attention_scores_masked

        # Scale gradients
        d_attention_scores = d_attention_scores / np.sqrt(self.E)
        
        # Calculate gradients for Q and K
        d_Q = d_attention_scores @ self.K  # (..., Hq, L, E)
        d_K = np.transpose(d_attention_scores, (0, 1, 3, 2)) @ self.Q  # (..., H, S, E)
        
        return d_Q, d_K, d_V

