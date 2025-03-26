import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass

        # Store input for backward pass
        self.A = A
        # Store original shape for backward pass
        self.input_shape = A.shape
        
        # Reshape input to 2D: (batch_size, in_features) where batch_size = prod(*)
        batch_size = np.prod(A.shape[:-1])
        A_2d = A.reshape(batch_size, A.shape[-1])
        
        # Compute output and reshape back to original batch dimensions
        Z = A_2d @ self.W.T + self.b
        Z = Z.reshape(*self.input_shape[:-1], -1)
        
        # Return output
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # Reshape gradients to 2D: (batch_size, out_features) where batch_size = prod(*)
        batch_size = np.prod(dLdZ.shape[:-1])
        dLdZ_2d = dLdZ.reshape(batch_size, dLdZ.shape[-1])

        # Reshape input to 2D: (batch_size, in_features) where batch_size = prod(*)
        A_2d = self.A.reshape(batch_size, self.A.shape[-1])
        
        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = dLdZ_2d @ self.W
        self.dLdW = dLdZ_2d.T @ A_2d
        self.dLdb = dLdZ_2d.sum(axis=0)
        
        # Reshape dLdA back to match input shape
        self.dLdA = self.dLdA.reshape(*self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA
