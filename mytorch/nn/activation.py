import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        '''
        # TODO: Implement the forward pass for the Softmax activation function.
        # NOTE: Make sure you implement this in a numerically stable way.
        # NOTE: Make sure you apply the softmax to the dimension specified by the `dim` parameter.
        self.A = raise NotImplementedError("Forward pass not implemented")
        '''
        self.A = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A /= np.sum(self.A, axis=self.dim, keepdims=True) + 1e-8
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt post-activation output
        :return: Gradient of loss with respect to pre-activation input
        """

        '''
        # TODO: Implement the backward pass for the Softmax activation function.
        shape = self.A.shape
        C = shape[self.dim]
        self.dLdZ = raise NotImplementedError("Backward pass not implemented")
        '''
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D if necessary
        if len(shape) > 2:
            # Move target dimension to end and flatten other dimensions
            self.A = np.moveaxis(self.A, self.dim, -1)
            dLdA   = np.moveaxis(dLdA, self.dim, -1)
            # Make sure to store the shape post moving axis and pre-reshaping
            # so we can restore the original shape after computing gradients    
            shape  = self.A.shape
            self.A = self.A.reshape(-1, C)
            dLdA   = dLdA.reshape(-1, C)

        # Compute gradients
        diag_A  = np.einsum('ij,jk->ijk', self.A, np.eye(C))
        outer_A = np.einsum('ij,ik->ijk', self.A, self.A)
        J       = diag_A - outer_A
        dLdZ    = np.einsum('ij,ijk->ik', dLdA, J)

        # Reshape back to original dimensions if necessary
        if len(shape) > 2:
            # Restore shapes to original
            self.A = np.moveaxis(self.A.reshape(shape), -1, self.dim)
            dLdZ = np.moveaxis(dLdZ.reshape(shape), -1, self.dim)

        return dLdZ
 

    