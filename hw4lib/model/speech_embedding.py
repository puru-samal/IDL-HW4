import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

'''
This file implements speech feature embedding for ASR (Automatic Speech Recognition) tasks.
It contains three key components:

1. BiLSTMEmbedding: Multi-layer bidirectional LSTM for sequence processing
   - Processes input sequences using bidirectional LSTM layers
   - Handles variable-length sequences using packed sequence operations
   - Returns processed sequences maintaining original lengths

2. Conv2DSubsampling: Convolutional downsampling for speech features
   - Reduces sequence length and feature dimensions using strided convolutions
   - Configurable stride factors for both time and feature dimensions
   - Uses GELU activation and optional dropout
   - Converts convolutional features back to sequence form

3. SpeechEmbedding: Complete speech feature processor
   - Combines Conv2DSubsampling and optional BiLSTM processing
   - Handles both feature processing and length calculations
   - Supports CNN-only processing by setting lstm_layers=0

Key Features:
- Flexible downsampling ratios through factorized strides
- Proper handling of sequence padding and lengths
- Optional multi-layer LSTM processing
- Configurable kernel sizes and dropout

Architecture Flow:
1. Input features (batch_size, seq_len, input_dim)
2. Conv2D downsampling with GELU activation
3. Optional BiLSTM processing
4. Output features with adjusted sequence lengths
'''

## -------------------------------------------------------------------------------------------------
## BiLSTMEmbedding Class
## -------------------------------------------------------------------------------------------------    
class BiLSTMEmbedding(nn.Module):
    '''
    Multi-layer BiLSTM Network 
    '''
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(BiLSTMEmbedding, self).__init__()
        self.bilstm = nn.LSTM(
                input_dim, output_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
        )

    def forward(self, x,  x_len):
        """
        Args:
            x.    : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)
        # Unpack the sequence to restore the original padded shape
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True, total_length=x.shape[1])
        return output, output_lengths

## -------------------------------------------------------------------------------------------------
## Conv2DSubsampling Class
## -------------------------------------------------------------------------------------------------    
class Conv2DSubsampling(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, 
                 time_stride: int = 2, feature_stride: int = 2, kernel_size: int = 3):
        """
        Conv2dSubsampling module with configurable downsampling for time and feature dimensions.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            dropout (float): Dropout rate (default: 0.0)
            time_stride (int): Total stride along the time dimension (default: 2)
            feature_stride (int): Total stride along the feature dimension (default: 2)
            kernel_size (int): Size of the convolutional kernel (default: 3)
        """
        super(Conv2DSubsampling, self).__init__()
        
        if not all(x > 0 for x in [input_dim, output_dim, time_stride, feature_stride, kernel_size]):
            raise ValueError("All dimension and stride values must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be between 0 and 1")

        self.kernel_size = kernel_size
        self.time_stride1, self.time_stride2       = self.closest_factors(time_stride)
        self.feature_stride1, self.feature_stride2 = self.closest_factors(feature_stride)

        self.feature_stride = feature_stride
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, output_dim, kernel_size=3, stride=(self.time_stride1, self.feature_stride1)),
            torch.nn.GELU(),
            torch.nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=(self.time_stride2, self.feature_stride2)),
            torch.nn.GELU(),
        )

        # Calculate output dimension for the linear layer
        conv_out_dim = self.calculate_downsampled_length(input_dim, self.feature_stride1, self.feature_stride2)
        conv_out_dim = output_dim * conv_out_dim
        self.out = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x, x_len):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            x_len (torch.Tensor): Non-padded lengths (batch_size)

        Returns:
            torch.Tensor: Downsampled output of shape (batch_size, new_seq_len, output_dim).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv2D
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x_len = self.calculate_downsampled_length(x_len, self.time_stride1, self.time_stride2)
        return x, x_len

    def closest_factors(self, n):
        factor = int(n**0.5)
        while n % factor != 0:
            factor -= 1
        # Return the factor pair
        return max(factor, n // factor), min(factor, n // factor)
    
    def calculate_downsampled_length(self, lengths: torch.Tensor, stride1: int, stride2: int) -> torch.Tensor:
        """
        Calculate the downsampled length for a given sequence length and strides.
        
        Args:
            lengths (torch.Tensor): Original sequence lengths (batch_size)
            stride1 (int): Stride for first conv layer
            stride2 (int): Stride for second conv layer
            
        Returns:
            torch.Tensor: Length after downsampling (batch_size)
        """ 
        lengths = (lengths - (self.kernel_size - 1) - 1) // stride1 + 1
        lengths = (lengths - (self.kernel_size - 1) - 1) // stride2 + 1
        return lengths

## -------------------------------------------------------------------------------------------------
## SpeechEmbedding Class
## -------------------------------------------------------------------------------------------------        
class SpeechEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, time_stride: int, 
                 feature_stride: int, dropout: float, lstm_layers: int = 1):
        """
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            time_stride (int): Stride along time dimension
            feature_stride (int): Stride along feature dimension
            dropout (float): Dropout rate
            lstm_layers (int): Number of LSTM layers (0 means no LSTM)
        """
        super(SpeechEmbedding, self).__init__()
        
        if not all(x > 0 for x in [input_dim, output_dim, time_stride, feature_stride]):
            raise ValueError("All dimension and stride values must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        if lstm_layers < 0:
            raise ValueError("lstm_layers must be non-negative")

        self.embedding_dim = output_dim
        self.cnn = Conv2DSubsampling(input_dim, self.embedding_dim, dropout=dropout, 
                                   time_stride=time_stride, feature_stride=feature_stride)
        self.blstm = BiLSTMEmbedding(self.embedding_dim, self.embedding_dim, 
                                    num_layers=lstm_layers) if lstm_layers > 0 else None

    def forward(self, x, x_len):
        """
        Args:
            x     : Input tensor (batch_size, seq_len, input_dim)
            x_len : Non-padded lengths (batch_size)
        Returns:
            tuple: (output tensor (batch_size, seq_len // stride, output_dim),
                   downsampled lengths (batch_size))
        """
        # First, apply Conv2D subsampling
        x, x_len = self.cnn(x, x_len)
        # Apply BiLSTM if it exists
        if self.blstm is not None:
            x, x_len = self.blstm(x, x_len)
        return x, x_len
    
    def calculate_downsampled_length(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate the downsampled length for a given sequence length.
        
        Args:
            lengths (torch.Tensor): Original sequence lengths (batch_size)
            
        Returns:
            torch.Tensor: Length after downsampling (batch_size)
        """
        return self.cnn.calculate_downsampled_length(lengths, self.cnn.time_stride1, self.cnn.time_stride2)