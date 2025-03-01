import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary
'''
TODO: Implement these Modules.

This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
   - Contains a stack of SelfAttentionDecoderLayers
   - Uses causal masking to prevent attending to future tokens
   - Includes optional weight tying and layer dropout features

    Key components to implement:
    1. Token Embedding Layer: Convert token IDs to vectors
    2. Positional Encoding: Add position information
    3. Decoder Stack: Process tokens sequentially
    4. Output Projection: Convert final representations to logits

    Architecture follows Pre-LN (Layer Normalization) design where:
    - Layer normalization is applied at the start of each sublayer
    - Residual connections wrap around each sublayer
    - Final layer norm is applied before output projection

    Implementation Notes:
    1. The forward pass should handle:
    - Proper masking (both padding and causal)
    - Collecting attention weights from all layers
    - Optional layer dropout during training
    
    2. The score method should:
    - Handle single token prediction
    - Not apply padding masks
    - Return only the final token's logits
'''

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        '''
        super().__init__()

        # Initialize the decoder
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        # TODO: Create a module list of decoder layers based on the number of layers
        self.dec_layers     = nn.ModuleList(
            [SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # TODO: Create target embedding and other layers
        self.target_embedding       = nn.Embedding(num_classes, d_model)
        self.positional_encoding    = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.final_linear           = nn.Linear(d_model, num_classes)
        self.dropout                = nn.Dropout(dropout)
        self.norm                   = nn.LayerNorm(d_model)

        # TODO: Initialize weights
        self.initialize_weights()

        # TODO: Weight tying
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight 

    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
        # DO NOT MODIFY 
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        # TODO: Create padding mask for padded_targets (use PadMask)
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = PadMask(padded_targets, target_lengths).to(padded_targets.device)
        
        # TODO: Create causal mask to prevent attending to future tokens (use CausalMask)
        causal_mask = CausalMask(padded_targets).to(padded_targets.device) 

        # TODO: Apply the embedding
        x = self.target_embedding(padded_targets)
        # TODO: Apply positional encoding
        x = self.positional_encoding(x)
        # TODO: Apply dropout 
        x = self.dropout(x)

        # TODO: Pass through all decoder layers, save attention masks
        runnint_att = {}
        for i in range(self.num_layers):
            # TODO: Optionally apply LayerDrop during training
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            # TODO: Pass through decoder layer
            x, attention = self.dec_layers[i](
                x,
                key_padding_mask=pad_mask_dec,
                attn_mask=causal_mask
            )
            # TODO: Save attention weights  
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention

        # TODO: Apply normalization
        x = self.norm(x)
        # TODO: Linear layer (Final Projection) for next character prediction
        seq_out = self.final_linear(x)
        # TODO: Return the output sequence and attention weights
        return seq_out, runnint_att
    
    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder. 
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded. 
        Can only handle batch_size = 1 or batch with same length and no padding. 
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        # TODO: Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # TODO: Return the last token's logits for next token prediction    
        logits     = seq_out[:, -1, :]
        return logits
    
    def initialize_weights(self):
        """
        Initialize the weights of the model using Xavier initialization for linear layers,
        normal distribution for embeddings, and scaled initialization for attention layers.
        """
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Normal distribution initialization for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # Layer norm weights are 1, biases are 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize all attention weights with Xavier uniform
                # Scale by 1/sqrt(head_dim) for stable training
                if hasattr(module, 'in_proj_weight'):
                    head_dim = module.head_dim
                    scale = (1.0 / (head_dim ** 0.5))
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=scale)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                
                # Initialize output projection
                if hasattr(module, 'out_proj'):
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
                
        self.apply(_init_weights)

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int, 
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.

        Args:
            input_dim: int, dimension of input speech features
            time_reduction: int, stride along time dimension, the amount of reduction to apply to the time dimension
            reduction_method: Literal['lstm', 'conv', 'both'], the source_embedding reduction method
            num_encoder_layers: int, number of encoder layers
            num_encoder_heads: int, number of encoder attention heads
            d_ff_encoder: int, feed-forward dimension for encoder
            num_decoder_layers: int, number of decoder layers
            num_decoder_heads: int, number of decoder attention heads
            d_ff_decoder: int, feed-forward dimension for decoder
            d_model: int, model dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
            skip_encoder_pe: bool, whether to skip positional encoding for encoder (default: False)
            skip_decoder_pe: bool, whether to skip positional encoding for decoder (default: False)
        '''
        super().__init__()

        # Initialize model attributes
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        # TODO: Create encoder layers
        self.enc_layers = nn.ModuleList(
            [SelfAttentionEncoderLayer(d_model, num_encoder_heads, d_ff_encoder, dropout) for _ in range(num_encoder_layers)]
        )

        # TODO: Create decoder layers
        self.dec_layers = nn.ModuleList(
            [CrossAttentionDecoderLayer(d_model, num_decoder_heads, d_ff_decoder, dropout) for _ in range(num_decoder_layers)]
        )

        # TODO: Create source and target embeddings and other layers
        self.source_embedding = SpeechEmbedding(
            input_dim=input_dim,
            output_dim=d_model,
            time_reduction=time_reduction,
            reduction_method=reduction_method,
            dropout=dropout
        )
        self.target_embedding    = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.final_linear        = nn.Linear(d_model, num_classes)
        self.dropout             = nn.Dropout(dropout)
        self.encoder_norm        = nn.LayerNorm(d_model)
        self.decoder_norm        = nn.LayerNorm(d_model)
        self.ctc_head            = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        # TODO: Initialize weights
        self.initialize_weights()

        # TODO: Weight tying if enabled
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        '''
        Encode the source features.
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
        Returns:
            x_enc: Encoded representation. shape: (batch_size, src_len, d_model)
            pad_mask_src: Source padding mask
            running_att: Dictionary containing encoder self-attention weights
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''
        # TODO: Apply speech embedding, positional encoding, and dropout
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)
        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)
        x_enc = self.dropout(x_enc)

        pad_mask_src = PadMask(x_enc, x_enc_lengths).to(x_enc.device)  

        # TODO: Pass through encoder layers
        running_att = {}
        for i in range(self.num_encoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_enc, attention = self.enc_layers[i](x_enc, pad_mask_src)
            running_att[f'layer{i+1}_enc_self'] = attention

        # TODO: Apply normalization
        x_enc = self.encoder_norm(x_enc)
        # TODO: Project to CTC logits
        ctc_logits = self.ctc_head(x_enc.permute(1, 0, 2))

        return x_enc, pad_mask_src, running_att, {'log_probs': ctc_logits, 'lengths': x_enc_lengths}

    def decode(
        self, 
        padded_targets: torch.Tensor, 
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Decode using encoder output and target sequence.
        Args:
            padded_targets: The padded target sequence. shape: (batch_size, tgt_len)
            encoder_output: Output from encoder. shape: (batch_size, src_len, d_model)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            pad_mask_src: Source padding mask from encoder
        Returns:
            seq_out: The output sequence. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing decoder attention weights
        '''
        # TODO: Create target padding mask
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = PadMask(padded_targets, target_lengths).to(padded_targets.device)

        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None, unless you are using the decoder as a standalone model or doing inference, you should provide target_lengths")

        # TODO: Create causal mask
        causal_mask = CausalMask(padded_targets).to(padded_targets.device)

        # TODO: Apply the embedding, positional encoding, and dropout
        x_dec = self.target_embedding(padded_targets)
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        # TODO: Pass through decoder layers
        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            x_dec, self_attn, cross_attn = self.dec_layers[i](
                x_dec,
                encoder_output,
                pad_mask_tgt,
                pad_mask_src,
                causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        # TODO: Final normalization and projection
        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)

        return seq_out, running_att

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the encoder-decoder transformer.
        
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            padded_targets: The padded target sequences. shape: (batch_size, tgt_len)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            
        Returns:
            seq_out: The output sequence logits. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing all attention weights from both encoder and decoder
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''
        # During training, we need target lengths
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")

        # TODO: Encode the source sequence
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(padded_sources, source_lengths)
        
        # TODO: Decode using encoder output
        seq_out, dec_running_att = self.decode(
            padded_targets,
            encoder_output,
            target_lengths,
            pad_mask_src
        )
        
        # TODO: Combine attention dictionaries
        running_att = {**enc_running_att, **dec_running_att}
        
        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor, pad_mask_src: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder given encoder output.
        Args:
            batch_prompts: tensor of token sequences to score for next token. shape: (batch_size, seq_len)
            encoder_output: encoder output tensor. shape: (batch_size, src_len, d_model)
            pad_mask_src: source padding mask. shape: (batch_size, src_len)
        Returns:
            logits: Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # TODO: Use decode function with no target lengths (no padding mask for targets)
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        
        # Return only the last token's logits
        return seq_out[:, -1, :]

    def initialize_weights(self):
        """
        Initialize the weights of the model using Xavier initialization for linear layers,
        normal distribution for embeddings, and scaled initialization for attention layers.
        """
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight'):
                    head_dim = module.head_dim
                    scale = (1.0 / (head_dim ** 0.5))
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=scale)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                
                if hasattr(module, 'out_proj'):
                    nn.init.xavier_uniform_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
                
        self.apply(_init_weights)

    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
        freeze_transferred: bool = False,
        decoder_lr_factor: float = 0.1
    ) -> Tuple['EncoderDecoderTransformer', list]:
        """
        Create an encoder-decoder transformer with decoder weights initialized from a pretrained decoder-only model.
        
        Args:
            decoder_checkpoint_path: Path to decoder-only transformer checkpoint
            config: Configuration dictionary for the encoder-decoder model
            freeze_transferred: Whether to freeze transferred parameters
            decoder_lr_factor: Learning rate multiplier for transferred parameters
            
        Returns:
            model: Initialized encoder-decoder transformer
            param_groups: List of parameter groups for optimizer
        """
        # Create new encoder-decoder model
        model = cls(
            input_dim=config['input_dim'],
            time_stride=config['time_stride'],
            feature_stride=config['feature_stride'],
            lstm_layers=config['lstm_layers'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_len=config['max_len'],
            num_classes=config['num_classes'],
            weight_tying=config.get('weight_tying', False),
            layer_drop_rate=config.get('layer_drop_rate', 0.0)
        )

        # Load decoder checkpoint
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu')
        decoder_state_dict = checkpoint['model_state_dict']
        
        # Track parameters for optimizer groups
        transferred_params = []
        new_params = []
        
        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v 
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            target_module.load_state_dict(module_state_dict)
            if freeze_transferred:
                for param in target_module.parameters():
                    param.requires_grad = False
            else:
                transferred_params.extend(target_module.parameters())

        # Transfer shared components
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')
        
        # Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        
        for i in range(num_layers):
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )
        
        # Collect new parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                is_new = True
                for transferred_param in transferred_params:
                    if param is transferred_param:
                        is_new = False
                        break
                if is_new:
                    new_params.append(param)
        
        # Create parameter groups
        param_groups = []
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr_factor': 1.0,
                'name': 'new_params'
            })
        
        if not freeze_transferred and transferred_params:
            param_groups.append({
                'params': transferred_params,
                'lr_factor': decoder_lr_factor,
                'name': 'transferred_params'
            })
        
        return model, param_groups

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


def setup_asr_model(config):
    # Create model with pretrained decoder weights
    model, param_groups = EncoderDecoderTransformer.from_pretrained_decoder(
        decoder_checkpoint_path=config['training']['decoder_checkpoint'],
        config=config['model'],
        freeze_transferred=config['training'].get('freeze_transferred', False),
        decoder_lr_factor=config['training'].get('decoder_lr_factor', 0.1)
    )
    
    # Log parameter groups
    model.log_param_groups(param_groups)
    
    # Create optimizer with parameter groups
    base_lr = config['training']['learning_rate']
    optimizer = torch.optim.Adam([
        {
            'params': group['params'],
            'lr': base_lr * group['lr_factor']
        }
        for group in param_groups
    ])
    
    return model, optimizer


def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(num_layers: int = 12, num_heads: int = 8, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1, max_len: int = 300, num_classes: int = 1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "__main__":
    test_decoder_only()