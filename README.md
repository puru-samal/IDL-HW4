# HW4 Handout



## 📓 Notebooks & Core Files

This should be the directory structure from your project root directory:

```
.
├── HW4P1_nb.ipynb
├── HW4P2_nb.ipynb
├── README.md
├── hw4lib/
├── mytorch/
└── hw4_data/


```

## HW4P1

> `NOTE`: All implementations have detailed specification, implementation details, and hints in their respective source files. Make sure to read all the comments and docstrings in their entirety to understand the implementation details!

### Task 1: MyTorch Implementation

In `HW4P1` and `HW4P2`, you will build and train Transformer models using PyTorch’s `nn.MultiHeadAttention`. To deepen your understanding of its internals, you will also implement a custom `MultiHeadAttention` module from scratch as part of your `mytorch` library, designed to closely match PyTorch’s interface.

We recommend developing the components incrementally and using the command provided in the `MyTorch Implementations` cell of `HW4P1_nb.ipynb` to test your implementation.

#### 1.1 Linear Layer (`mytorch/nn/linear.py`)
There is nothing to implement for this section. Simply copy-paste your implementation of `Linear` from previous assignments into `mytorch/nn/linear.py`.

#### 1.2 Generic Softmax (`mytorch/nn/activation.py`)
In HW1P1, you implemented a `Softmax` activation function that took a 2D array of shape `(N, C)` and computed the softmax of each element over the `C` classes. In this assignment, you will implement a more general `Softmax` class in `mytorch/nn/activation.py`, where the number of dimensions of the input can be arbitrary and the softmax can be computed over any dimension. The implementation will largely be the same as HW1P1 but with a few key differences. 

##### 1.2.1 Forward Pass

Given a single input vector **Z** with shape `(C,)`, whose *m*-th element is denoted by $z_m$, the `softmax` function will return a vector **A** with shape `(C,)`, where the *m*-th element $a_m$ is given by:

$$
a_m = \frac{\exp(z_m)}{\sum\limits_{k=1}^{C} \exp(z_k)}
$$

Similar calculations would apply for any **N**-dimensional input tensor **Z**.
You must return a tensor with the same shape as **Z** but with the softmax probabilities **along the dimension specified by the `dim` parameter** defined in the `Softmax` class constructor.

##### 1.2.2 Backward Pass

For an N-dimensional input tensor, the backward pass follows similar principles to the 2D case in HW1P1. For any slice along the specified dimension `dim`, we need to compute how changes in the input affect the output probabilities.

For a slice of the input tensor along dimension `dim`, let's call the input values **z** and output probabilities **a**. The Jacobian **J** for this slice has elements given by:

$$
J_{mn} = \begin{cases}
a_m(1-a_m) & \text{if } m = n \\
-a_m a_n & \text{if } m \neq n
\end{cases}
$$

where $a_m$ refers to the m-th element of the probability vector **a**.

The gradient for this slice is then computed as:

$$
\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \cdot \mathbf{J}
$$

This calculation needs to be performed for each slice along the specified dimension while keeping all other dimensions fixed. The final gradient tensor will have the same shape as the input tensor. 

We recommend you take the following approach while implementing the backward pass which we will illustrate with an example:

If the input tensor **Z** has shape `(N, C, H, W)` and softmax was applied along dimension `C`, then:

1. Find the dimension along which softmax was applied in the forward pass: `dim = 1`
2. Move dimension 1 to the last dimension to get a tensor of shape `(N, H, W, C)`
3. Flatten the remaining dimensions to get a 2D tensor of shape `(N*H*W, C)`
4. Then you can operate on this 2D tensor the same way you did in HW1P1.
4. Finally, reshape back to 4D tensor of shape `(N, C, H, W)` and move the last dimension back to its original position.

`Hint`: You might find `np.moveaxis` helpful in this implementation.


#### 1.3 Scaled Dot-Product Attention (`mytorch/nn/scaled_dot_product_attention.py`)

Implement the scaled-dot product attention in `mytorch/nn/scaled_dot_product_attention.py` in a setting similar to what you will deal with in `HW4P1` and `HW4P2`.

##### 1.3.1 Forward Pass

Implement the `forward` method for the `ScaledDotProductAttention` class in `mytorch/nn/scaled_dot_product_attention.py`.
The scaled dot-product attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where, **Q**, **K**, and **V** are the query, key, and value matrices respectively and $d_k$ is the embedding dimension of the query and key matrices.

In our setting, we have:
- A query matrix **Q** of shape `(N, ..., Hq, L, E)`
- A key matrix **K** of shape `(N, ..., H, S, E)`
- A value matrix **V** of shape `(N, ..., H, S, Ev)`
- Optionally, we will have a boolean mask matrix **M** of shape `(N, ..., H, L, S)`

where:
- `N` is the batch size
- `Hq` is the number of attention heads for the query
- `H` is the number of attention heads for the key
- `L` is the length of the target sequence
- `S` is the length of the source sequence
- `E` is the embedding dimension
- `Ev` is the value dimension


Where, the output is of shape `(N, ..., H, L, Ev)`. If a mask if provided, be use to use it to set the attention scores prior to applying the softmax to `-self.eps` for positions that should not be attended to (i.e. `mask == True`) and leave the rest of the scores unchanged. 

- `NOTE`: Remember to store the softmax output for the backward pass.
- `NOTE`: The code refers to the input of the softmax function as the attention scores.


##### 1.3.2 Backward Pass

Implement the `backward` method for the `ScaledDotProductAttention` class in `mytorch/nn/scaled_dot_product_attention.py`. Given the gradient of some arbitrary loss with respect to the output, compute the gradients with respect to Q, K, and V matrices using the chain rule.

The backward pass follows these steps:

1. Gradient with respect to V:
$$
\frac{\partial L}{\partial V} = A^T \cdot \frac{\partial L}{\partial O}
$$

2. Gradient with respect to attention scores A:
$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} \cdot V^T
$$

3. Gradient with respect to pre-softmax inputs:
$$
\frac{\partial L}{\partial I} = \text{softmax\_backward}(\frac{\partial L}{\partial A})
$$

4. Gradient with respect to Q:
$$
\frac{\partial L}{\partial Q} = (\frac{\partial L}{\partial I} \cdot \frac{1}{\sqrt{d_k}}) \cdot K
$$

5. Gradient with respect to K:
$$
\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial I} \cdot \frac{1}{\sqrt{d_k}})^T \cdot Q
$$

Where:
- $A$: Output of the softmax function
- $\frac{\partial L}{\partial O}$: Gradient of loss with respect to output
- $\frac{\partial L}{\partial I}$: Gradient of loss with respect to softmax input
- $d_k$: Embedding dimension of query and key matrices

**Note**: If a mask was used in the forward pass, remember to apply it to $\frac{\partial L}{\partial I}$ before computing the gradients for Q and K.

**Hint**: You might find `np.transpose` useful. Let the shapes guide you.


#### 1.4 Multi-Head Attention (`mytorch/nn/multi_head_attention.py`)

Multi-head attention allows the model to jointly attend to information from different representation subspaces. You will implement the `MultiHeadAttention` class that processes queries, keys, and values through multiple parallel attention heads. You will use the `Linear`, the generic `Softmax` and `ScaledDotProductAttention` classes you implemented to build the `MultiHeadAttention` class.

##### 1.4.1 Forward Pass

Implement the `forward` method for the `MultiHeadAttention` class in `mytorch/nn/multi_head_attention.py`.  

The multi-head attention mechanism takes three inputs:
- Query matrix Q of shape (N, L, E)
- Key matrix K of shape (N, S, E)
- Value matrix V of shape (N, S, E)
- Optional key padding mask of shape (N, S)
- Optional attention mask of shape (L, S)

where:
- `N` is the batch size
- `L` is the length of the target sequence
- `S` is the length of the source sequence
- `E` is the embedding dimension

The forward pass follows these steps:

1. Project the query, key, and value inputs into the same embedding dimension using the `Linear` layers. Note: You might have to do some reshaping depending on your `Linear` implementation.

$$
Q' = W_q \cdot Q + b_q
$$

$$
K' = W_k \cdot K + b_k
$$

$$
V' = W_v \cdot V + b_v
$$

2. Implement the `_split_heads` method and use it to split the projected query, key, and value matrices into multiple heads. $H$ is the number of attention heads specified in the `MultiHeadAttention` class constructor.

$$
Q (N, L, E) \rightarrow Q'(N, H, L, E/H)
$$

$$
K (N, S, E) \rightarrow K'(N, H, S, E/H)
$$

$$
V (N, S, E) \rightarrow V'(N, H, S, E/H)
$$

3. Implement the `_merge_masks` method and use it to merge the key padding mask (N, S) and attention mask (L, S) into a single mask of shape (N, H, L, S).

4. Apply the scaled dot-product attention mechanism to each head:

$$
O' = \text{ScaledDotProductAttention}(Q', K', V', mask)
$$

4. Implement the `_merge_heads` method and use it to concatenate the attention outputs from all heads.

$$
O' (N, H, L, E/H) \rightarrow O'' (N, L, E)
$$  

5. Project the concatenated attention outputs back to the original embedding dimension to get the final output:
$$
O = W_o \cdot O'' + b_o
$$


##### 1.4.2 Backward Pass

Implement the `backward` method for the `MultiHeadAttention` class in `mytorch/nn/multi_head_attention.py`. Given the gradient of some arbitrary loss with respect to the output, compute the gradients with respect to Q, K, V.

The steps are:
1. Backpropagate through output projection
2. Split gradients into multiple heads using the `_split_heads` method
3. Backpropagate through scaled dot-product attention for each head
4. Merge heads of gradients for Q, K, V using the `_merge_heads` method 
5. Backpropagate through input projections


### Task 2: Language Modeling using a Causal Transformer Decoder

In both `HW4P1` and `HW4P2`, you will incrementally implement components of `hw4lib` to build and train two models: a Decoder-only Transformer for causal language modeling and an Encoder-Decoder Transformer for end-to-end speech recognition. The following sections will walk you through the steps needed to complete the former.


#### 2.1 Data-Processing Components

To get going, we will need some basic tools for converting raw text into sequences of the appropriate form. Typical preprocessing pipelines execute the following steps:

1. Load text as strings into memory.
2. Split the strings into tokens (e.g., words or characters).
3. Build a vocabulary dictionary to associate each vocabulary element with a numerical index.
4. Convert the text into sequences of numerical indices.

##### 2.1.1 About the `H4Tokenizer`(`hw4lib/data/tokenizer.py`)

Tokens are the smallest units of text your model will process. Each time step corresponds to one token, but what constitutes a "token" depends on the tokenization strategy you choose. For example, the sentence “Baby needs a new pair of shoes” can be represented as:

- A sequence of 7 word-level tokens, drawn from a large vocabulary (typically tens or hundreds of thousands of words).
- Or as a sequence of 30 character-level tokens, using a much smaller vocabulary (e.g., 256 ASCII characters).

For this assignment, you will convert each text transcript into:

- A sequence of tokens.
- A corresponding sequence of numerical indices, where each index represents the token's position in the vocabulary.

These numerical sequences will serve as inputs to and outputs from your model. During inference or generation, you will also need to reverse this process—converting indices back to tokens, and then reconstructing the original text.

We have provided the H4Tokenizer class in `hw4lib/data/tokenizer.py` to handle tokenization for both `HW4P1` and `HW4P2`. This tokenizer supports several strategies:

- Character-level tokenization
- Subword tokenization with a vocabulary size of 1,000
- Subword tokenization with a vocabulary size of 5,000
- Subword tokenization with a vocabulary size of 10,000

Character-level tokenization is straightforward, while subword tokenization splits words into smaller subword units. The subword method uses [Byte Pair Encoding (BPE)](https://arxiv.org/pdf/1508.07909) to learn and apply subword merges.

As part of this assignment, you will explore how different tokenization strategies affect model performance in both `HW4P1` and `HW4P2`.

Before proceeding, familiarize yourself with the H4Tokenizer class and its key methods:

- `tokenize`
- `encode`
- `decode`

You will use these methods both when preparing datasets and during model decoding.


##### 2.1.2 Dataset Implementation (`hw4lib/data/lm_dataset.py`)

For `HW4P1`, you will be working with a dataset located in the `hw4p1_data` subdirectory inside the `hw4_data` directory. The structure is organized as follows:
```
hw4_data/
├── hw4p1_data/
│ ├── train/
│ ├── valid/
│ └── test/
 ...
```
The `train, valid, and test` folders contain the dataset splits, with each split consisting of text files stored in `.npy` format.

To work with this dataset, you will use the **`LMDataset`** class provided in `hw4lib/data/lm_dataset.py`. This class is designed to:

- Load the .npy text files,
- Tokenize the sequences using a provided tokenizer,
- Prepare two versions of each tokenized sequence:
    - A **shifted** version with a Start-of-Sequence (SOS) token prepended,
    - A **golden** version with an End-of-Sequence (EOS) token appended,
- Track dataset statistics such as total characters, tokens, and sequence lengths,
- Provide a custom collate_fn function for batching data correctly.

Your task is to complete parts of the `__init__` method and fully implement the `__len__`, `__getitem__`, and `collate_fn` methods according to the provided specifications. Your implementation should ensure sequences are properly aligned, padded, and formatted for training and evaluating an autoregressive language model.

Run the command in the `Dataset Implementation` section of `HW4P1_nb.ipynb` to test your implementation incrementally.


#### 2.2 Model Implementations

The following sections will guide you through the incremental process of building a Decoder-Only Transformer model, Specifically, we will implement the pre-norm variant of the decoder-only transformer architecture.


> **NOTE**: As you incrementally implement each component, you can test your implementations using the commands provided in the `Model Implementations` section of `HW4P1_nb.ipynb`. There are also cells to enable you to visualize some of your implementations.


##### 2.2.1 Masks (`hw4lib/model/masks.py`)

Before implementing the decoder-only transformer, we need to create helper functions for masking, which will be implemented in `hw4lib/model/masks.py`.

> **NOTE**: While it's possible to implement these functions naively using a `for` loop, we recommend using vectorized operations to speed up the implementation, as these functions will be called repeatedly during training. Some useful PyTorch functions for this task include `torch.ones_like`, `torch.arange`, `torch.expand`, `torch.repeat`, `torch.tril`, and various boolean operations (~, >=, ==, !=, etc.) on tensors. It is possible to implement these functions using PyTorch's built-in functions with just 3-5 lines of code each.

###### 2.2.1.1 Causal Mask

Implement the **`CausalMask`** function in `hw4lib/model/masks.py`. This function should take a padded batch of input sequences with shape `(N, T, ...)` or `(N, T)` and return a mask of shape `(T, T)`, where `T` is the sequence length. The mask should be `True` for positions that should be masked (i.e., positions that should not attend to future tokens), and `False` for positions that are allowed.

For example, if we have a batch of two sequences with variable lengths, padded to the same length (with `0` as the padding token):


$$
\text{input} = \begin{bmatrix}
1 & 2 & 3 & 0 & 0 \\
1 & 2 & 0 & 0 & 0
\end{bmatrix}
$$  

Then the output mask will be:   

$$
\text{mask} = \begin{bmatrix}
False & True & True & True & True \\
False & False & True & True & True \\
False & False & False & True & True \\
False & False & False & False & True \\
False & False & False & False & False
\end{bmatrix}
$$

> **NOTE**: As with the previous example, while this illustrates the `(N, T)` case, your function should also support the `(N, T, ...)` case.

###### 2.2.1.2 Pad Mask

Implement the **`PadMask`** function in `hw4lib/model/masks.py`. This function should take a padded batch of input sequences with shape `(N, T, ...)` or `(N, T)` and a tensor of input lengths with shape `(N,)`, and return a mask of shape `(N, T)` where `N` is the batch size and `T` is the sequence length. The mask should be `True` for padding positions and `False` for non-padding positions.

For example, if we have a batch of two sequences of variable lengths, padded to the same length within the batch (with `0` as the padding token):


$$
\text{input} = \begin{bmatrix}
1 & 2 & 3 & 0 & 0 \\
1 & 2 & 0 & 0 & 0
\end{bmatrix}
$$

And the lengths tensor:

$$
\text{lengths} = \begin{bmatrix}
3 \\
2
\end{bmatrix}
$$  

Then the output mask will be:

$$
\text{mask} = \begin{bmatrix}
False & False & False & True & True \\
False & False & True & True & True
\end{bmatrix}
$$

> **NOTE**: While this example illustrates the `(N, T)` case, your function implementation should also support the `(N, T, ...)` case.


##### 2.2.2 Positional Encoding (`hw4lib/model/positional_encoding.py`)

The Transformer model does not rely on recurrence or convolution, treating all tokens in a sequence as independent of one another. To inject information about token positions into the model, **positional encoding** is used to retain the notion of order.

In this section, you will implement a fixed positional encoding scheme based on sine and cosine functions, as introduced in the original Transformer paper ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

Given an input representation consisting of $d$-dimensional embeddings for $T$ tokens, positional encoding generates a matrix $\mathbf{P} \in \mathbb{R}^{T \times d}$ of the same shape. The elements of $\mathbf{P}$ are computed as follows:

$$
\mathbf{P}_{t, 2i} = \sin\left(\frac{t}{10000^{\frac{2i}{d}}}\right), \quad \mathbf{P}_{t, 2i+1} = \cos\left(\frac{t}{10000^{\frac{2i}{d}}}\right)
$$

Implement the following methods of the `PositionalEncoding` class in `hw4lib/model/positional_encoding.py`:

- **`create_pe_table`**: Constructs the positional encoding matrix $\mathbf{P}$ of shape `(max_len, d_model)` where:
  - `max_len`: The maximum length of input sequences.
  - `d_model`: The dimensionality of each token embedding.

- **`forward`**: Adds the positional encoding matrix $\mathbf{P}$ to the input embeddings before feeding them into the model.

> **NOTE**: Remember that the positional encoding matrix should be registered as a buffer (e.g., using `self.register_buffer()`) to ensure it is not treated as a learnable parameter.

After implementing the `create_pe_table` and `forward` methods, you can test your implementation using the commands provided in the `Positional Encoding` subsection of the `Model Implementations` cell of `HW4P1_nb.ipynb`.

##### 2.2.3 Transformer Sublayers (`hw4lib/model/sublayers.py`)

In the following sections, you will implement the sublayers of the layer's present in the pre-norm variant of the decoder-only transformer. We recommend you familiarize yourself with PyTorch's [`nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) class before implementing the sublayers.

###### 2.2.3.1 SelfAttentionLayer

Implement the `SelfAttentionLayer` class in `hw4lib/model/sublayers.py`. Refer to the diagram below for the architecture of the layer. This class will contain the logic for the self-attention mechanism.

<img src="figures/selfattnlayer.png" alt="Transformer diagram" width="200" height="400"/>


###### 2.2.3.2 FeedForwardLayer

Implement the `FeedForwardLayer` class in `hw4lib/model/sublayers.py`. Refer to the diagram below for the architecture of the layer.

<img src="figures/ffnlayer.png" alt="Transformer diagram" width="200" height="400"/>

> **NOTE**: Implement the `FeedForwardNetwork` part by setting the `ffn` attribute to a `nn.Sequential` module consisting of two linear layers with a GELU activation and dropout in between, where the input of dimension `d_model` is first projected to a higher-dimensional space `d_ff`, non-linearly transformed, regularized via dropout, and then projected back to `d_model`. The `FeedForwardLayer` class should inherit from `nn.Module` and implement the `forward` method. **YOU MUST FOLLOW THIS SPECIFICATION EXACTLY TO BE COMPATIBLE WITH THE TEST SUITE.**

##### 2.2.4 Transformer Self-Attention Decoder Layer (`hw4lib/model/decoder_layers.py`)

Implement the `SelfAttentionDecoderLayer` class in `hw4lib/model/decoder_layers.py`. Refer to the diagram below for the architecture of the layer. This class just contains the SelfAttentionLayer and FeedForwardLayer as submodules.

<img src="figures/decselfattnlayer.png" alt="Transformer diagram" width="200" height="400"/>


##### 2.2.5 Decoder-Only Transformer (`hw4lib/model/transformers.py`)

Finally, put all the pieces you've implemented together to implement the Decoder-Only Transformer. Implement the `DecoderOnlyTransformer` class in `hw4lib/model/transformers.py`. Refer to the diagram below for the architecture of the model.

<img src="figures/transf-dec-only.png" alt="Transformer diagram" width="200" height="400"/>


#### 2.3 Decoding Implementation

Implement the `generate_greedy` method of the `SequenceGenerator` class in `hw4lib/decoding/sequence_generator.py`. Run the command in the `Decoding Implementation` section of `HW4P1_nb.ipynb` to test your implementation. 

Use the following pseudocode as a guide:

```text
function generate_greedy(x, temperature, repeat_penalty):
    
    # Initialize scores and flags
    initialize scores as zeros of shape (batch_size,)
    initialize finished flags as False for each sequence
    
    for t in range(max_length - current_sequence_length):
        if all sequences are finished:
            break

        logits = score_fn(x)
        logits = apply_repeat_penalty(logits, x, repeat_penalty)
        logits = logits / temperature
        log_probs = log_softmax applied to raw logits

        next_tokens = tokens with highest log_probs for each sequence
        token_scores = log_probs corresponding to next_tokens

        update scores only for unfinished sequences
        
        append next_tokens to x
        
        update finished flags to True if EOS token is generated for that sequence

    return x, scores
```

Here:
- `x`: Input tensor of shape `(batch_size, seq_len)` representing token sequences of the same length and no padding.
- `temperature`: Scalar value used to scale logits before selecting the next token.
- `repeat_penalty`: Scalar factor applied to penalize repeated tokens during decoding.
- `score_fn`: Function that takes the current sequences `x` of shape `(batch_size, seq_len)` and returns logits of shape `(batch_size, vocab_size)` for the next token.
- `apply_repeat_penalty`: Function that modifies logits to penalize repeated tokens; input and output shapes are `(batch_size, vocab_size)`.

For `score_fn` and `apply_repeat_penalty`, you can use the `score_fn` attribute and `_apply_repeat_penalty` method of the `SequenceGenerator` class.

> **NOTE**: While it's possible to implement these functions naively using a `for` loop, we recommend using vectorized operations to speed up the implementation, as these functions will be called repeatedly during inference. Some useful PyTorch functions for this task include `torch.zeros`, `torch.zeros_like`, `torch.all`, `torch.where`, `torch.gather`, `torch.cat`, `torch.log_softmax`, and various boolean operations (|, ==, !=, etc.) on tensors. It is possible to implement each line of the pseudocode using just 1-2 lines of code.

#### 2.4 LMTrainer (`hw4lib/trainers/lm_trainer.py`)

## HW4P2



## 📊 Dataset Structure



```

hw4_data_subset/
├── hw4p1_data/
│ ├── train/
│ ├── valid/
│ └── test/
└── hw4p2_data/
├── dev-clean/
│ ├── fbank/
│ └── text/
├── test-clean/
│ ├── fbank/
│ └── text/
└── train-clean-100/
├── fbank/
└── text/

```

## 🔧 Implementation Files

### Main Library (hw4lib/)

```

hw4lib/
├── data/
│ ├── tokenizer_jsons/
│ ├── asr_dataset.py
│ ├── lm_dataset.py
│ └── tokenizer.py
├── decoding/
│ └── sequence_generator.py
├── model/
│ ├── masks.py
│ ├── positional_encoding.py
│ ├── speech_embedding.py
│ ├── sublayers.py
│ ├── decoder_layers.py
│ ├── encoder_layers.py
│ └── transformers.py
└── trainers/
├── base_trainer.py
├── asr_trainer.py
└── lm_trainer.py

```

### MyTorch Library Components (mytorch/)

```

mytorch/nn/
├── activation.py
├── linear.py
├── scaled_dot_product_attention.py
└── multi_head_attention.py

```

### Test Suite (tests/)

```

tests/
├── testing_framework.py
├── test_mytorch*.py
├── test_dataset*.py
├── test_mask*.py
├── test_positional_encoding.py
├── test_sublayers*.py
├── test_encoderlayers*.py
├── test_decoderlayers*.py
├── test_transformers\*.py
├── test_hw4p1.py
└── test_decoding.py

```

# Internal TODO:

- [ ] Remove `tests/test_hw4p1.py` from the handout
- [ ] Update threshold in `tests/test_hw4p1.py` and re-run `make create_autograde`
- [ ] Replace autolab's `autograde.tar` after threshold is updated
