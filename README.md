# **HW4 Handout**

In this assignment, we are introducing a new format for assignment delivery, designed to enhance your development workflow. The key motivations for this change are:  

- **Test Suite Integration**: Your code will be tested in a manner similar to `HWP1's`.  
- **Local Development**: You will be able to perform most of your initial development locally, reducing the need for compute resources.  
- **Hands-on Experience**: This assignment provides an opportunity to build an end-to-end deep learning pipeline from scratch. We will be substantially reducing the amount of abstractions compared to previous assignments.   
  
For our provided notebook's to work, your notebook's current working directory must be the same as the handout.
This is important because the relative imports in the notebook's depend on the current working directory.
This can be achieved by:
1. Physically moving the notebook's into the handout directory.
2. Changing the notebook's current working directory to the handout directory using the `os.chdir()` function.

Your current working directory should have the following files for this assignment: 

```
.
├── README.md
├── hw4lib/
├── mytorch/
├── tests/
├── hw4_data_subset/
└── requirements.txt
```

## 📊 Dataset Structure

We have provided a subset of the dataset for you to use. This subset has been provided with the intention of allowing you to implement and test your code locally. The subset follows the same structure as the original dataset and is organized as follows:

```
hw4_data_subset/
├── hw4p1_data/ # For causal language modeling
│ ├── train/
│ ├── valid/
│ └── test/
└── hw4p2_data/ # For end-to-end speech recognition
├── dev-clean/
│ ├── fbank/
│ └── text/
├── test-clean/
│ └── fbank/
└── train-clean-100/
├── fbank/
└── text/

```

## 🔧 Implementation Files

### Main Library (`hw4lib/`)
For `HW4P1` and `HW4P2`, you will incrementally implement components of `hw4lib` to build and train two models:  

- **HW4P1**: A *Decoder-only Transformer* for causal language modeling.  
- **HW4P2**: An *Encoder-Decoder Transformer* for end-to-end speech recognition.  

Many of the components you implement will be reusable across both parts, reinforcing modular design and efficient implementation. You should see the following files in the `hw4lib/` directory (`__init__.py`'s are not shown):  

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
├── trainers/
│  ├── base_trainer.py
│  ├── asr_trainer.py
│  └── lm_trainer.py
└── utils/
   ├── create_lr_scheduler.py
   └── create_optimizer.py
```

### MyTorch Library Components (`mytorch/`)
In `HW4P1` and `HW4P2`, you will build and train Transformer models using PyTorch’s `nn.MultiHeadAttention`. To deepen your understanding of its internals, you will also implement a custom `MultiHeadAttention` module from scratch as part of your `mytorch` library, designed to closely match the PyTorch interface. You should see the following files in the `mytorch/` directory:

```

mytorch/nn/
├── activation.py
├── linear.py
├── scaled_dot_product_attention.py
└── multi_head_attention.py

```

### Test Suite (`tests/`)
In `HW4P1` and `HW4P2`, you will be provided with a test suite that will be used to test your implementation. You should see the following files in the `tests/` directory:

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

---

Developed by: Puru Samal