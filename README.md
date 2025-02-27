# HW4 Handout Directory Structure

The handout contains the following key components:

## 📓 Notebooks & Core Files

```
├── HW4P1_nb.ipynb
├── HW4P2_nb.ipynb
├── README.md
├── Makefile
├── autograde-Makefile
├── simulate_autolab.py
└── autograde.tar
```

## 📊 Dataset Structure

```
hw4_data_subset/
├── dev-clean/
│   ├── fbank/
│   └── text/
├── test-clean/
│   ├── fbank/
│   └── text/
├── train-clean-100/
│   ├── fbank/
│   └── text/
└── text-for-LM/
    ├── train-text/
    ├── valid-text/
    └── test-text/
```

## 🔧 Implementation Files

### Main Library (hw4lib/)

```
hw4lib/
├── data/
│   ├── tokenizer_jsons/
│   ├── asr_dataset.py
│   ├── lm_dataset.py
│   └── tokenizer.py
├── decoding/
│   └── sequence_generator.py
├── model/
│   ├── masks.py
│   ├── positional_encoding.py
│   ├── speech_embedding.py
│   ├── sublayers.py
│   ├── decoder_layers.py
│   ├── encoder_layers.py
│   └── transformers.py
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
├── test_transformers*.py
├── test_hw4p1.py
└── test_decoding.py
```

# TODO:

- [ ] Remove `tests/test_hw4p1.py` from the handout
- [ ] Update threshold in `tests/test_hw4p1.py` and re-run `make create_autograde`
- [ ] Replace autolab's `autograde.tar` after threshold is updated
