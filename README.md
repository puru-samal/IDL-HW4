# Setup

Create Conda Environment (Both PSC and Local)

```bash
# Create Conda Environment
module load anaconda # PSC only
conda create -n your_env_name python=3.12.4

# Deactivate Conda Environment
conda deactivate

# Activate Conda Environment
conda activate your_env_name

# Install Dependencies
pip install -r requirements.txt
```

# PSC Instructions

Load the anaconda module:

```bash
# Load the anaconda module
module load anaconda

# Activate Conda Environment
conda activate your_env_name

# Start an 8 hour interact session:
interact -p GPU-shared --gres=gpu:v100-32:1 -t 8:00:00
```

Note: Sometimes PSC automatically reverts to the default environment (`base`). If this happens, you can deactivate the default environment and activate your environment again with:

```bash
conda deactivate
conda activate your_env_name
```

`Optional`: Open a `tmux` session:

```bash
tmux new -s your_session_name
```

`Optional`: Download the dataset into the node's `$LOCAL` directory for faster disk access (storage is not persistent):

```bash
cd $LOCAL
python download_data.py
tar -xvzf hw4_data.tar.gz
rm hw4_data.tar.gz
cd path/to/project
```

# HW4 Handout Directory Structure

This should be the directory structure from your projectroot directory:

## 📓 Notebooks & Core Files

```
.
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

# TODO:

- [ ] Remove `tests/test_hw4p1.py` from the handout
- [ ] Update threshold in `tests/test_hw4p1.py` and re-run `make create_autograde`
- [ ] Replace autolab's `autograde.tar` after threshold is updated
