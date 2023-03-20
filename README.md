# pytorch-lightning-Transformer
This is an unofficial implementation of the paper "Attention is all you need" https://arxiv.org/abs/1706.03762

## Installation

Clone the repo, then create a conda envirnoment from `envirnoment.yml` and install the dependecies.

```bash
  conda env create --file=environment.yml
  conda activate transformer
  pip install -r requirements.txt
```

## Usage

You will find an example of translation task (en-it) in the folder train_example.

```python
cd train_example
python train_translator.py -b [batch_size] --lr [learning_rate]
```
