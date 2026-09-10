# ELMo From Scratch

This project is a simplified implementation of **[ELMo](https://github.com/adi18-ui/ELMo-fron-scratch/blob/main/ELMo%20paper.pdf)(Embeddings from Language Models)** using PyTorch.

The model creates contextual word embeddings, meaning that the representation of a word changes according to the sentence in which it appears.

## Dataset

The model was trained using the **[WikiText-2 Raw dataset](https://huggingface.co/datasets/Salesforce/wikitext)(only a subsection of it, not the full dataset)**. It contains text collected from Wikipedia articles and is commonly used for language-modelling experiments.

## Architecture

The model follows these steps:

```text
Character IDs
    ↓
Character Embedding
    ↓
Character CNN + Max Pooling
    ↓
Highway Layer
    ↓
Forward and Backward LSTMs
    ↓
Contextual Word Embeddings
```

The forward LSTM predicts the next word(left -> right direction), while the backward LSTM(right -> left direction) predicts the previous word.

## Features

* Character-level word representation
* Character CNN
* Max pooling
* Highway layer
* Forward and backward language models
* Next-word and previous-word prediction

## Note

The 1st version is a simplified implementation created from scratch to understand how ELMo works. It does not reproduce every detail of the original large ELMo architecture. The model is trained only for 100 epochs and with fewer LSTM units because of GPU limitations.


