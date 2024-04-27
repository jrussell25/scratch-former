# scratch-former

Getting my hands dirty building a language model from scratch.

Intall with ` pip install -e .` (only dependence is `pytorch`).
See [Usage](#usage) and [Data Preparation](#data-preparation) below.

## Introduction

Language models are increasing in power and usefulness at an incredible rate.
As a user of the commercial LLMs with some background in ML I have been tracking
the key developments at a high level but I felt there were some missing pieces in
my understanding when it came to actually getting an LM to work. This project set
out to address those gaps.

One major issue to address up front is what it means to make something "from scratch,"
after all as Carl Sagan says, "If you wish to make an apple pie from scratch, you must
first invent the universe" and something similar is definitely true for transformer
models.

This implementation uses only what I consider to be core elements of pytorch to create
a gpt-2 like model. Core elements boils down to the autograd capability of torch
functions, `torch.nn.Module` for building the model, the `Adam` optimizer, and some
simple layers and activations (Linear, LayerNorm, ReLU). It is modular enough to support
building different text generation models and tries to strike a balance between
readability and functionality.

### Primary model

### Tokenization - A major cut corner

### Sampling strategies

### Other learnings

### Interesting follow ups

## Practicalities

### Data Preparation

### Usage
