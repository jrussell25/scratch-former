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

### Model

The model I train here is a 4-layer, decoder-only transformer. The Multi-Headed Self
Attention (MHSA) layer follows the description in Vaswani(?) 2017 [CITE]. And uses
`d_model=512`, `d_ff=1024`, and a fixed sequence length of 256 tokens.

#### Training

- Follows LR schedule described in Vaswani 2017. Also uses label smoothing as described.
  No dropout for now.
- GPT-2 style prediction task with masked input.
- Dataset - Wikitext-2, randomly sample sequences. Join articles with `<|endoftext|>`.


#### Results

### Inference Strategies

### Other learnings
- BPE Tokenization -- a very clever solution and a major cut corner
- Cloud computing --

### Interesting follow ups

Doing this project made me want to make training and inference more efficient.
After a bit of research it seems like a lot of the ideas I was considering were rolled
into Llama(2).

- KV caching - Can this work with fixed sinusoid PE? Or need relative/rotary encoding
- Mixed precision - save memory during training and speed up inference. Also a
  preliminary read suggests this may be a 1-2 line adjustment with `torch.amp`
- Multi-Query attention - got this from reading the Llama2 paper but pretty amazing that
  it has such a minimal effect on model quality.
- Multi-GPU data parallel - I have avoided this for years but seems essential to take
  scaling law type steps.

The software engineer in me also got thinking about writing tests for ML projects. I'd
like to add a couple here for good measure but its definitely seems to be a different
set of challenges than I'd write for a library. Seems like the techniques are
- Check the shapes of outputs
- Check against some small dataset for consistency
- Harder to check that things are working they way they should i.e. testing your
  understanding. But a few of these would be good things like, for the causal model, the
predictions upstream of a change in the input should not change.

## Practicalities

### Data Preparation

### Usage

## References

1. Attention is All You Need
1. GPT-2
1. Llama2
1. Nucleus Sampling
1. [Multi-Query Attention] (https://arxiv.org/pdf/1911.02150)
