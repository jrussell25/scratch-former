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
  *NB* As of this writing it seems like the original links are broken. I grabbed the
text from huggingface.
- Randomly sampled training sequences of 256 tokens with causal masking and predicting the next token at each position. I was working with pretty limited gpu resources so I used batch size of 10 sequences.
- I trained for 200k steps following the learning rate schedule described in Vaswami
  2017 with 4000 warmup steps.
- Saved model checkpoint every 10k steps.

#### Results

After training, this model achieves a sliding-window perplexity PPL of 35.XX (CHECK!).
This is comparable to quoted values for models like `gpt2-small` which has about 50%
more paramters than my model and was of course trained on more data with more compute.

(MAKE A PLOT)


### Inference Strategies

I got particularly interested in the inference process for this and similar models. I
started where most do with greedy search: predicting un-normalized logits for the next
token in the sequence and then selecting the highest probability token.

** Greedy Samples **

** Beam Search Samples **

The beam search results are somewhat interesting because you can see where the model
knows patterns but not facts. For example it often knows that a sentence needs to refer
to a year but has no idea what year things happen in.

Another interesting phenomenon was the way these models get caught in "loops," highly
repetitive sequences. It turns out this is a well documented phenomenon. Apparently pure
sampling approaches help with repetition though tend to devolve quickly into nonsense.
I'm most interested to try *nucleus sampling* where one computes the logits, keep only
the top ~95%, and randomly sample from those.

### Other things I learned along the way

- BPE Tokenization -- a very clever solution and a major cut corner. When I was first
  learning about language modeling (ca 2017) word embeddings like GLoVE were still
quite popular and that was actually where I started for this model.
- Cloud computing -- I was surprised by how tricky it was to get high-end GPUs (e.g. V100 let lone A100) on AWS. In retrospect it is pretty reasonable to prevent new users to get on these expensive GPUs when Amazon doesnt trust them to pay. I ended up using paperspace.
- Value of better GPUs -- I did some very preliminary tests of training throughput on
  different hardware. It my tests it was worth paying up for better machines in the
sense of reducing the cost per training iteration.

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
1. [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
1. [Nucleus Sampling](https://arxiv.org/abs/1904.09751)
1. [Multi-Query Attention](https://arxiv.org/pdf/1911.02150)
1. [Microsoft Guide to ML
   Testing](https://microsoft.github.io/code-with-engineering-playbook/machine-learning/ml-testing/)
