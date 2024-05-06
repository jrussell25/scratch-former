# Generate some text from a seed sequence using all the different inference stratgeies
# python generate.py path/to/model/weights.pt path/to/test_tokens.npy device
import sys
from time import perf_counter
import torch
import numpy as np
import tiktoken

from scratch_former.model import Transformer
from scratch_former.sample import (
    predict_beam,
    predict_greedy,
    sample_topk,
    sample_nucleus,
)


weight_path = sys.argv[1]
test_tokens_path = sys.argv[2]
device = sys.argv[3]

if device not in ["cpu", "cuda"]:
    raise ValueError(f"device must be 'cpu' or 'cuda' but found {device}")

if device == "cuda" and not torch.cuda.is_available():
    raise ValueError(
        "CUDA not available. Use 'cpu' instead or move" "to computer with CUDA"
    )

model = Transformer(n_blocks=4, d_model=512, d_ff=1024, n_heads=8)
model.load_state_dict(torch.load(weight_path, map_location=device))

bpe = tiktoken.get_encoding("cl100k_base")
eot = bpe.encode_single_token("<|endoftext|>")

# SAMPLING PARAMS
N_GEN = 64
N_BATCH = 3
CTX_LENGTH = 256

BEAM = 10
K = 128
T_LOW = 0.7
P_NUC = 0.95

# Get data select seed sequence
test_tokens = np.load(test_tokens_path)
article_starts = np.concatenate(([0], np.nonzero(test_tokens == eot)[0] + 1))
# seed_idx = np.random.default_rng().choice(article_starts)
seed_idx = article_starts[35]  # lets just talk about brad
seed = torch.tensor(test_tokens[seed_idx : seed_idx + CTX_LENGTH])[None]
seed_string = bpe.decode(seed[0].cpu().numpy())

ctx_inds = torch.triu_indices(CTX_LENGTH, CTX_LENGTH, 1)

# Start the generation process
t0 = perf_counter()
greedy_tokens = predict_greedy(model, seed, ctx_inds, N_GEN)
greedy_string = bpe.decode_batch(greedy_tokens.cpu().numpy())[0]
t1 = perf_counter()

print(
    f"Greedy Sampling -- 1 seq x {N_GEN} tokens -- {N_GEN/(t1-t0):0.3f} tokens/second"
)
print(f"Seed: {seed_string}")
print(f"Response: {greedy_string}")
print()


t0 = perf_counter()
beam_tokens: torch.Tensor = predict_beam(model, BEAM, N_GEN, seed)
beam_string = bpe.decode(beam_tokens.cpu().numpy())
t1 = perf_counter()

print(
    f"Beam Sampling ({BEAM=})-- 1 seq x {N_GEN} tokens -- {N_GEN/(t1-t0):0.3f} tokens/second"
)
print(f"Seed: {seed_string}")
print(f"Response: {beam_string}")
print()


t0 = perf_counter()
topk_tokens = sample_topk(model, seed, N_GEN, K, batch_size=N_BATCH)
topk_string = bpe.decode_batch(topk_tokens.cpu().numpy())
t1 = perf_counter()

print(
    f"Top K Sampling ({K=})-- {N_BATCH} seq x {N_GEN} tokens -- {N_BATCH*N_GEN/(t1-t0):0.3f} tokens/second"
)
print(f"Seed: {seed_string}")
for i, s in enumerate(topk_string):
    print(f"Response {i}: {s}")
print()

t0 = perf_counter()
topktemp_tokens = sample_topk(
    model, seed, N_GEN, K, temperature=T_LOW, batch_size=N_BATCH
)
topktemp_string = bpe.decode_batch(topktemp_tokens.cpu().numpy())
t1 = perf_counter()

print(
    f"Top K Sampling ({K=} {T_LOW=})-- {N_BATCH} seq x {N_GEN} tokens -- {N_BATCH*N_GEN/(t1-t0):0.3f} tokens/second"
)
print(f"Seed: {seed_string}")
for i, s in enumerate(topktemp_string):
    print(f"Response {i}: {s}")
print()


t0 = perf_counter()
nucleus_tokens = sample_nucleus(model, seed, N_GEN, P_NUC, batch_size=N_BATCH)
nucleus_string = bpe.decode_batch(nucleus_tokens.cpu().numpy())
t1 = perf_counter()

print(
    f"Nucleus Sampling ({P_NUC=}) -- {N_BATCH} seq x {N_GEN} tokens -- {N_BATCH*N_GEN/(t1-t0):0.3f} tokens/second"
)
print(f"Seed: {seed_string}")
for i, s in enumerate(nucleus_string):
    print(f"Response {i}: {s}")
