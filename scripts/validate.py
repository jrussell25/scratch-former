import sys
import glob
import re
from pathlib import Path

import numpy as np
import torch
from scratch_former.model import Transformer

_, checkpoint_dir, val_tokens_path = sys.argv

checkpoint_path = Path(checkpoint_dir)
checkpoint_files = glob.glob(str(checkpoint_dir) + "*pt")
checkpoint_nums = torch.tensor(
    [int(re.findall("(\d+)", Path(x).name)[0]) for x in checkpoint_files]
)
checkpoint_inds = torch.argsort(checkpoint_nums)

assert len(checkpoint_files) > 0, "No checkpoint files found"

val_tokens = np.load(str(Path(val_tokens_path).expanduser()))

batch_size = 64
seq_length = 257  # 256 token input, compute probability of next token
n_batches = val_tokens.shape[0] // (batch_size * seq_length)
batched = torch.tensor(
    val_tokens[: batch_size * n_batches * seq_length].reshape(
        n_batches, batch_size, seq_length
    )
)


model = Transformer(n_blocks=4, d_model=512, d_ff=1024, n_heads=8)
ctx_inds = torch.triu_indices(seq_length - 1, seq_length - 1, 1)

for i in checkpoint_inds:
    model.load_state_dict(torch.load(checkpoint_files[i.item()], map_location="cpu"))

    total_log_prob = 0
    for batch in batched:
        next_tokens = batch[:, -1]
        with torch.no_grad():
            outputs = model(batch[:, :-1], mask_inds=ctx_inds)

        log_p = torch.log_softmax(outputs[:, -1], dim=-1)  # batch_size x vocab_size

        log_p_next = log_p[torch.arange(batch_size), next_tokens]  # batch_size

        total_log_prob += log_p_next.sum()

    perplexity = torch.exp(-1 * total_log_prob / (n_batches * batch_size)).item()
    print(f"{checkpoint_nums[i].item()} iterations -- PPL={perplexity:0.3f}")
