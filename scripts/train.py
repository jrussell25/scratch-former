import torch
import numpy as np

from time import perf_counter
from scratch_former.model import Transformer
from scratch_former.data import get_batch_tokens

last_checkpoint = 100000
d_model = 512
warmup = 4000

model = Transformer(n_blocks=4, d_model=d_model, d_ff=1024, n_heads=8)
if last_checkpoint is not None:
    model.load_state_dict(torch.load(f"checkpoint_{last_checkpoint}.pt"))
    start = last_checkpoint
else:
    start = 0

model = model.cuda()

train_arr = np.load("train_tokens.npy")

celoss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
opt = torch.optim.Adam(model.parameters(), lr=1.0)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda step: (d_model**-0.5) * min(1.0 / np.sqrt(step), step / warmup**1.5)
)

if last_checkpoint is not None:
    sched.last_epoch = last_checkpoint - 1
    sched.step()

n_iters = int(2e5)
log_every = 500
chkpt_every = 10000
context_length = 256
ctx_inds = torch.triu_indices(context_length, context_length, 1).cuda()

losses = []
t0 = perf_counter()
ttotal = t0
total_loss = 0
for i in range(start, n_iters):
    batch_src, batch_tgt = get_batch_tokens(
        train_arr, 10, context_length, device="cuda"
    )
    z = model(batch_src, mask_inds=ctx_inds)
    z = z.reshape(-1, z.shape[-1])
    loss = celoss(z, batch_tgt.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    total_loss += loss.clone().detach().cpu().item()
    if (i + 1) % log_every == 0:
        loss_avg = total_loss / log_every
        losses.append(loss_avg)
        print(
            f"Iterations: {i+1} -- Loss = {loss_avg:0.4f} -- LR = {sched.get_last_lr()[0]:0.4g} -- {perf_counter()-t0:0.3f} seconds/{log_every} iterations"
        )
        t0 = perf_counter()
        total_loss = 0
    if (i + 1) % chkpt_every == 0:
        torch.save(model.state_dict(), f"checkpoint_{i+1}.pt")
