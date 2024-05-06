import torch
from scratch_former.model import Transformer


def predict_greedy(
    model: Transformer,
    seed_tokens: torch.Tensor,
    mask_inds: torch.Tensor,
    n_pred: int = 256,
) -> torch.Tensor:
    greedy_preds = torch.zeros((seed_tokens.shape[0], n_pred), dtype=torch.long)

    x = seed_tokens
    for i in range(n_pred):
        pred = model(x, mask_inds)
        next_token = pred[:, -1].argmax(-1)
        greedy_preds[:, i] = next_token
        x = torch.cat([x[:, 1:], next_token[None]], dim=1)
    return greedy_preds


def predict_beam(
    model: Transformer, b: int, n_gen: int, seed_seq: int, return_beam: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    vocab_size = model.vocab_size
    seq = torch.atleast_2d(seed_seq)
    ctx_mask = torch.triu_indices(seq.shape[1], seq.shape[1], 1)
    total_log_prob = torch.zeros((b))
    for i in range(n_gen):
        with torch.no_grad():
            x = model(seq, mask_inds=ctx_mask)
        log_p = torch.log_softmax(x[:, -1], dim=-1)
        top = log_p.ravel().topk(b)
        row = top.indices // vocab_size
        col = top.indices % vocab_size
        total_log_prob = total_log_prob[row] + top.values
        # print(row, col, total_log_prob)
        seq = torch.cat((seq[row, 1:], col[:, None]), dim=1)
    if return_beam:
        return seq[:, -n_gen:], total_log_prob
    else:
        return seq[total_log_prob.argmax(), -n_gen:]


def sample_topk(
    model: Transformer,
    seed_seq: int,
    n_gen: int,
    k: int,
    temperature: float = 1.0,
    batch_size: int = 1,
) -> torch.Tensor:
    seq = torch.atleast_2d(seed_seq)
    seq = seq.expand((batch_size, seq.shape[1]))
    ctx_mask = torch.triu_indices(seq.shape[1], seq.shape[1], 1)
    for i in range(n_gen):
        with torch.no_grad():
            x = model(seq, mask_inds=ctx_mask)
        next_token = torch.zeros((batch_size), dtype=torch.int)
        top = torch.topk(x[:, -1], k)
        probs = torch.softmax(top.values / temperature, dim=-1)
        next_token = top.indices[
            torch.arange(batch_size), torch.multinomial(probs, num_samples=1).squeeze()
        ]
        seq = torch.cat((seq[:, 1:], next_token[:, None]), dim=1)
    return seq[:, -n_gen:]


def sample_nucleus(
    model: Transformer,
    seed_seq: torch.Tensor,
    n_pred: int,
    p: float = 0.95,
    batch_size: int = 1,
) -> torch.Tensor:
    seq = torch.atleast_2d(seed_seq)
    seq = seq.expand((batch_size, seq.shape[1]))
    ctx_mask = torch.triu_indices(seq.shape[1], seq.shape[1], 1)
    for i in range(n_pred):
        with torch.no_grad():
            x = model(seq, mask_inds=ctx_mask)
        probs = torch.softmax(x[:, -1], dim=-1)
        p_sorted, inds = torch.sort(probs, descending=True)
        keep = torch.cumsum(p_sorted, dim=-1) < p
        next_token = torch.zeros((batch_size), dtype=torch.int)
        for b in range(
            batch_size
        ):  # has to be a loop because of variable sizes for each batch
            probs_nuc = p_sorted[b, keep[b]]
            probs_nuc /= probs_nuc.sum()
            if probs_nuc.shape[0] == 0:
                next_token[b] = inds[b][0]
            else:
                next_token[b] = inds[b][torch.multinomial(probs_nuc, num_samples=1)]
        seq = torch.cat((seq[:, 1:], next_token[:, None]), dim=1)
    return seq[:, -n_pred:]
