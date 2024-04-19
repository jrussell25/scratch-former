import torch
from .model import Transformer


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
