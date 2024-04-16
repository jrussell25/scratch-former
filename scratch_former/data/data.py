import numpy as np
import torch


def get_batch_tokens(
    token_arr: np.ndarray, batch_size: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(len(token_arr) - batch_size - 1, size=(batch_size,))
    tokens = torch.zeros((batch_size, context_length + 1), dtype=torch.long)
    for i, idx in enumerate(starts):
        tokens[i] = torch.tensor(
            token_arr[idx : idx + context_length + 1], dtype=torch.float
        )
    return tokens[:, :-1], tokens[:, 1:]
