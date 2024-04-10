import torch

__all__ = [
    "ResidualFF",
    "MHSA",
]


class ResidualFF(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear1(x)
        z = self.relu(x)
        z = self.linear2(x)
        return x + z


class MHSA(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be evenly divisble by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        self.Wq = torch.nn.Parameter(d_model, d_model)
        self.Wk = torch.nn.Parameter(d_model, d_model)
        self.Wv = torch.nn.Parameter(d_model, d_model)
        self.Wout = torch.nn.Parameter(d_model, d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:  # x should be N x L x d_model
        # each is N x L x H x A
        # think about the translation or non-self attention case
        # Lq does not necessarily need to be the same a Lk
        q = torch.matmul(x, self.Wq).reshape((x.shape[:2], self.n_heads, self.d_head))
        k = torch.matmul(x, self.Wk).reshape((x.shape[:2], self.n_heads, self.d_head))
        v = torch.matmul(x, self.Wv).reshape((x.shape[:2], self.n_heads, self.d_head))

        # Attn map is N x L x H
        attn = torch.einsum("nqhd,nkhd->nqkh", q, k)
        attn = torch.softmax(attn * self.scale, dim=-2)

        z = torch.einsum("nqkh,nkhd->nqhd", attn, v).reshape(
            (*x.shape[:2], self.d_model)
        )
        z = torch.einsum("nqd,de->nqe", z, self.Wout)

        return x + z
