import torch


class ResidualFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear1(x)
        z = self.relu(z)
        z = self.linear2(z)
        return x + z


class MHSA(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be evenly divisble by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32))

        self.proj_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.proj_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.proj_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.proj_out = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask_inds: torch.Tensor | None
    ) -> torch.Tensor:  # x should be N x L x d_model
        # each is N x L x H x A
        # think about the translation or non-self attention case
        # Lq does not necessarily need to be the same a Lk
        q = self.proj_q(x).reshape((*x.shape[:2], self.n_heads, self.d_head))
        k = self.proj_k(x).reshape((*x.shape[:2], self.n_heads, self.d_head))
        v = self.proj_v(x).reshape((*x.shape[:2], self.n_heads, self.d_head))

        # Attn map is N x Lq x Lk x H
        attn = torch.einsum("nqhd,nkhd->nqkh", q, k)
        if mask_inds is not None:
            attn[:, mask_inds[0], mask_inds[1]] = -1e9
        attn = torch.softmax(attn * self.scale, dim=-2)

        z = torch.einsum("nqkh,nkhd->nqhd", attn, v).reshape(
            (*x.shape[:2], self.d_model)
        )
        z = self.proj_out(z)
        return x + z


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, Lmax: int = 1000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        pos = torch.arange(Lmax).unsqueeze(1)
        log_w0 = torch.log(torch.tensor(1e-5))
        w = torch.exp(torch.arange(0, d_model, 2) * log_w0)

        self.pe = torch.nn.Parameter(
            data=torch.zeros((1, Lmax, d_model)), requires_grad=False
        )
        self.pe.data[..., ::2] = torch.sin(w * pos)
        self.pe.data[..., 1::2] = torch.cos(w * pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]
