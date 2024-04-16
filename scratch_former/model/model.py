from typing import Optional
import torch
from scratch_former.layers import MHSA, PositionalEncoding, ResidualFFN


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.mhsa = MHSA(self.d_model, self.n_heads)
        self.ffn = ResidualFFN(self.d_model, self.d_ff)
        self.mhsa_norm = torch.nn.LayerNorm(self.d_model)
        self.ffn_norm = torch.nn.LayerNorm(self.d_model)

    def forward(
        self, x: torch.Tensor, mask_inds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        z = self.mhsa(x, mask_inds=mask_inds)
        z = self.mhsa_norm(z)
        z = self.ffn(z)
        z = self.ffn_norm(z)
        return z


class Transformer(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        vocab_size: int = 100277,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.vocab_size = vocab_size

        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(self.d_model)
        self.layer_norm = torch.nn.LayerNorm(self.d_model)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(self.d_model, self.d_ff, self.n_heads)
                for _ in range(self.n_blocks)
            ]
        )

        self.linear_out = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, mask_inds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        z = self.embedding(x)
        z = self.pe(z)
        z = self.layer_norm(z)
        for block in self.transformer_blocks:
            z = block(z, mask_inds=mask_inds)

        z = self.linear_out(z)

        return z
