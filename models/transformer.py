import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Decoder-only Model, so using mask
    - sequence length variavle <= block_size
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model // n_heads should be zero"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # register the mask for further masking
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        # broadcast its shape to (1, 1, T, T) to further match (B, H, T, T)
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, C)
        key_padding_mask: (B, T); True for valid token, False for padding
        return:
          out: (B, T, C)
        """
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        # cal Q, K, V
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        # Transfer to Multi-Head
        q = q.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # Scaled dot-product attention
        # scores: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(D))

        # 1) causal mask
        causal = self.causal_mask[..., :T, :T]  # (1,1,T,T)
        scores = scores.masked_fill(~causal, float("-inf"))

        # 2) padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True=valid, False=pad
            # extended to shape (B, 1, 1, T)
            kpm = key_padding_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(~kpm, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # 合并头 (B, T, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    """FeedForward Network: Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Block: Pre-LN + Self-Attention + Residual + Pre-LN + FFN + Residual
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x