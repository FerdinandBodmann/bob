"""A complete GPT-style decoder-only transformer.

Everything a "full" language model needs — embeddings, causal self-attention,
MLP blocks, weight tying, proper initialization, checkpointing — except
training. Weights are random until someone runs train.py on them.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256      # char-level default; overwrite for a real tokenizer
    block_size: int = 256      # max context length (positions)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False         # LayerNorm/Linear bias, GPT-2 style off-by-default


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with KV caching for generation."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        # query, key, value projections fused into one matrix for efficiency
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # is_causal is only valid when query and key lengths match; with a KV
        # cache the query is the last position, so an explicit mask is needed.
        if past_kv is None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            S = k.size(2)
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), new_kv


class MLP(nn.Module):
    """Position-wise feed-forward block (GPT-2 formulation: 4x expansion)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        a, kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),   # token embeddings
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),   # position embeddings
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds context {self.cfg.block_size}"

        if past_kvs is None:
            start = 0
            pos = torch.arange(T, dtype=torch.long, device=idx.device)
        else:
            start = past_kvs[0][0].size(2)
            pos = torch.arange(start, start + T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_kvs = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, past_kv=past, use_cache=use_cache)
            if use_cache:
                new_kvs.append(kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference: only the last position's logits are needed
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, seed=None):
        """Sample tokens autoregressively. Supports temperature and top-k.

        When the context window fills up, the KV cache is dropped and the last
        `block_size` tokens are re-encoded (sliding window) instead of crashing.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        block = self.cfg.block_size
        idx = idx[:, -block:]  # never feed more than the model's context
        past_kvs = None
        for _ in range(max_new_tokens):
            if past_kvs is not None and past_kvs[0][0].size(2) >= block:
                past_kvs = None          # context full: drop cache, re-encode window
                idx = idx[:, -block:]
            idx_cond = idx if past_kvs is None else idx[:, -1:]
            logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs, use_cache=True)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """AdamW with decay applied only to 2D+ params (not biases/LN)."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)

    def save_checkpoint(self, path, **extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "config": self.cfg, **extra}, path)

    @classmethod
    def load_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        return model.to(device), ckpt
