"""Compact GPT with pluggable auxiliary heads.

Approaches plug in here:
  * AUX_HEADS: a name -> (n_classes) table; each head is a Linear on the final hidden state that
    predicts a property of the NEXT token (train-time only, dropped at inference). Add a new
    structural signal by adding a tag stream in prepare.py and one line in AUX_HEADS.
  * Distillation lives in train.py (teacher logits), not here.
  * Swap the output layer (class-factored softmax, adaptive softmax...) by replacing `self.head`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_HEADS = {"upos": 18, "chunk": 18}   # sizes match prepare.UPOS_TAGS / CHUNK_TAGS


class Block(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.h, self.drop = h, dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q = q.view(B, T, self.h, D // self.h).transpose(1, 2)
        k = k.view(B, T, self.h, D // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, mcfg):
        super().__init__()
        d, self.ctx = mcfg["d_model"], mcfg["ctx"]
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(self.ctx, d)
        self.drop = nn.Dropout(mcfg.get("dropout", 0.0))
        self.blocks = nn.ModuleList([Block(d, mcfg["n_head"], mcfg.get("dropout", 0.0)) for _ in range(mcfg["n_layer"])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok.weight                       # weight tying
        self.aux = nn.ModuleDict({name: nn.Linear(d, AUX_HEADS[name]) for name in mcfg.get("aux_heads", [])})
        self.apply(self._init)
        for n, p in self.named_parameters():                     # GPT-2 style scaled init on residual projections
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok.weight.numel() if non_embedding else n

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        x = self.ln(x)
        return self.head(x), x                                    # logits, hidden (for aux heads)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40, top_p=0.95,
                 repetition_penalty=1.2, no_repeat_ngram=3, stop_ids=()):
        """Sampling with a repetition penalty on every token already in the window and a hard ban on repeating
        any n-gram already present (no_repeat_ngram=3 stops the "Tokyo, Japan, Japan, Japan" loops; 0 disables)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.ctx:])
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    logits[b, seen] = torch.where(logits[b, seen] > 0, logits[b, seen] / repetition_penalty,
                                                  logits[b, seen] * repetition_penalty)
            if no_repeat_ngram and idx.size(1) >= no_repeat_ngram:
                n = no_repeat_ngram
                for b in range(idx.size(0)):
                    seq = idx[b].tolist()
                    prefix = tuple(seq[-(n - 1):])
                    banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n - 1]) == prefix]
                    if banned:
                        logits[b, banned] = float("-inf")
            logits = logits / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, order = torch.sort(logits, descending=True)
                cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                mask = cum - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_ids and idx.size(0) == 1 and nxt.item() in stop_ids:
                break
        return idx
