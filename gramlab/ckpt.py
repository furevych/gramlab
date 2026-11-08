"""Checkpoint helpers shared by train / finetune / evaluate / chat."""
import os
import torch
from .model import GPT
from .tokenizer import load_tokenizer


def save_checkpoint(path, model, optimizer, step, cfg, tok_dir, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
             "step": step, "config": cfg, "tokenizer_dir": tok_dir, "vocab_size": model.tok.weight.shape[0]}
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = GPT(state["vocab_size"], cfg["model"]).to(device)
    model.load_state_dict(state["model"])
    tok = load_tokenizer(state["tokenizer_dir"])
    return model, tok, state
