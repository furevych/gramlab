"""Mixed-precision choice shared by train.py and finetune.py.

Ampere+ GPUs (A100, L4, RTX 30xx...) -> bf16 autocast, no loss scaling needed.
T4 / V100 / P100                     -> fp16 autocast + dynamic loss scaling (bf16 is only emulated there and is slow).
CPU                                  -> fp32.
"""
import torch


def pick_precision(device: str):
    """Returns (amp_dtype, use_amp, use_fp16, scaler, name)."""
    use_bf16 = False
    if device == "cuda":
        try:                                                    # real bf16 hardware only, not emulation
            use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:                                       # older torch: no such kwarg
            use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):                         # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    name = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    return amp_dtype, (use_bf16 or use_fp16), use_fp16, scaler, name
