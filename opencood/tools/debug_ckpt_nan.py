# -*- coding: utf-8 -*-
"""Check a training checkpoint for NaN/Inf weights, per module."""
import sys
import torch

ckpt_path = sys.argv[1]
sd = torch.load(ckpt_path, map_location="cpu")
sd = sd.get("model_state_dict", sd)

nan_keys, inf_keys = [], []
for k, v in sd.items():
    if v.is_floating_point():
        n_nan = torch.isnan(v).sum().item()
        n_inf = torch.isinf(v).sum().item()
        if n_nan:
            nan_keys.append((k, n_nan, v.numel()))
        if n_inf:
            inf_keys.append((k, n_inf, v.numel()))

print(f"total params: {len(sd)} | NaN tensors: {len(nan_keys)} | Inf tensors: {len(inf_keys)}")
for k, n, t in nan_keys[:20]:
    print(f"  NaN {k}: {n}/{t}")
for k, n, t in inf_keys[:20]:
    print(f"  INF {k}: {n}/{t}")
if not nan_keys and not inf_keys:
    print("checkpoint weights are all finite -> NaN comes from forward/loss numerics, not weights")
