#!/usr/bin/env python3
import os, struct, hmac, hashlib, math, sys, argparse
from typing import List, Literal, Tuple

import numpy as np
import torch
from transformers import GPT2TokenizerFast

MODEL_ID = "openai-community/gpt2"
WATERMARK_KEY = b"MyReallySecretKeyForTournament!!"
G_DIST: Literal["uniform","bernoulli"] = "bernoulli"
H_WINDOW = 4
NSEC = 128
GROUP_SIZE = 2
M_LAYERS = 6
ALPHA = 1e-3

def derive_seed_sliding_window(prev_tokens: List[int], key: bytes, H: int = 4, nsec: int = 128) -> int:
    window = prev_tokens[-H:] if len(prev_tokens) >= H else prev_tokens
    msg = b"\x00SEED|" + struct.pack("<I", len(window))
    for token_id in window:
        msg += struct.pack("<I", int(token_id) & 0xFFFFFFFF)
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") & ((1 << nsec) - 1)

def g_value_from_seed_detector(token_id: int, layer: int, seed_rt: int, *, nsec: int = 128,
                               dist: Literal["bernoulli","uniform"] = "bernoulli") -> float:
    msg = bytearray(b"\x00GVAL|")
    msg += struct.pack("<I", int(token_id) & 0xFFFFFFFF)
    msg += struct.pack("<I", int(layer) & 0xFFFFFFFF)
    sb = seed_rt.to_bytes((nsec + 7) // 8, "big")
    msg += struct.pack("<I", len(sb)) + sb
    h = hashlib.sha256(bytes(msg)).digest()
    u_bits = int.from_bytes(h, "big") & ((1 << nsec) - 1)
    u = u_bits / float(1 << nsec)
    if dist == "bernoulli":
        return 1.0 if u < 0.5 else 0.0
    else:
        return float(u)

def detect_watermark_score(token_ids: List[int], key: bytes, *, m: int, H: int, nsec: int,
                           dist: Literal["bernoulli", "uniform"]) -> Tuple[float, float, bool]:
    T = len(token_ids)
    if T == 0:
        raise ValueError("No tokens for detection.")
    sum_g = 0.0
    for t in range(1, T + 1):
        prev = token_ids[:t-1]
        r_t = derive_seed_sliding_window(prev, key, H=H, nsec=nsec)
        x_t = token_ids[t-1]
        for ell in range(1, m + 1):
            sum_g += g_value_from_seed_detector(x_t, ell, r_t, nsec=nsec, dist=dist)

    Score = sum_g / (T * m)
    mu0 = 0.5  # expected mean for random (non-watermarked) text
    label = "WATERMARKED" if Score > mu0 else "NOT_WATERMARKED"
    return Score, mu0, label

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect watermark from a text file (simplified)")
    p.add_argument("--in", dest="infile", type=str, required=True,
                   help="Path to text file (prompt+generation)")
    return p

def main():
    args = build_arg_parser().parse_args()
    with open(args.infile, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    token_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    Score, mu0, label = detect_watermark_score(
        token_ids=token_ids, key=WATERMARK_KEY,
        m=M_LAYERS, H=H_WINDOW, nsec=NSEC, dist=G_DIST
    )
    print("\n--- Watermark Detection Report ---", flush=True)
    print(f"Tokens: {len(token_ids)}", flush=True)
    print(f"Score: {Score:.6f}  (μ₀={mu0:.3f})", flush=True)
    print(f"Decision: {label}\n", flush=True)

if __name__ == "__main__":
    main()
