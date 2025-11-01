#!/usr/bin/env python3
# wm_generate.py (with tournament JSON trace)
import os, json, struct, hmac, hashlib, math, sys, argparse
from typing import Optional, Sequence, List, Literal, Dict, Any

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ===========================
# CONFIG (safe defaults)
# ===========================
MODEL_ID = "openai-community/gpt2"
OUTPUT_TXT = "output.txt"

MAX_NEW_TOKENS = 60
TEMPERATURE = 1.0
TOP_K: Optional[int] = None
TOP_P: Optional[float] = None
SEED = 1234

# ---- Watermark & tournament params (KEEP THESE IN SYNC WITH DETECTOR) ----
WATERMARK_KEY = b"MyReallySecretKeyForTournament!!"  # change to your secret
G_DIST: Literal["uniform","bernoulli"] = "bernoulli"
H_WINDOW = 4
NSEC = 128

GROUP_SIZE = 2
M_LAYERS = 6                    # <= small; detector uses the SAME value
MAX_CANDIDATES = 64             # = 2**M_LAYERS, safe for laptops

# ---- Tournament trace (JSON) ----
TRACE_TOURNAMENT = True         # set False to disable JSON trace
TRACE_TOP_TOKENS = 15           # top-N probs recorded each step
TRACE_DIR = "tournament_traces" # folder for step_XXX_tournament.json

# ===========================
# Helpers
# ===========================
def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def apply_filters(logits: torch.Tensor, temperature: float,
                  top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        mask = torch.full_like(logits, float("-inf")); mask.scatter_(1, topk_idx, topk_vals); logits = mask
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        keep = cumprobs <= top_p; keep[:, 0] = True
        filtered = torch.full_like(sorted_logits, float("-inf")); filtered[keep] = sorted_logits[keep]
        inv = torch.empty_like(sorted_idx)
        inv.scatter_(1, sorted_idx, torch.arange(sorted_idx.size(1), device=logits.device).unsqueeze(0))
        logits = filtered.gather(1, inv)
    return logits

def derive_seed_sliding_window(prev_tokens: List[int], key: bytes, H: int = 4, nsec: int = 128) -> int:
    window = prev_tokens[-H:] if len(prev_tokens) >= H else prev_tokens
    msg = b"\x00SEED|" + struct.pack("<I", len(window))
    for token_id in window:
        msg += struct.pack("<I", int(token_id) & 0xFFFFFFFF)
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") & ((1 << nsec) - 1)

def g_value_from_seed(token_id: int, layer: int, seed_rt: int, nsec: int = 128,
                      dist: Literal["uniform","bernoulli"] = "bernoulli") -> float | int:
    msg = bytearray(b"\x00GVAL|")
    msg += struct.pack("<I", int(token_id) & 0xFFFFFFFF)
    msg += struct.pack("<I", int(layer) & 0xFFFFFFFF)
    sb = seed_rt.to_bytes((nsec + 7) // 8, "big")
    msg += struct.pack("<I", len(sb)) + sb
    h = hashlib.sha256(bytes(msg)).digest()
    u_bits = int.from_bytes(h, "big") & ((1 << nsec) - 1)
    u = u_bits / float(1 << nsec)
    if dist == "uniform": return u
    if dist == "bernoulli": return 1 if u < 0.5 else 0
    raise ValueError("dist must be 'uniform' or 'bernoulli'")

@torch.inference_mode()
def tournament_sample(
    probs: torch.Tensor,
    prev_tokens: Sequence[int],
    key: bytes,
    *,
    m: int,
    group_size: int,
    nsec: int,
    g_dist: Literal["uniform","bernoulli"],
    H: int,
    generator: Optional[torch.Generator] = None,
    tokenizer: Optional[GPT2TokenizerFast] = None,
    trace_dict: Optional[Dict[str, Any]] = None,
) -> int:
    """Runs a capped multi-layer tournament and optionally fills trace_dict for JSON."""
    assert probs.dim() == 1 and group_size >= 2 and m >= 1
    r_t = derive_seed_sliding_window(list(prev_tokens), key=key, H=H, nsec=nsec)

    # Initial candidates (capped)
    N0 = min(int(group_size ** m), int(MAX_CANDIDATES))
    candidates = torch.multinomial(probs, num_samples=N0, replacement=True, generator=generator)
    current = candidates.tolist()

    # Trace header
    if trace_dict is not None:
        trace_dict["seed_rt"] = int(r_t)
        init = []
        for cid in current:
            item = {"id": int(cid), "p": float(probs[cid].item())}
            if tokenizer is not None:
                item["token"] = tokenizer.convert_ids_to_tokens([cid])[0]
            init.append(item)
        trace_dict["initial_candidates"] = init
        trace_dict["layers"] = []

    # Layers
    for layer in range(1, m + 1):
        if len(current) == 1:
            break
        perm = torch.randperm(len(current), generator=generator).tolist()
        shuffled = [current[i] for i in perm]
        winners: List[int] = []

        layer_rec: Dict[str, Any] = {"layer": layer, "groups": []} if trace_dict is not None else None

        for i in range(0, len(shuffled), group_size):
            group = shuffled[i:i + group_size]
            if len(group) < group_size:
                group.extend(torch.multinomial(probs, num_samples=(group_size - len(group)),
                                               replacement=True, generator=generator).tolist())

            scores = [g_value_from_seed(tok, layer, r_t, nsec=nsec, dist=g_dist) for tok in group]
            max_score = max(scores)
            tied_idxs = [j for j, sc in enumerate(scores) if sc == max_score]
            if len(tied_idxs) == 1:
                win_tok = group[tied_idxs[0]]
                tie_flag = False
            else:
                idx_local = torch.multinomial(torch.ones(len(tied_idxs)), 1, generator=generator).item()
                win_tok = group[tied_idxs[idx_local]]
                tie_flag = True

            winners.append(win_tok)

            if layer_rec is not None:
                group_rec = {"candidates": [], "winner": {}, "tie": tie_flag}
                for tok, sc in zip(group, scores):
                    entry = {"id": int(tok), "g": float(sc)}
                    if tokenizer is not None:
                        entry["token"] = tokenizer.convert_ids_to_tokens([tok])[0]
                    group_rec["candidates"].append(entry)
                win_entry = {"id": int(win_tok)}
                if tokenizer is not None:
                    win_entry["token"] = tokenizer.convert_ids_to_tokens([win_tok])[0]
                group_rec["winner"] = win_entry
                layer_rec["groups"].append(group_rec)

        current = winners
        if layer_rec is not None:
            trace_dict["layers"].append(layer_rec)

    final_winner = int(current[0])
    if trace_dict is not None and tokenizer is not None:
        trace_dict["final_winner"] = {
            "id": final_winner,
            "token": tokenizer.convert_ids_to_tokens([final_winner])[0]
        }
    return final_winner

@torch.inference_mode()
def generate_text(prompt: str, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, device: str,
                  key: bytes, max_new_tokens: int, temperature: float,
                  top_k: Optional[int], top_p: Optional[float], seed: Optional[int]) -> str:
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    os.makedirs(TRACE_DIR, exist_ok=True) if TRACE_TOURNAMENT else None

    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prev_tokens = prompt_ids[0].tolist()
    input_ids = prompt_ids
    past_key_values = None
    generated_ids: List[int] = []

    for step in range(1, max_new_tokens + 1):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        probs = torch.softmax(apply_filters(logits, temperature, top_k, top_p), dim=-1).squeeze(0)
        g = torch.Generator(device=probs.device).manual_seed((seed or 0) + step) if seed is not None else None

        # Trace container for this step
        trace: Dict[str, Any] = {} if TRACE_TOURNAMENT else None

        next_token_id = tournament_sample(
            probs=probs, prev_tokens=prev_tokens, key=key,
            m=M_LAYERS, group_size=GROUP_SIZE, nsec=NSEC, g_dist=G_DIST, H=H_WINDOW,
            generator=g, tokenizer=tokenizer, trace_dict=trace
        )

        # Augment trace with top probs and selected token
        if TRACE_TOURNAMENT:
            topk = min(TRACE_TOP_TOKENS, probs.numel())
            vals, idx = torch.topk(probs, topk)
            trace["top_probs"] = [
                {"rank": i+1,
                 "id": int(idx[i]),
                 "token": tokenizer.convert_ids_to_tokens([int(idx[i])])[0],
                 "p": float(vals[i])}
                for i in range(topk)
            ]
            trace["selected"] = {
                "id": int(next_token_id),
                "token": tokenizer.convert_ids_to_tokens([int(next_token_id)])[0],
                "p": float(probs[int(next_token_id)])
            }
            trace["step"] = step
            trace["m_layers"] = M_LAYERS
            trace["group_size"] = GROUP_SIZE
            trace["g_dist"] = G_DIST
            trace["H_window"] = H_WINDOW
            trace["nsec"] = NSEC
            trace["temperature"] = temperature
            trace["top_k"] = top_k
            trace["top_p"] = top_p

            with open(os.path.join(TRACE_DIR, f"step_{step:03d}_tournament.json"), "w", encoding="utf-8") as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

        generated_ids.append(next_token_id)
        prev_tokens.append(next_token_id)
        input_ids = torch.tensor([[next_token_id]], device=device)
        if next_token_id == tokenizer.eos_token_id: break

    return prompt + tokenizer.decode(generated_ids, skip_special_tokens=True)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate watermarked text (with tournament JSON traces)")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompt-file", type=str, default=None)
    p.add_argument("--out", type=str, default=OUTPUT_TXT)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top-k", type=int, default=TOP_K if TOP_K is not None else -1)
    p.add_argument("--top-p", type=float, default=TOP_P if TOP_P is not None else -1.0)
    p.add_argument("--no-trace", action="store_true", help="Disable per-step tournament JSON traces")
    return p

def read_prompt(args) -> str:
    if args.prompt is not None: return args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f: return f.read().strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data and data.strip(): return data.strip()
    try:
        print("Enter your prompt: ", end="", flush=True); return input().strip()
    except EOFError:
        return ""

def main():
    global TRACE_TOURNAMENT
    args = build_arg_parser().parse_args()
    if args.no_trace:
        TRACE_TOURNAMENT = False

    device = get_device(); print(f"Using device: {device}", flush=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    top_k = None if args.top_k is None or args.top_k < 0 else int(args.top_k)
    top_p = None if args.top_p is None or args.top_p < 0 else float(args.top_p)

    prompt = read_prompt(args)
    if not prompt:
        print("No prompt provided (empty). Exiting.", flush=True)
        sys.exit(0)

    combined_text = generate_text(
        prompt=prompt, model=model, tokenizer=tokenizer, device=device,
        key=WATERMARK_KEY, max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature), top_k=top_k, top_p=top_p, seed=SEED
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"\nSaved generated text to {args.out}", flush=True)

if __name__ == "__main__":
    main()













