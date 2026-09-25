"""
estimate_delta_v3.py

Correct implementation of delta^{(t,ell)} from Proposition 1.

The key insight from the proof: delta^{(t,ell)} is the TV distance between
p_LCE and p_JCE at position (t,ell) CONDITIONAL on the two trajectories
agreeing at all previous positions. Under Gumbel-Max coupling, once they
agree at a position, the shared noise means they will sample the same token
iff TV < 1. The bound sums TV at each position where they could FIRST diverge.

Correct algorithm:
1. Tokenize the LCE and JCE prefixes (up to k*)
2. After k*, generate tokens JOINTLY:
   - At each position, compute p_LCE and p_JCE from their respective prefixes
   - Compute TV(p_LCE, p_JCE)
   - Simulate shared Gumbel noise: sample one token that both would agree on
     with probability 1 - TV (maximal coupling)
   - If they agree: append same token to both, continue
   - If they disagree: record position, stop (bound contribution = TV at this pos)
     Actually: sum all TV values up to first disagreement
3. Sum delta^{(t,ell)} = sum of TV at each position before first divergence

This is O(L) forward passes where L = downstream tokens until divergence.
"""

import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NEUTRAL_BASELINE = "I have no additional recommendation. Please continue."

ADV_AGENTS = {
    "travel_planning": "PLANNER_AGENT",
    "financial_article_writing": "CHIEF_EDITOR",
    "code_generation": "CEO",
    "multi_agent_debate": "agent_4",
}

def safe_model_name(model_id="Qwen/Qwen3-8B"):
    return model_id.split("/")[-1]


def parse_trajectory(traj_str):
    messages = []
    for m in re.finditer(r"TextMessage\(source='([^']+)'", traj_str):
        source = m.group(1)
        start = m.start()
        chunk = traj_str[start:start+10000]
        sq = re.search(r"content='((?:[^'\\]|\\.)*)'", chunk, re.DOTALL)
        dq = re.search(r'content="((?:[^"\\]|\\.)*)"', chunk, re.DOTALL)
        if sq and dq:
            content = sq.group(1) if sq.start() < dq.start() else dq.group(1)
        elif sq:
            content = sq.group(1)
        elif dq:
            content = dq.group(1)
        else:
            content = ""
        content = content.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
        messages.append({"source": source, "content": content})
    return messages


def load_row(baseline_root, environment, safe, row_id, model_id="Qwen/Qwen3-8B"):
    model_name = safe_model_name(model_id)
    adv = ADV_AGENTS[environment]
    safe_suffix = "/safe=True" if safe else ""
    row_dir = (Path(baseline_root) / f"env={environment}" / f"model={model_name}"
               / f"adv={adv}{safe_suffix}" / f"row={row_id}")
    index_path = row_dir / "index.json"
    factual_path = row_dir / "factual.json"
    if not index_path.exists() or not factual_path.exists():
        return None, None
    return json.load(open(factual_path)), json.load(open(index_path))


def get_intervention_info(index):
    cf_runs = index.get("counterfactual_runs", [])
    if not cf_runs:
        return None
    intervention = cf_runs[0].get("intervention", {})
    k_star = intervention.get("resolved_cf_call_idx",
              intervention.get("cf_call_idx"))
    if k_star is None:
        return None
    return {
        "k_star": k_star,
        "cf_agent": intervention.get("cf_agent", ""),
        "factual_response_text": intervention.get("factual_response_text", ""),
    }


def build_prefix_texts(factual, info):
    """
    Returns (lce_prefix_text, jce_prefix_text, k_star_msg_idx).
    Both prefixes include all messages up to AND including k*.
    The downstream messages after k* are NOT included — we will generate them.
    """
    traj_str = factual.get("factual", {}).get("trajectory", "")
    messages = parse_trajectory(traj_str)
    if not messages:
        raise ValueError("No messages parsed")

    k_star = info["k_star"]
    cf_agent = info["cf_agent"]

    lce_parts = []
    jce_parts = []
    k_star_msg_idx = None

    for global_idx, msg in enumerate(messages):
        source = msg["source"]
        content = msg["content"]

        if source == cf_agent:
            is_k_star = (global_idx == k_star)
            is_pre_target = (global_idx <= k_star)
            lce_content = NEUTRAL_BASELINE if is_k_star else content
            jce_content = NEUTRAL_BASELINE if is_pre_target else content
            if is_k_star:
                k_star_msg_idx = global_idx
        else:
            lce_content = content
            jce_content = content

        lce_parts.append(f"[{source}]: {lce_content}")
        jce_parts.append(f"[{source}]: {jce_content}")

        # stop after k* — don't include downstream messages in prefix
        if global_idx == k_star:
            break

    if k_star_msg_idx is None:
        # k* not found — use all messages up to k_star index
        k_star_msg_idx = min(k_star, len(messages) - 1)

    lce_prefix = "\n".join(lce_parts)
    jce_prefix = "\n".join(jce_parts)
    return lce_prefix, jce_prefix


@torch.no_grad()
def compute_delta_sum(model, tokenizer, lce_prefix, jce_prefix,
                      max_downstream=512, device="cuda:0"):
    """
    Correctly compute sum of delta^{(t,ell)} by jointly generating downstream
    tokens under maximal Gumbel-Max coupling.

    At each step:
    - Compute p_LCE and p_JCE from current prefixes
    - TV = 0.5 * |p_LCE - p_JCE|_1
    - Accumulate TV into delta_sum
    - Sample shared token via maximal coupling:
        with prob (1-TV): both sample the same token (agreement)
        with prob TV: they sample different tokens (divergence)
    - If divergence: stop (bound is already accumulated)
    - If agreement: append token to both prefixes, continue
    """
    lce_ids = tokenizer.encode(lce_prefix, return_tensors="pt").to(device)
    jce_ids = tokenizer.encode(jce_prefix, return_tensors="pt").to(device)

    delta_sum = 0.0
    tv_values = []
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    for step in range(max_downstream):
        # forward pass under each prefix
        lce_logits = model(lce_ids).logits[0, -1, :].float()
        jce_logits = model(jce_ids).logits[0, -1, :].float()

        p_lce = torch.softmax(lce_logits, dim=-1)
        p_jce = torch.softmax(jce_logits, dim=-1)

        # TV distance at this position
        tv = 0.5 * (p_lce - p_jce).abs().sum().item()
        delta_sum += tv
        tv_values.append(tv)

        # maximal coupling: determine if they agree
        # under maximal coupling, disagreement prob = TV
        u = torch.rand(1, generator=rng).item()
        if u < tv:
            # disagreement — stop
            break

        # agreement — sample shared token from coupling distribution
        # the coupling samples from min(p_lce, p_jce) / (1-TV) on agreement
        min_p = torch.minimum(p_lce, p_jce).cpu()
        if (1 - tv) > 1e-8:
            coupling_dist = min_p / (1 - tv)
        else:
            coupling_dist = p_lce.cpu()
        shared_token = torch.multinomial(coupling_dist, 1, generator=rng).to(device)

        # append shared token to both sequences
        lce_ids = torch.cat([lce_ids, shared_token.unsqueeze(0)], dim=1)
        jce_ids = torch.cat([jce_ids, shared_token.unsqueeze(0)], dim=1)

    return delta_sum, np.array(tv_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="results_v2_qwen_baseline")
    parser.add_argument("--per-row", default="outputsALL/per_row.csv")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", default="outputsALL/")
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-downstream", type=int, default=256,
                        help="Max downstream tokens to generate per row")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print("Model loaded.")

    per_row = pd.read_csv(args.per_row)
    per_row["safe"] = per_row["safe"].astype(str).str.lower().isin(["true", "1"])
    baseline = per_row[
        (per_row["cf_mode"] == "baseline") & (~per_row["safe"]) &
        per_row["causal_effect_mean"].notna()
    ].copy()

    if args.environment:
        baseline = baseline[baseline["environment"] == args.environment]

    results = []
    for _, row in baseline.head(args.max_rows).iterrows():
        env = row["environment"]
        row_id = int(row["row_id"])
        cfe = row["causal_effect_mean"]
        print(f"\n[{env} row={row_id}] CFE={cfe:.1f}", end=" ", flush=True)

        factual, index = load_row(args.baseline_root, env, False, row_id, args.model_id)
        if factual is None:
            print("SKIP (missing files)")
            continue

        info = get_intervention_info(index)
        if info is None:
            print("SKIP (no intervention info)")
            continue

        try:
            lce_prefix, jce_prefix = build_prefix_texts(factual, info)
        except Exception as e:
            print(f"SKIP (prefix error: {e})")
            continue

        # check if LCE == JCE (single pre-target call case)
        if lce_prefix == jce_prefix:
            print("delta_sum=0.0 (LCE==JCE, single pre-target call)")
            results.append({
                "environment": env, "row_id": row_id, "cfe": cfe,
                "delta_sum": 0.0, "n_steps": 0,
                "mean_tv": 0.0, "max_tv": 0.0,
                "lce_eq_jce": True,
            })
            continue

        try:
            delta_sum, tv_vals = compute_delta_sum(
                model, tokenizer, lce_prefix, jce_prefix,
                max_downstream=args.max_downstream, device=device
            )
        except Exception as e:
            print(f"SKIP (compute error: {e})")
            continue

        n_steps = len(tv_vals)
        mean_tv = float(tv_vals.mean()) if n_steps > 0 else 0.0
        max_tv = float(tv_vals.max()) if n_steps > 0 else 0.0

        print(f"delta_sum={delta_sum:.4f} n_steps={n_steps} "
              f"mean_tv={mean_tv:.4f} max_tv={max_tv:.4f}")

        results.append({
            "environment": env, "row_id": row_id, "cfe": cfe,
            "delta_sum": delta_sum, "n_steps": n_steps,
            "mean_tv": mean_tv, "max_tv": max_tv,
            "lce_eq_jce": False,
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, "delta_estimates_v3.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    if len(df) > 0:
        print(f"\nMean delta_sum (upper bound on E[|IE_beta|]): {df['delta_sum'].mean():.4f}")
        print(f"Max delta_sum: {df['delta_sum'].max():.4f}")
        print(f"Rows with LCE==JCE (delta=0 by construction): {df['lce_eq_jce'].sum()}")
        print(f"\nBy environment:")
        print(df.groupby("environment")[["delta_sum","n_steps","mean_tv"]].mean().to_string())