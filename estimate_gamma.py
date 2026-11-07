"""
estimate_gamma.py

Rerun of the prefix sensitivity estimate (Table 3, app:sensitivity) with the corrected
Proposition 1. The bound uses gamma, the probability that Gumbel-Max with shared noise
picks different tokens under the LCE and JCE prefixes, in place of the total variation
distance delta:

    gamma(p, q) = sum_v [ p_v - 1 / S_v ],   S_v = sum_u max(p_u / p_v, q_u / q_v),
    TV(p, q) <= gamma(p, q) <= 2 TV / (1 + TV).

Prompts are reconstructed exactly as in estimate_delta_v4.py (same build_prompts). The
downstream path is generated with shared-noise Gumbel-Max: at every step one Gumbel vector
G is drawn and both prefixes take argmax(z / T + G). Sums of delta and gamma include the
position of the first disagreement. estimate_delta_v4.py instead sampled a path of a
simulated maximal coupling (disagree with probability TV, else draw from min(p, q)/(1-TV)),
so paths here differ from the original run; delta itself is computed the same way and the
disagreement position was also included there.

Code Generation is identical under the safe and non-safe conditions, so it is computed once
(condition NS).

Usage:
    python estimate_gamma.py --test                    # unit tests only (CPU)
    CUDA_VISIBLE_DEVICES=1 python estimate_gamma.py --model qwen
    CUDA_VISIBLE_DEVICES=1 python estimate_gamma.py --model llama
Output: outputs/prefix_sensitivity/prefix_sensitivity_gamma_<model>.csv (make_table3.py
concatenates them).
"""

import argparse
import math
import os
from pathlib import Path

import torch

ENV_SHORT = {"travel_planning": "TP", "code_generation": "CG",
             "multi_agent_debate": "MAD", "financial_article_writing": "FAW"}
MODELS = {
    "qwen": dict(model_id="Qwen/Qwen3-8B", dtype="float16",
                 baseline_root="results/results_v2_qwen_baseline", per_row="outputs/outputs_qwen/per_row.csv"),
    "llama": dict(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct", dtype="bfloat16",
                  baseline_root="results/results_v2_llama_baseline", per_row="outputs/outputs_llama/per_row.csv"),
}


def delta_gamma(logits_l, logits_j, temperature=0.7):
    """(TV distance, shared-noise Gumbel-Max disagreement probability) between the
    tempered full-vocabulary softmaxes of two logit vectors. O(V log V), float64."""
    lp = torch.log_softmax(logits_l.double() / temperature, dim=-1)
    lq = torch.log_softmax(logits_j.double() / temperature, dim=-1)
    p, q = lp.exp(), lq.exp()
    delta = 0.5 * (p - q).abs().sum()
    order = torch.argsort(lp - lq, descending=True)
    p_s, q_s = p[order], q[order]
    head_p = torch.cumsum(p_s, dim=0)
    tail_q = torch.flip(torch.cumsum(torch.flip(q_s, [0]), dim=0), [0]) - q_s
    # tokens with p_v = 0 (underflow) contribute p_v - 1/S_v = 0 - 0; guard the division
    mask = p_s > 0
    S = head_p[mask] / p_s[mask] + tail_q[mask] / q_s[mask]
    gamma = (p_s[mask] - 1.0 / S).sum()
    return delta.item(), gamma.item()


# ---------------------------------------------------------------- unit tests
def _gamma_brute(p, q):
    p, q = p.double(), q.double()
    S = torch.maximum(p[None, :] / p[:, None], q[None, :] / q[:, None]).sum(1)  # S_v, row v
    return (p - 1.0 / S).sum().item()


def run_unit_tests():
    results = {}
    g = torch.Generator().manual_seed(0)
    dirich = torch.distributions.Dirichlet(torch.ones(50, dtype=torch.float64))
    torch.manual_seed(0)
    err = 0.0
    for _ in range(100):
        p, q = dirich.sample(), dirich.sample()
        _, gm = delta_gamma(p.log(), q.log(), 1.0)
        err = max(err, abs(gm - _gamma_brute(p, q)))
    results["T1 brute force (V=50, 100 pairs)"] = (err < 1e-12, f"max abs err {err:.2e}")

    p = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    q = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    d, gm = delta_gamma(p.log(), q.log(), 1.0)
    results["T2 known value"] = (abs(gm - 0.3) < 1e-12 and abs(d - 0.25) < 1e-12, f"gamma {gm:.15f}, delta {d:.15f}")

    d, gm = delta_gamma(p.log(), p.log(), 1.0)
    results["T3 identity"] = (abs(gm) < 1e-12 and abs(d) < 1e-12, f"gamma {gm:.1e}, delta {d:.1e}")

    dir1k = torch.distributions.Dirichlet(torch.full((1000,), 0.3, dtype=torch.float64))
    p, q = dir1k.sample(), dir1k.sample()
    _, gm = delta_gamma(p.log(), q.log(), 1.0)
    n, dis = 10**6, 0
    lp, lq = p.log(), q.log()
    for _ in range(10):                       # 10 chunks of 1e5 draws
        u = torch.rand(n // 10, 1000, generator=g, dtype=torch.float64).clamp_min(1e-300)
        G = -torch.log(-torch.log(u))
        dis += ((lp + G).argmax(1) != (lq + G).argmax(1)).sum().item()
    emp = dis / n
    se = math.sqrt(gm * (1 - gm) / n)
    results["T4 Monte Carlo (V=1000, 1e6 draws)"] = (abs(emp - gm) < 3 * se,
                                                     f"gamma {gm:.5f}, empirical {emp:.5f}, 3SE {3*se:.5f}")
    return results


# ---------------------------------------------------------------- coupled path
@torch.no_grad()
def coupled_gumbel_path(model, tokenizer, lce, jce, max_tokens, temperature, stop_ids, gen):
    dev = model.device
    out_l = model(input_ids=tokenizer(lce, return_tensors="pt").input_ids.to(dev), use_cache=True)
    out_j = model(input_ids=tokenizer(jce, return_tensors="pt").input_ids.to(dev), use_cache=True)
    sum_d = sum_g = sum_ub = 0.0
    n_pos, disagreed, viol = 0, False, 0
    for _ in range(max_tokens):
        zL, zJ = out_l.logits[0, -1], out_j.logits[0, -1]
        d, gm = delta_gamma(zL, zJ, temperature)
        ub = 2 * d / (1 + d)
        if not (d - 1e-9 <= gm <= ub + 1e-9):
            viol += 1
        sum_d += d; sum_g += gm; sum_ub += ub; n_pos += 1
        u = torch.rand(zL.shape[-1], generator=gen, device=dev, dtype=torch.float64).clamp_min(1e-300)
        G = -torch.log(-torch.log(u))
        xL = int((zL.double() / temperature + G).argmax())
        xJ = int((zJ.double() / temperature + G).argmax())
        if xL != xJ:
            disagreed = True
            break
        if xL in stop_ids:
            break
        t = torch.tensor([[xL]], device=dev)
        out_l = model(input_ids=t, past_key_values=out_l.past_key_values, use_cache=True)
        out_j = model(input_ids=t, past_key_values=out_j.past_key_values, use_cache=True)
    return dict(sum_delta=sum_d, sum_gamma=sum_g, sum_2d_over_1pd=sum_ub,
                n_positions=n_pos, disagreed=disagreed, bound_violations=viol)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--test", action="store_true", help="run the unit tests only")
    ap.add_argument("--output-dir", default="outputs/prefix_sensitivity")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tests = run_unit_tests()
    for name, (ok, msg) in tests.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {msg}")
    with open(Path(args.output_dir) / "unit_tests.txt", "w") as f:
        for name, (ok, msg) in tests.items():
            f.write(f"{'PASS' if ok else 'FAIL'}\t{name}\t{msg}\n")
    if not all(ok for ok, _ in tests.values()):
        raise SystemExit("unit tests failed")
    if args.test:
        raise SystemExit(0)

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import estimate_delta_v4 as ev

    cfg = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], dtype=getattr(torch, cfg["dtype"]), device_map="cuda:0").eval()
    renderer = ev._Renderer(tok)
    stop_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|eom_id|>", "<|im_end|>")
                                     if tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id)}
    model_name = cfg["model_id"].split("/")[-1]
    out_path = Path(args.output_dir) / f"prefix_sensitivity_gamma_{args.model}.csv"
    rows, spec_cache = [], {}

    for environment, adv in ev.ADV_AGENTS.items():
        conds = [("NS", False)] if environment == "code_generation" else [("N", False), ("S", True)]
        for cond, safe in conds:
            base = Path(cfg["baseline_root"]) / f"env={environment}" / f"model={model_name}" / f"adv={adv}"
            if safe:
                base = base / "safe=True"
            for row_dir in sorted(base.glob("row=*"), key=lambda p: int(p.name.split("=")[1])):
                if not (row_dir / "index.json").exists():
                    continue
                row_id = int(row_dir.name.split("=")[1])
                rec = dict(model=args.model, env=ENV_SHORT[environment], condition=cond, scenario_idx=row_id)
                try:
                    built, why = ev.build_prompts(row_dir, environment, safe, renderer, spec_cache)
                except Exception as e:
                    built, why = None, f"error: {type(e).__name__}: {e}"
                if built is None:
                    print(f"[{environment} {cond} row={row_id}] skip: {why}", flush=True)
                    continue
                rec.update(k_star=built["k_star"], prompt_match=bool(built["prompt_match"]))
                if built["lce"] == built["jce"]:
                    rec.update(lce_eq_jce=True, n_positions=0, disagreed=False, sum_delta=0.0,
                               sum_gamma=0.0, sum_2d_over_1pd=0.0, bound_violations=0)
                else:
                    gen = torch.Generator(device=model.device).manual_seed(args.seed + row_id)
                    rec.update(lce_eq_jce=False, **coupled_gumbel_path(
                        model, tok, built["lce"], built["jce"], ev.MAX_TOKENS.get(environment, 256),
                        args.temperature, stop_ids, gen))
                print(f"[{environment} {cond} row={row_id}] match={rec['prompt_match']} "
                      f"sum_d={rec['sum_delta']:.3f} sum_g={rec['sum_gamma']:.3f} n={rec['n_positions']} "
                      f"dis={rec['disagreed']}", flush=True)
                rows.append(rec)
            pd.DataFrame(rows).to_csv(out_path, index=False)
    cols = ["model", "env", "condition", "scenario_idx", "k_star", "lce_eq_jce", "n_positions", "disagreed",
            "sum_delta", "sum_gamma", "sum_2d_over_1pd", "prompt_match", "bound_violations"]
    pd.DataFrame(rows)[cols].to_csv(out_path, index=False)
    print(f"saved {out_path} ({len(rows)} rows)")
