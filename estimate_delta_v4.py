"""
estimate_delta_v4.py

Empirical estimate of the sensitivity bound of Proposition 1 (app:sensitivity):

    E|IE_beta| <= E[ sum_{(t,l) downstream} delta^{(t,l)} ],
    delta^{(t,l)} = TV( p(. | prefix_LCE), p(. | prefix_JCE) ),

where LCE replaces only beta's last pre-target call k* with the neutral text and
JCE replaces every pre-target call of beta with it.

Changes from estimate_delta_v3.py (whose numbers should not be used):
  - prompts are the ones the agents actually saw: the downstream agent's own
    message history from the saved team state, its system prompt (adversarial
    prompt if it is the adversary) and tool schemas, rendered with the same
    _messages_to_prompt as run_experiments_v3.py. v3 fed raw "[agent]: text"
    lines with no chat template or system prompt.
  - k* is a call index; v3 used it as an index into the parsed TextMessage list,
    which also contains the user task and omits tool/reflection calls.
  - the intervention text is the cf_text actually injected in that row (v3 used
    a generic sentence that no run used).
  - TV is computed on the tempered sampling distributions (temperature 0.7, as in
    every run); v3 used temperature 1.
  - model dtype matches the runs (Qwen float16, Llama bfloat16).

For every row the reconstructed *factual* prompt of call k*+1 is checked against
the last 250 characters recorded for that call in the factual call log
(prompt_match). Downstream tokens are generated jointly under the maximal
(Gumbel-Max) coupling of the two next-token distributions, summing TV until the
first disagreement or the end of the downstream call (k*+1). This one-sample
estimate covers only the first downstream call, so it is a lower estimate of the
full right-hand side of the bound, which also sums over later calls.

Usage:
    CUDA_VISIBLE_DEVICES=1 python estimate_delta_v4.py --model-id Qwen/Qwen3-8B --dtype float16 \
        --baseline-root results/results_v2_qwen_baseline --per-row outputs/outputs_qwen/per_row.csv \
        --output-dir outputs/outputs_qwen/
    CUDA_VISIBLE_DEVICES=1 python estimate_delta_v4.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dtype bfloat16 --baseline-root results/results_v2_llama_baseline \
        --per-row outputs/outputs_llama/per_row.csv --output-dir outputs/outputs_llama/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from pydantic import TypeAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.argv, _argv = sys.argv[:1], sys.argv          # run_experiments_v3 parses nothing on import
import run_experiments_v3 as rx                     # noqa: E402
sys.argv = _argv
from agents.adversarial_agent import AdversarialAgent  # noqa: E402
from autogen_core.models import LLMMessage, SystemMessage  # noqa: E402

ADV_AGENTS = {
    "travel_planning": "PLANNER_AGENT",
    "financial_article_writing": "CHIEF_EDITOR",
    "code_generation": "CEO",
    "multi_agent_debate": "agent_4",
}
MAX_TOKENS = {"financial_article_writing": 512}  # 256 elsewhere, as in the runs
MSG = TypeAdapter(LLMMessage)


class _DummyClient:
    model_info = {"family": "hf", "function_calling": True, "vision": False,
                  "json_output": False, "structured_output": False}

    async def create(self, *a, **k):
        raise RuntimeError("not used")


class _Renderer:
    """Just enough of HFModelClient to call its _messages_to_prompt unchanged."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        template = getattr(tokenizer, "chat_template", None) or ""
        self._tool_format = "hermes" if "<tool_call>" in template else "json"

    def render(self, messages, tools):
        return rx.HFModelClient._messages_to_prompt(self, messages, tools=tools)


def _content_key(c):
    return str(c).strip()[:200]


def agent_specs(environment, safe, target_action):
    """name -> (system prompt, tools) with the adversary installed in its slot."""
    env = rx._make_environment(environment, _DummyClient(), safe)
    specs = {a.name: (a._system_messages[0].content, list(getattr(a, "_tools", None) or []))
             for a in env.agent_models}
    adv = ADV_AGENTS[environment]
    adv_prompt = AdversarialAgent(environment).adversarial_system_messages[adv].format(goal=target_action)
    specs[adv] = (adv_prompt, [])
    return specs


def build_prompts(row_dir, environment, safe, renderer, spec_cache):
    idx = json.load(open(row_dir / "index.json"))
    iv = idx["counterfactual_runs"][0]["intervention"]
    k = int(iv["resolved_cf_call_idx"])
    beta = iv["cf_agent"]
    cf_text = iv["cf_text"]
    call_log = json.load(open(row_dir / "tape_meta.json"))["call_log"]
    factual = json.load(open(row_dir / "factual.json"))
    states = factual["factual"]["team_states"]["agent_states"]
    # first downstream call made by an agent; calls logged as "unknown" are the group
    # chat's speaker selection, which is not an agent and has no saved context
    nxt = next((c for c in call_log[k + 1:] if c["agent"] in states), None)
    if nxt is None:
        return None, "no downstream agent call"
    a = nxt["agent"]
    if a not in states:
        return None, f"no saved state for {a}"
    ctx = states[a]["agent_state"]["llm_context"]["messages"]

    key = (environment, safe, factual.get("target_action"))
    if key not in spec_cache:
        spec_cache[key] = agent_specs(environment, safe, factual.get("target_action"))
    system, all_tools = spec_cache[key][a]

    def render(messages):
        is_reflection = bool(messages) and messages[-1].get("type") == "FunctionExecutionResultMessage"
        tools = all_tools if (all_tools and not is_reflection) else None
        msgs = [SystemMessage(content=system)] + [MSG.validate_python(m) for m in messages]
        return renderer.render(msgs, tools)

    # cut the downstream agent's history right after beta's k* message. The same
    # text can occur more than once, so among beta's messages whose content matches
    # the k* response, take the one whose rendered factual prompt reproduces the
    # recorded prompt tail of call k*+1; otherwise the first content match.
    target = _content_key(call_log[k]["response_text"])  # already speaker-tag stripped by the client
    candidates = [j for j, m in enumerate(ctx)
                  if m.get("source") == beta and isinstance(m.get("content"), str)
                  and (_content_key(m["content"]) == target
                       or _content_key(m["content"]).startswith(target[:80]))]
    if not candidates:
        return None, "k* message not found in downstream context"
    cut = next((j for j in candidates if render(ctx[:j + 1])[-250:] == nxt["prompt_preview"]), candidates[0])
    hist = ctx[:cut + 1]

    def replaced(only_last):
        out = []
        for j, m in enumerate(hist):
            m = dict(m)
            if m.get("source") == beta and (j == cut or not only_last):
                m["content"] = cf_text
                if m.get("type") == "AssistantMessage":
                    m.pop("thought", None)
            out.append(m)
        return out

    fact_prompt = render(hist)
    return {
        "k_star": k, "beta": beta, "downstream_agent": a,
        "n_beta_pre_calls": sum(1 for m in hist if m.get("source") == beta),
        "prompt_match": fact_prompt[-250:] == nxt["prompt_preview"],
        "downstream_call": int(nxt["call_idx"]),
        "lce": render(replaced(True)), "jce": render(replaced(False)),
    }, None


@torch.no_grad()
def coupled_delta_sum(model, tokenizer, lce, jce, max_tokens, temperature, stop_ids, gen):
    """Sum TV along one path of the maximal coupling until the first disagreement
    or the end of the downstream message."""
    dev = model.device
    ids_l = tokenizer(lce, return_tensors="pt").input_ids.to(dev)
    ids_j = tokenizer(jce, return_tensors="pt").input_ids.to(dev)
    out_l = model(input_ids=ids_l, use_cache=True)
    out_j = model(input_ids=ids_j, use_cache=True)
    tvs, reason = [], "max_tokens"
    for _ in range(max_tokens):
        p = torch.softmax(out_l.logits[0, -1].float() / temperature, -1)
        q = torch.softmax(out_j.logits[0, -1].float() / temperature, -1)
        tv = 0.5 * (p - q).abs().sum().item()
        tvs.append(tv)
        if torch.rand(1, generator=gen).item() < tv:
            reason = "diverged"
            break
        shared = torch.minimum(p, q)
        tok = torch.multinomial((shared / shared.sum()).cpu(), 1, generator=gen).item()
        if tok in stop_ids:
            reason = "end_of_message"
            break
        t = torch.tensor([[tok]], device=dev)
        out_l = model(input_ids=t, past_key_values=out_l.past_key_values, use_cache=True)
        out_j = model(input_ids=t, past_key_values=out_j.past_key_values, use_cache=True)
    return sum(tvs), tvs, reason


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--baseline-root", default="results/results_v2_qwen_baseline")
    ap.add_argument("--per-row", default="outputs/outputs_qwen/per_row.csv")
    ap.add_argument("--output-dir", default="outputs/outputs_qwen/")
    ap.add_argument("--conditions", nargs="+", default=["nonsafe", "safe"], choices=["nonsafe", "safe"])
    ap.add_argument("--environment", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=getattr(torch, args.dtype), device_map="cuda:0").eval()
    renderer = _Renderer(tok)
    stop_ids = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|eom_id|>", "<|im_end|>")
                                     if tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id)}
    per_row = pd.read_csv(args.per_row)
    per_row["safe"] = per_row["safe"].astype(str).str.lower().isin(["true", "1"])
    cfe = {(r.environment, bool(r.safe), int(r.row_id)): r.causal_effect_mean
           for r in per_row[per_row.cf_mode == "baseline"].itertuples()}
    model_name = args.model_id.split("/")[-1]
    out_path = os.path.join(args.output_dir, "delta_estimates.csv")
    rows, spec_cache = [], {}

    for environment in ADV_AGENTS:
        if args.environment and environment != args.environment:
            continue
        for cond in args.conditions:
            safe = cond == "safe"
            base = Path(args.baseline_root) / f"env={environment}" / f"model={model_name}" / f"adv={ADV_AGENTS[environment]}"
            if safe:
                base = base / "safe=True"
            for row_dir in sorted(base.glob("row=*"), key=lambda p: int(p.name.split("=")[1])):
                if not (row_dir / "index.json").exists():
                    continue
                row_id = int(row_dir.name.split("=")[1])
                rec = {"environment": environment, "safe": safe, "row_id": row_id,
                       "cfe": cfe.get((environment, safe, row_id))}
                try:
                    built, why = build_prompts(row_dir, environment, safe, renderer, spec_cache)
                except Exception as e:  # keep going; record the reason
                    built, why = None, f"error: {type(e).__name__}: {e}"
                if built is None:
                    rec["skip_reason"] = why
                    print(f"[{environment} safe={safe} row={row_id}] skip: {why}")
                    rows.append(rec)
                    continue
                rec.update({k: built[k] for k in ("k_star", "downstream_call", "downstream_agent", "n_beta_pre_calls", "prompt_match")})
                if built["lce"] == built["jce"]:
                    rec.update(delta_sum=0.0, n_steps=0, mean_tv=0.0, max_tv=0.0,
                               stop_reason="lce_eq_jce", lce_eq_jce=True)
                else:
                    gen = torch.Generator().manual_seed(args.seed + row_id)
                    d, tvs, reason = coupled_delta_sum(model, tok, built["lce"], built["jce"],
                                                       MAX_TOKENS.get(environment, 256), args.temperature,
                                                       stop_ids, gen)
                    rec.update(delta_sum=d, n_steps=len(tvs), mean_tv=sum(tvs) / len(tvs),
                               max_tv=max(tvs), stop_reason=reason, lce_eq_jce=False)
                print(f"[{environment} safe={safe} row={row_id}] match={rec['prompt_match']} "
                      f"delta_sum={rec['delta_sum']:.3f} steps={rec['n_steps']} {rec['stop_reason']}", flush=True)
                rows.append(rec)
            pd.DataFrame(rows).to_csv(out_path, index=False)   # checkpoint after each condition

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    ok = df[df["delta_sum"].notna()]
    print(f"\nSaved -> {out_path}  ({len(ok)} estimated, {df['delta_sum'].isna().sum()} skipped)")
    if len(ok):
        print(ok.groupby(["environment", "safe"]).agg(
            n=("row_id", "count"), prompt_match=("prompt_match", "mean"),
            median_delta=("delta_sum", "median"), frac_below_1=("delta_sum", lambda x: (x < 1).mean()),
            lce_eq_jce=("lce_eq_jce", "sum")).round(3).to_string())
