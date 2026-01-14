from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

load_dotenv()

token = os.getenv("HF_TOKEN")
login(token=token)

model_id = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()


def sample_gumbel(shape, generator: torch.Generator, device, eps: float = 1e-20):
    # Gumbel(0,1) via -log(-log(U))
    U = torch.rand(shape, generator=generator, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

@torch.no_grad()
def gumbel_max_step(
    model,
    input_ids: torch.LongTensor,
    temperature: float,
    generator: torch.Generator,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tuple[int, torch.Tensor]:
    """
    One decoding step using the Gumbel-Max SCM:
      t = argmax_v (log p(v) + g_v)
    Optionally with top-k / top-p restriction (see paper remarks).
    Returns: chosen_token_id, probs (full vocab probs for inspection)
    """
    out = model(input_ids=input_ids)
    logits = out.logits[:, -1, :]  # [1, vocab]
    logits = logits / max(temperature, 1e-8)

    probs = F.softmax(logits, dim=-1)  # [1, vocab]
    logp = torch.log(probs + 1e-20)    # [1, vocab]

    vocab = logp.shape[-1]

    # Candidate restriction (optional).
    # Note: the paper says top-k/top-p may break strict "stability" guarantees,
    # but it's often used in practice. :contentReference[oaicite:3]{index=3}
    mask = torch.zeros((vocab,), dtype=torch.bool, device=logp.device)

    if top_k is not None:
        topk_vals, topk_ids = torch.topk(probs[0], k=min(top_k, vocab))
        mask[topk_ids] = True
    elif top_p is not None:
        sorted_probs, sorted_ids = torch.sort(probs[0], descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        mask[sorted_ids[keep]] = True
    else:
        mask[:] = True

    # Draw Gumbel noise for all vocab (or you can draw only for masked and keep -inf elsewhere)
    g = sample_gumbel((vocab,), generator=generator, device=logp.device)

    scores = logp[0] + g
    scores[~mask] = -float("inf")

    next_id = int(torch.argmax(scores).item())
    return next_id, probs[0].detach()

@torch.no_grad()
def factual_generate_with_rng_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: int = 1234,
):
    """
    Factual generation + store RNG states per step (Algorithm 1 idea). :contentReference[oaicite:4]{index=4}
    Returns:
      generated_ids (including prompt),
      rng_states (list of torch ByteTensor states, length = #steps),
      probs_trace (optional: list of top info per step if you want)
    """
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    rng_states: List[torch.ByteTensor] = []
    probs_trace: List[torch.Tensor] = []

    for _ in range(max_new_tokens):
        # store RNG state *before* sampling this token
        rng_states.append(gen.get_state())

        next_id, probs = gumbel_max_step(
            model=model,
            input_ids=input_ids,
            temperature=temperature,
            generator=gen,
            top_k=top_k,
            top_p=top_p,
        )
        probs_trace.append(probs)

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1
        )

        if next_id == tokenizer.eos_token_id:
            break

    return input_ids, rng_states, probs_trace

@torch.no_grad()
def counterfactual_generate_reusing_rng_states(
    model,
    tokenizer,
    new_prompt: str,
    rng_states: List[torch.ByteTensor],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """
    Counterfactual generation: same sampler noise via restoring RNG state per step
    (regen u_j from r_j), per Algorithm 1. :contentReference[oaicite:5]{index=5}
    """
    device = model.device
    input_ids = tokenizer(new_prompt, return_tensors="pt").input_ids.to(device)

    gen = torch.Generator(device=device)

    # We will reuse exactly as many steps as we have stored states for,
    # or max_new_tokens, whichever is smaller.
    steps = min(max_new_tokens, len(rng_states))

    for t in range(steps):
        gen.set_state(rng_states[t])

        next_id, _ = gumbel_max_step(
            model=model,
            input_ids=input_ids,
            temperature=temperature,
            generator=gen,
            top_k=top_k,
            top_p=top_p,
        )

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1
        )

        if next_id == tokenizer.eos_token_id:
            break

    return input_ids


prompt_factual = "Plan a 4-day trip to Paris."
prompt_cf = "Plan a 5-day trip to Paris."

max_new = 200
temp = 0.8

# 1) Factual run (store RNG states)
factual_ids, rng_states, _ = factual_generate_with_rng_states(
    model, tokenizer,
    prompt_factual,
    max_new_tokens=max_new,
    temperature=temp,
    top_p=None,   # start with None for cleanest theory
    top_k=None,
    seed=2025,
)

# 2) Counterfactual run (reuse RNG states under modified prompt)
cf_ids = counterfactual_generate_reusing_rng_states(
    model, tokenizer,
    prompt_cf,
    rng_states=rng_states,
    max_new_tokens=max_new,
    temperature=temp,
    top_p=None,
    top_k=None,
)

print("\n=== FACTUAL ===")
print(tokenizer.decode(factual_ids[0], skip_special_tokens=True))

print("\n=== COUNTERFACTUAL (Gumbel-Max, same noise) ===")
print(tokenizer.decode(cf_ids[0], skip_special_tokens=True))