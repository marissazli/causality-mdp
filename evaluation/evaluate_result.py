"""
evaluate_result.py  (v2 — multi-mode evaluation)

Walks baseline and replay results directories and produces CSVs needed for
all paper figures:

  outputs/asr.csv              — ASR per (environment, safe)
  outputs/ase.csv              — ASE per (environment, safe, cf_mode)
  outputs/per_row.csv          — per-row causal effect (all modes)
  outputs/replay_check.csv     — replay causal effects (sanity check)
  outputs/harm_category.csv    — causal effect joined with BAD-ACTS harm category

Usage:
    python evaluate_result.py \
        --baseline-root results_v2_qwen_baseline \
        --replay-root results_v2_qwen_replay \
        --dataset datasets/BAD-ACTS.csv \
        --output-dir outputs/
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os

from evaluation_functions import (
    evaluate_travel_planning,
    evaluate_financial_article_writing,
    evaluate_code_generation,
    evaluate_MAD,
)

ENVIRONMENTS = [
    "travel_planning",
    "financial_article_writing",
    "code_generation",
    "multi_agent_debate",
]

ADV_AGENTS = {
    "travel_planning": "PLANNER_AGENT",
    "financial_article_writing": "CHIEF_EDITOR",
    "code_generation": "CEO",
    "multi_agent_debate": "agent_4",
}

EVAL_FNS = {
    "travel_planning": evaluate_travel_planning,
    "financial_article_writing": evaluate_financial_article_writing,
    "code_generation": lambda dp: evaluate_code_generation(dp, dp["keywords"]),
    "multi_agent_debate": evaluate_MAD,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def safe_model_name(model_id="Qwen/Qwen3-8B"):
    return model_id.split("/")[-1]


def make_factual_dp(factual):
    merged = {}
    for key in ("id", "target_agent", "adversarial_agent", "target_action",
                "keywords", "backend", "environment", "task", "seed", "safe", "guardian"):
        if key in factual:
            merged[key] = factual[key]
    factual_run = factual.get("factual", {})
    for key in ("team_states", "sent_messages", "tickets", "files", "trajectory"):
        if key in factual_run:
            merged[key] = factual_run[key]
    return merged


def make_cf_dp(factual, cf_full):
    merged = {}
    for key in ("id", "target_agent", "adversarial_agent", "target_action",
                "keywords", "backend", "environment", "task", "seed", "safe", "guardian"):
        if key in factual:
            merged[key] = factual[key]
    for key in ("team_states", "sent_messages", "tickets", "files",
                "trajectory", "call_log", "tape_status"):
        if key in cf_full:
            merged[key] = cf_full[key]
        elif key in factual.get("factual", {}):
            merged[key] = factual["factual"][key]
    return merged


def get_cf_entries(row_dir):
    """Return list of (call_idx, sample_idx, cf_full_dict) from cf/ or index.json."""
    cf_dir = row_dir / "cf"
    entries = []

    if cf_dir.exists():
        for cf_path in sorted(cf_dir.glob("call_*_sample_*_full.json")):
            stem = cf_path.stem.replace("_full", "")
            parts = stem.split("_")
            try:
                call_idx = int(parts[1])
                sample_idx = int(parts[3])
            except (IndexError, ValueError):
                continue
            entries.append((call_idx, sample_idx, load_json(cf_path)))

    if not entries:
        # fallback: index.json counterfactual_runs
        index_path = row_dir / "index.json"
        if index_path.exists():
            index = load_json(index_path)
            # single-agent CF stored under "counterfactual_runs"
            for run in index.get("counterfactual_runs", []):
                call_idx = run.get("intervention", {}).get("resolved_cf_call_idx",
                           run.get("intervention", {}).get("cf_call_idx", 0))
                entries.append((int(call_idx), 0, run))
            # all-calls CF stored under "counterfactual_runs_by_call"
            for call_entry in index.get("counterfactual_runs_by_call", []):
                call_idx = int(call_entry["cf_call_idx"])
                for run in call_entry.get("runs", []):
                    entries.append((call_idx, int(run.get("sample_idx", 0)), run))

    return entries


def scan_results_root(root, environment, safe, cf_mode, eval_fn):
    """
    Scan a results root directory for one (environment, safe, cf_mode) combo.
    Returns list of row dicts.
    """
    model_name = safe_model_name()
    adv = ADV_AGENTS[environment]
    safe_suffix = "/safe=True" if safe else ""
    env_dir = Path(root) / f"env={environment}" / f"model={model_name}" / f"adv={adv}{safe_suffix}"

    if not env_dir.exists():
        return []

    rows = []
    for row_dir in sorted(env_dir.glob("row=*")):
        factual_path = row_dir / "factual.json"
        if not factual_path.exists():
            continue

        factual = load_json(factual_path)
        row_id = factual.get("id", row_dir.name)

        factual_dp = make_factual_dp(factual)
        try:
            y_factual = float(bool(eval_fn(factual_dp)))
        except Exception as e:
            print(f"  [warn] factual eval failed row {row_id} ({environment} safe={safe}): {e}")
            y_factual = float("nan")

        cf_entries = get_cf_entries(row_dir)

        if not cf_entries:
            # factual-only row (no CF yet)
            rows.append({
                "environment": environment,
                "safe": safe,
                "cf_mode": cf_mode,
                "row_id": row_id,
                "call_idx": None,
                "sample_idx": None,
                "y_factual": y_factual,
                "y_cf": float("nan"),
                "causal_effect": float("nan"),
                "target_agent": factual.get("target_agent"),
                "cf_agent": None,
                "keywords": factual.get("keywords"),
                "cf_mode_stored": None,
            })
            continue

        for call_idx, sample_idx, cf_full in cf_entries:
            cf_dp = make_cf_dp(factual, cf_full)
            try:
                y_cf = float(bool(eval_fn(cf_dp)))
            except Exception as e:
                print(f"  [warn] CF eval failed row {row_id} call {call_idx}: {e}")
                y_cf = float("nan")

            intervention = cf_full.get("intervention", {})
            rows.append({
                "environment": environment,
                "safe": safe,
                "cf_mode": cf_mode,
                "row_id": row_id,
                "call_idx": call_idx,
                "sample_idx": sample_idx,
                "y_factual": y_factual,
                "y_cf": y_cf,
                "causal_effect": y_cf - y_factual,
                "target_agent": factual.get("target_agent"),
                "cf_agent": intervention.get("cf_agent"),
                "keywords": factual.get("keywords"),
                "cf_mode_stored": intervention.get("cf_mode"),
            })

    return rows


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--baseline-root", type=str, default="results_v2_qwen_baseline")
    parser.add_argument("--replay-root", type=str, default="results_v2_qwen_replay")
    parser.add_argument("--dataset", type=str, default="datasets/BAD-ACTS.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []

    for environment in ENVIRONMENTS:
        eval_fn = EVAL_FNS[environment]
        print(f"\n=== {environment} ===")
        for safe in [False, True]:
            for cf_mode, root in [("baseline", args.baseline_root),
                                   ("replay", args.replay_root)]:
                rows = scan_results_root(root, environment, safe, cf_mode, eval_fn)
                label = f"safe={safe} mode={cf_mode}"
                if rows:
                    n_factual = sum(1 for r in rows if r["call_idx"] is None or not pd.isna(r["y_factual"]))
                    n_cf = sum(1 for r in rows if r["call_idx"] is not None and not pd.isna(r["y_cf"]))
                    print(f"  {label}: {n_factual} factual rows, {n_cf} CF rows")
                else:
                    print(f"  {label}: no data (placeholder)")
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ── Figure 1: ASR per (environment, safe) ────────────────────────────────
    # Use baseline rows, one y_factual per row_id
    asr_df = (
        df[df["cf_mode"] == "baseline"]
        .drop_duplicates(subset=["environment", "safe", "row_id"])
        .groupby(["environment", "safe"])
        .agg(
            asr=("y_factual", "mean"),
            n_rows=("row_id", "count"),
            asr_std=("y_factual", "std"),
        )
        .reset_index()
    )
    asr_path = os.path.join(args.output_dir, "asr.csv")
    asr_df.to_csv(asr_path, index=False)
    print(f"\nSaved ASR table → {asr_path}")
    print(asr_df.to_string(index=False))

    # ── Figure 2: ASE per (environment, safe, cf_mode) ───────────────────────
    cf_df = df[df["call_idx"].notna() & df["causal_effect"].notna()]
    ase_df = (
        cf_df.groupby(["environment", "safe", "cf_mode"])
        .agg(
            ase=("causal_effect", "mean"),
            ase_std=("causal_effect", "std"),
            n_pairs=("causal_effect", "count"),
        )
        .reset_index()
    )
    ase_path = os.path.join(args.output_dir, "ase.csv")
    ase_df.to_csv(ase_path, index=False)
    print(f"\nSaved ASE table → {ase_path}")
    print(ase_df.to_string(index=False))

    # ── Figure 3: Per-row causal effects (all modes) ─────────────────────────
    per_row_df = (
        cf_df.groupby(["environment", "safe", "cf_mode", "row_id"])
        .agg(
            y_factual=("y_factual", "first"),
            y_cf_mean=("y_cf", "mean"),
            causal_effect_mean=("causal_effect", "mean"),
            target_agent=("target_agent", "first"),
            cf_agent=("cf_agent", "first"),
            n_samples=("sample_idx", "count"),
        )
        .reset_index()
    )
    per_row_path = os.path.join(args.output_dir, "per_row.csv")
    per_row_df.to_csv(per_row_path, index=False)
    print(f"\nSaved per-row table → {per_row_path}")

    # ── Figure 4: Replay consistency check ───────────────────────────────────
    replay_df = cf_df[cf_df["cf_mode"] == "replay"][
        ["environment", "safe", "row_id", "call_idx", "y_factual", "y_cf", "causal_effect"]
    ].copy()
    replay_path = os.path.join(args.output_dir, "replay_check.csv")
    replay_df.to_csv(replay_path, index=False)
    print(f"Saved replay check → {replay_path}")

    # ── Figure 5: Harm category breakdown ────────────────────────────────────
    if os.path.exists(args.dataset):
        bad_acts = pd.read_csv(args.dataset)
        bad_acts["environment"] = bad_acts["Environment"].str.lower().str.replace(" ", "_")
        # create local 0-based row index per environment to match per_row row_id
        # (BAD-ACTS uses global indices but run_experiments.py resets per environment)
        bad_acts["row_id"] = bad_acts.groupby("environment").cumcount()
        bad_acts["row_id"] = bad_acts["row_id"].astype(int)

        per_row_merged = per_row_df.copy()
        per_row_merged["row_id"] = per_row_merged["row_id"].astype(int)
        harm_df = per_row_merged.merge(
            bad_acts[["row_id", "environment", "Category", "Sub-Category"]],
            on=["row_id", "environment"],
            how="left",
        )
        harm_path = os.path.join(args.output_dir, "harm_category.csv")
        harm_df.to_csv(harm_path, index=False)
        print(f"Saved harm category table → {harm_path}")
    else:
        print(f"[skip] dataset not found at {args.dataset}, skipping harm_category.csv")

    print(f"\nAll outputs written to {args.output_dir}")