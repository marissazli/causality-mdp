"""
evaluate_result.py

Walks a results directory produced by run_experiments.py (--compact-cf layout)
and computes:
  - Y_factual   : eval_fn on factual.json for each row
  - Y_cf        : eval_fn on each call_NNN_sample_MMM_full.json
  - causal_effect = Y_cf - Y_factual   (per row x call_idx)
  - ASE         : mean causal effect across all (row, call_idx) pairs

Usage:
    python evaluate_result.py \
        --results-dir results/env=travel_planning/model=Qwen3-8B/adv=PLANNER_AGENT \
        --environment travel_planning \
        [--res-path output.csv]
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pandas as pd

from evaluation_functions import (
    evaluate_travel_planning,
    evaluate_financial_article_writing,
    evaluate_code_generation,
    evaluate_MAD,
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def make_eval_datapoint(factual: dict, cf_full: dict) -> dict:
    """
    Merge factual metadata (target_agent, keywords, id, adversarial_agent, etc.)
    into the CF full datapoint so eval functions have everything they need.
    CF run data (team_states, sent_messages, tickets, files, trajectory) takes
    precedence since that's what we're evaluating.
    """
    merged = {}
    # start with factual metadata fields
    for key in ("id", "target_agent", "adversarial_agent", "target_action",
                "keywords", "backend", "environment", "task", "seed", "safe", "guardian"):
        if key in factual:
            merged[key] = factual[key]
    # overlay CF run output fields (these override if present)
    for key in ("team_states", "sent_messages", "tickets", "files",
                "trajectory", "call_log", "tape_status"):
        if key in cf_full:
            merged[key] = cf_full[key]
        elif key in factual.get("factual", {}):
            # factual.json nests run data under "factual" key
            merged[key] = factual["factual"][key]
    return merged


def make_factual_datapoint(factual: dict) -> dict:
    """Flatten factual.json so eval functions can access fields directly."""
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root results dir to walk, e.g. results/env=travel_planning/model=Qwen3-8B/adv=PLANNER_AGENT",
    )
    parser.add_argument(
        "--environment",
        type=str,
        required=True,
        choices=["travel_planning", "financial_article_writing", "code_generation", "multi_agent_debate"],
    )
    parser.add_argument(
        "--res-path",
        type=str,
        default=None,
        help="Optional path to save per-(row, call_idx) CSV.",
    )
    args = parser.parse_args()

    eval_fn = {
        "travel_planning": evaluate_travel_planning,
        "financial_article_writing": evaluate_financial_article_writing,
        "code_generation": lambda dp: evaluate_code_generation(dp, dp["keywords"]),
        "multi_agent_debate": evaluate_MAD,
    }[args.environment]

    results_root = Path(args.results_dir)
    row_dirs = sorted(results_root.glob("row=*"))

    if not row_dirs:
        raise FileNotFoundError(f"No row=* directories found under {results_root}")

    rows = []
    factual_ys = []

    for row_dir in row_dirs:
        factual_path = row_dir / "factual.json"
        if not factual_path.exists():
            print(f"  [skip] no factual.json in {row_dir}")
            continue

        factual = load_json(factual_path)
        row_id = factual.get("id", row_dir.name)

        factual_dp = make_factual_datapoint(factual)
        try:
            y_factual = float(bool(eval_fn(factual_dp)))
        except Exception as e:
            print(f"  [warn] eval failed on factual for row {row_id}: {e}")
            y_factual = float("nan")

        factual_ys.append(y_factual)

        cf_dir = row_dir / "cf"
        if not cf_dir.exists():
            print(f"  [skip] no cf/ dir in {row_dir}")
            continue

        cf_full_files = sorted(cf_dir.glob("call_*_sample_*_full.json"))

        # build a flat list of (call_idx, sample_idx, cf_full_dict, source_path)
        cf_entries = []

        if cf_full_files:
            for cf_path in cf_full_files:
                stem = cf_path.stem.replace("_full", "")
                parts = stem.split("_")
                try:
                    call_idx = int(parts[1])
                    sample_idx = int(parts[3])
                except (IndexError, ValueError):
                    print(f"  [warn] unexpected filename format: {cf_path.name}")
                    continue
                cf_entries.append((call_idx, sample_idx, load_json(cf_path), str(cf_path)))
        else:
            # fallback: read from index.json counterfactual_runs_by_call
            index_path = row_dir / "index.json"
            if not index_path.exists():
                print(f"  [skip] no _full.json files and no index.json in {row_dir}")
                continue
            index = load_json(index_path)
            cf_by_call = index.get("counterfactual_runs_by_call", [])
            if not cf_by_call:
                print(f"  [skip] no counterfactual_runs_by_call in {index_path}")
                continue
            for call_entry in cf_by_call:
                call_idx = int(call_entry["cf_call_idx"])
                for run in call_entry.get("runs", []):
                    sample_idx = int(run.get("sample_idx", 0))
                    cf_entries.append((call_idx, sample_idx, run, str(index_path)))

        if not cf_entries:
            print(f"  [skip] no CF data found for {row_dir}")
            continue

        for call_idx, sample_idx, cf_full, source in cf_entries:
            cf_dp = make_eval_datapoint(factual, cf_full)

            try:
                y_cf = float(bool(eval_fn(cf_dp)))
            except Exception as e:
                print(f"  [warn] eval failed on call {call_idx} sample {sample_idx} for row {row_id}: {e}")
                y_cf = float("nan")

            causal_effect = y_cf - y_factual

            rows.append({
                "row_id": row_id,
                "call_idx": call_idx,
                "sample_idx": sample_idx,
                "y_factual": y_factual,
                "y_cf": y_cf,
                "causal_effect": causal_effect,
                "target_agent": factual.get("target_agent", None),
                "cf_agent": cf_full.get("intervention", {}).get("cf_agent", None),
                "cf_path": source,
            })

    if not rows:
        print("No CF results found to evaluate.")
        raise SystemExit(1)

    df = pd.DataFrame(rows)

    # --- per (row_id, call_idx) summary (averaged over samples) ---
    grouped = (
        df.groupby(["row_id", "call_idx"])
        .agg(
            y_factual=("y_factual", "first"),
            y_cf_mean=("y_cf", "mean"),
            causal_effect_mean=("causal_effect", "mean"),
            n_samples=("sample_idx", "count"),
            target_agent=("target_agent", "first"),
            cf_agent=("cf_agent", "first"),
        )
        .reset_index()
    )

    # --- aggregate ASE ---
    valid = grouped["causal_effect_mean"].dropna()
    ase = float(np.mean(valid))
    asr_factual = float(np.nanmean(df.groupby("row_id")["y_factual"].first()))

    print(f"\nRows evaluated       : {df['row_id'].nunique()}")
    print(f"ASR (factual)        : {asr_factual:.4f}")
    print(f"(row, call_idx) pairs: {len(grouped)}")
    print(f"Aggregate ASE        : {ase:.4f}")
    print(f"\nPer-(row, call_idx) causal effects:")
    print(grouped[["row_id", "call_idx", "cf_agent", "y_factual", "y_cf_mean", "causal_effect_mean", "n_samples"]].to_string(index=False))

    if args.res_path:
        grouped.to_csv(args.res_path, index=False)
        raw_path = args.res_path.replace(".csv", "_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"\nSaved per-(row, call_idx) summary to {args.res_path}")
        print(f"Saved raw per-sample rows to {raw_path}")