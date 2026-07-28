"""
length_confound.py

Computes correlation between message-length difference (factual vs CF baseline)
and counterfactual effect (CFE) magnitude, to address reviewer concern about
length as a confound.

The CF baseline text is always the fixed neutral deferral:
"I have no additional recommendation. Please continue."
which is ~10 tokens / ~52 characters.

For each row we extract:
  - len_factual: character length of the CF agent's message at call k*
  - len_cf: character length of the neutral baseline text (constant)
  - len_diff: |len_factual - len_cf|
  - cfe: causal_effect_mean from per_row.csv

Then compute Pearson and Spearman correlations between len_diff and |cfe|.

Usage:
    python length_confound.py \
        --baseline-root results_v2_qwen_baseline \
        --per-row outputs/per_row.csv \
        --output-dir outputs/
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# The fixed neutral baseline text used in all CF runs
CF_BASELINE_TEXT = "I have no additional recommendation. Please continue."
LEN_CF = len(CF_BASELINE_TEXT)

ADV_AGENTS = {
    "travel_planning": "PLANNER_AGENT",
    "financial_article_writing": "CHIEF_EDITOR",
    "code_generation": "CEO",
    "multi_agent_debate": "agent_4",
}

def safe_model_name(model_id="Qwen/Qwen3-8B"):
    return model_id.split("/")[-1]


def get_factual_message_length(baseline_root, environment, safe, row_id):
    """
    Extract the character length of the CF agent's factual message at k*
    directly from the intervention dict stored in index.json under
    'factual_response_text'.
    """
    model_name = safe_model_name()
    adv = ADV_AGENTS[environment]
    safe_suffix = "/safe=True" if safe else ""
    row_dir = (Path(baseline_root) / f"env={environment}" / f"model={model_name}"
               / f"adv={adv}{safe_suffix}" / f"row={row_id}")

    index_path = row_dir / "index.json"
    if not index_path.exists():
        return None

    index = json.load(open(index_path))
    cf_runs = index.get("counterfactual_runs", [])
    if not cf_runs:
        return None

    intervention = cf_runs[0].get("intervention", {})

    # factual_response_text is stored directly in the intervention dict
    factual_text = intervention.get("factual_response_text", "")
    if factual_text:
        return len(factual_text)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="results_v2_qwen_baseline")
    parser.add_argument("--per-row", default="outputs/per_row.csv")
    parser.add_argument("--output-dir", default="outputs/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    per_row = pd.read_csv(args.per_row)
    per_row["safe"] = per_row["safe"].astype(str).str.lower().isin(["true", "1"])

    # only baseline rows with valid CFE
    baseline = per_row[
        (per_row["cf_mode"] == "baseline") &
        per_row["causal_effect_mean"].notna()
    ].copy()

    print(f"Total baseline rows with CFE: {len(baseline)}")

    # extract factual message lengths
    lengths = []
    for _, row in baseline.iterrows():
        length = get_factual_message_length(
            args.baseline_root,
            row["environment"],
            row["safe"],
            int(row["row_id"])
        )
        lengths.append(length)

    baseline["len_factual"] = lengths
    baseline["len_cf"] = LEN_CF
    baseline["len_diff"] = (baseline["len_factual"] - LEN_CF).abs()
    baseline["cfe_abs"] = baseline["causal_effect_mean"].abs()

    # drop rows where length extraction failed
    valid = baseline.dropna(subset=["len_factual"])
    n_missing = len(baseline) - len(valid)
    if n_missing > 0:
        print(f"[warn] could not extract factual message length for {n_missing} rows")

    print(f"\nRows with valid lengths: {len(valid)}")
    print(f"Mean factual message length: {valid['len_factual'].mean():.1f} chars")
    print(f"Fixed CF baseline length:    {LEN_CF} chars")
    print(f"Mean |len_diff|:             {valid['len_diff'].mean():.1f} chars")

    # ── Correlation analysis ───────────────────────────────────────────────────
    print("\n── Correlation: |len_diff| vs |CFE| ──")
    results = []
    for env in sorted(valid["environment"].unique()):
        for safe in [False, True]:
            sub = valid[(valid["environment"] == env) & (valid["safe"] == safe)]
            if len(sub) < 5:
                continue
            pearson_r, pearson_p = stats.pearsonr(sub["len_diff"], sub["cfe_abs"])
            spearman_r, spearman_p = stats.spearmanr(sub["len_diff"], sub["cfe_abs"])
            results.append({
                "environment": env,
                "safe": safe,
                "n": len(sub),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            })
            print(f"  {env} safe={safe}: n={len(sub)}, "
                  f"Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), "
                  f"Spearman r={spearman_r:.3f} (p={spearman_p:.3f})")

    # overall
    pearson_r, pearson_p = stats.pearsonr(valid["len_diff"], valid["cfe_abs"])
    spearman_r, spearman_p = stats.spearmanr(valid["len_diff"], valid["cfe_abs"])
    print(f"\n  Overall: n={len(valid)}, "
          f"Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), "
          f"Spearman r={spearman_r:.3f} (p={spearman_p:.3f})")

    results.append({
        "environment": "ALL",
        "safe": None,
        "n": len(valid),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    })

    # save
    corr_df = pd.DataFrame(results)
    corr_path = os.path.join(args.output_dir, "length_confound.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\nSaved → {corr_path}")

    detail_path = os.path.join(args.output_dir, "length_confound_detail.csv")
    valid[["environment", "safe", "row_id", "len_factual", "len_diff",
           "causal_effect_mean", "cfe_abs"]].to_csv(detail_path, index=False)
    print(f"Saved → {detail_path}")