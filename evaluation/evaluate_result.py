from argparse import ArgumentParser
from collections import defaultdict
import json
import numpy as np
import pandas as pd

from evaluation_functions import (
    evaluate_travel_planning,
    evaluate_financial_article_writing,
    evaluate_code_generation,
    evaluate_MAD,
)

if __name__ == "__main__":
    parser = ArgumentParser()

    # ---- existing args ----
    # For backward compatibility, keep a simple "single file" mode
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        help="Path to a single results JSON file (legacy ASR mode).",
    )
    parser.add_argument(
        "environment",
        type=str,
        choices=["travel_planning", "financial_article_writing", "code_generation", "multi_agent_debate"],
    )
    parser.add_argument("--res-path", type=str)

    # ---- new args for ASE ----
    parser.add_argument(
        "--ref-paths",
        type=str,
        nargs="+",
        help="One or more JSON files from the REFERENCE system (no intervention).",
    )
    parser.add_argument(
        "--int-paths",
        type=str,
        nargs="+",
        help="One or more JSON files from the INTERVENTION system (do(A_i,t = a_i,t)).",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["id", "id+target"],
        default="id",
        help="How to group trajectories when computing expectations.",
    )

    args = parser.parse_args()

    # select correct evaluation function
    eval_fn = {
        "travel_planner": evaluate_travel_planning,
        "financial_article_writing": evaluate_financial_article_writing,
        "code_generation": lambda dp: evaluate_code_generation(dp, dp["keywords"]),
        "multi_agent_debate": evaluate_MAD,
    }[args.environment]

    # -------------------------------
    #  MODE 1: legacy ASR on one file
    # -------------------------------
    if args.ref_paths is None and args.int_paths is None:
        if args.path is None:
            raise ValueError("Either provide `path` (legacy mode) or `--ref-paths/--int-paths` (ASE mode).")

        with open(args.path) as f:
            data = json.load(f)

        success = [bool(eval_fn(dp)) for dp in data]
        asr = sum(success) / len(success)
        print(f"Attack Success Rate: {asr:.4f}")

        if args.res_path:
            results = pd.read_csv("datasets/BAD-ACTS.csv")
            results = results[results["Environment"] == args.environment]
            results["Success"] = success
            results.to_csv(args.res_path, index=False)

        exit(0)

    # -------------------------------------
    #  MODE 2: ASE between ref & int runs
    # -------------------------------------

    if args.ref_paths is None or args.int_paths is None:
        raise ValueError("In ASE mode you must provide BOTH --ref-paths and --int-paths.")

    def load_many(paths):
        all_dp = []
        for p in paths:
            with open(p) as f:
                all_dp.extend(json.load(f))
        return all_dp

    ref_data = load_many(args.ref_paths)
    int_data = load_many(args.int_paths)

    # helper: key used to group trajectories that share the same harmful action
    def make_key(dp):
        if args.group_by == "id":
            return dp["id"]
        elif args.group_by == "id+target":
            return (dp["id"], dp["target_agent"].strip())

    # NEW: keep both means and the raw values + per-sample rows
    from itertools import count

    def compute_group_stats(data, label):
        """
        label: 'ref' or 'int'
        returns:
          means: dict[key] -> mean Y
          values: dict[key] -> list of Y
          rows: list[dict]  per-sample for CSV/debug
        """
        grouped = defaultdict(list)
        rows = []
        # simple counter so you can distinguish multiple runs for same id
        idx_counter = defaultdict(count)

        for dp in data:
            key = make_key(dp)
            y = float(bool(eval_fn(dp)))  # 1 if harmful action happened, else 0
            grouped[key].append(y)

            run_idx = next(idx_counter[key])
            rows.append(
                {
                    "group_key": key if not isinstance(key, tuple) else str(key),
                    "id": dp["id"],
                    "target_agent": dp.get("target_agent"),
                    "adversarial_agent": dp.get("adversarial_agent"),
                    "target_action": dp.get("target_action"),
                    "keywords": dp.get("keywords"),
                    "run_type": label,          # 'ref' or 'int'
                    "run_idx_in_group": run_idx,
                    "Y": y,
                }
            )

        means = {k: float(np.mean(v)) for k, v in grouped.items()}
        return means, grouped, rows

    ref_mean, ref_vals, ref_rows = compute_group_stats(ref_data, label="ref")
    int_mean, int_vals, int_rows = compute_group_stats(int_data, label="int")

    # align keys and compute ASE^N = E_int[Y] - E_ref[Y]
    keys = sorted(set(ref_mean.keys()) & set(int_mean.keys()))
    per_key_ase = {k: int_mean[k] - ref_mean[k] for k in keys}

    # quick console check: is there at least one key where outputs differ?
    differing_keys = []
    for k in keys:
        # compare means first (most important)
        if ref_mean[k] != int_mean[k]:
            differing_keys.append(k)
            continue
        # optional: also compare raw lists (in case counts differ)
        if ref_vals[k] != int_vals[k]:
            differing_keys.append(k)

    if differing_keys:
        print("Found keys where ref/int outputs differ.")
        # print a few examples
        for k in differing_keys[:5]:
            print("Key:", k)
            print("  ref values:", ref_vals[k])
            print("  int values:", int_vals[k])
    else:
        print("No differences found between ref + int Y values (at this granularity).")

    # overall ASE (averaged over all harmful behaviors)
    overall_ase = np.mean(list(per_key_ase.values()))
    print(f"Estimated ASE (averaged over groups): {overall_ase:.4f}")

    # optional: dump per-id ASE and per-sample Y's to CSV for later analysis
    if args.res_path:
        # group-level summary
        group_rows = []
        for k in keys:
            if isinstance(k, tuple):
                id_, target_agent = k
            else:
                id_, target_agent = k, None
            group_rows.append(
                {
                    "id": id_,
                    "target_agent": target_agent,
                    "E_Y_ref": ref_mean[k],
                    "E_Y_int": int_mean[k],
                    "ASE": per_key_ase[k],
                }
            )
        df_groups = pd.DataFrame(group_rows)
        df_groups.to_csv(args.res_path, index=False)
        print(f"Saved per-group ASE to {args.res_path}")

        # NEW: per-sample values file
        sample_path = args.res_path.replace(".csv", "_samples.csv")
        df_samples = pd.DataFrame(ref_rows + int_rows)
        df_samples.to_csv(sample_path, index=False)
        print(f"Saved per-sample Y values to {sample_path}")
