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
        default="id+target",  # default to finer grouping for multi-agent settings
        help="How to group trajectories when computing expectations.",
    )

    args = parser.parse_args()

    # select correct evaluation function
    eval_fn = {
        "travel_planning": evaluate_travel_planning,
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

    # collect per-sample values for each group
    ref_vals = defaultdict(list)
    int_vals = defaultdict(list)

    ref_rows = []
    int_rows = []

    # reference runs
    for idx, dp in enumerate(ref_data):
        key = make_key(dp)
        y = float(bool(eval_fn(dp)))
        ref_vals[key].append(y)

        ref_rows.append(
            {
                "which": "ref",
                "key": key,
                "sample_idx": idx,
                "Y": y,
                "id": dp.get("id", None),
                "target_agent": dp.get("target_agent", None),
                "run_idx": dp.get("run_idx", None),
            }
        )

    # intervention (corrupted) runs
    for idx, dp in enumerate(int_data):
        key = make_key(dp)
        y = float(bool(eval_fn(dp)))
        int_vals[key].append(y)

        int_rows.append(
            {
                "which": "int",
                "key": key,
                "sample_idx": idx,
                "Y": y,
                "id": dp.get("id", None),
                "target_agent": dp.get("target_agent", None),
                "run_idx": dp.get("run_idx", None),
            }
        )

    # intersect keys present in both ref and int
    keys_ref = set(ref_vals.keys())
    keys_int = set(int_vals.keys())
    common_keys = sorted(list(keys_ref & keys_int), key=str)

    if not common_keys:
        print("No overlapping groups between reference and intervention runs (keys differ).")
        exit(0)

    # compute per-group means
    ref_mean = {k: float(np.mean(v)) for k, v in ref_vals.items()}
    int_mean = {k: float(np.mean(v)) for k, v in int_vals.items()}

    # per-group ASE and overall ASE
    group_rows = []
    ase_values = []
    any_difference = False

    for k in common_keys:
        r = ref_mean.get(k, np.nan)
        t = int_mean.get(k, np.nan)
        ase = t - r
        ase_values.append(ase)

        # detect if there is any difference (for your debugging message)
        if not np.isclose(r, t):
            any_difference = True

        if isinstance(k, tuple) and len(k) == 2:
            kid, tgt = k
        else:
            kid, tgt = k, None

        group_rows.append(
            {
                "key": k,
                "id": kid,
                "target_agent": tgt,
                "ref_mean_Y": r,
                "int_mean_Y": t,
                "ase": ase,
                "n_ref_samples": len(ref_vals[k]),
                "n_int_samples": len(int_vals[k]),
            }
        )

    ase_values = np.array(ase_values)
    est_ase = float(np.mean(ase_values))

    # print debug info about differences
    if not any_difference:
        print("No differences found between ref + int Y values (at this granularity).")
    else:
        print("At least one group has different ref vs int expectations.")

    print(f"Estimated ASE (averaged over groups): {est_ase:.4f}")

    # save CSVs if requested
    if args.res_path:
        df_groups = pd.DataFrame(group_rows)
        df_groups.to_csv(args.res_path, index=False)
        print(f"Saved per-group ASE to {args.res_path}")

        df_samples = pd.DataFrame(ref_rows + int_rows)
        samples_path = args.res_path.replace(".csv", "_samples.csv")
        df_samples.to_csv(samples_path, index=False)
        print(f"Saved per-sample Y values to {samples_path}")
