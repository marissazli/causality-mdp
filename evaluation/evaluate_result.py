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

    # ---- args for ASE / counterfactual mode ----
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
        default="id+target",
        help="How to group trajectories when computing expectations (ASE mode).",
    )
    # NEW: strict pairwise, seed-matched mode
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help=(
            "If set, compute counterfactual effects by pairing ref & int samples "
            "with the same (id, run_idx) and averaging Y_int - Y_ref. "
            "Overrides --group-by."
        ),
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
    #  MODE 2: ASE / counterfactual between ref & int runs
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

    # ---------------------------------------------------------
    # 2A. PAIRWISE MODE: (id, run_idx) matched counterfactuals
    # ---------------------------------------------------------
    if args.pairwise:
        # Build lookup dicts keyed by (id, run_idx)
        def key_pair(dp):
            return (dp["id"], dp.get("run_idx", None))

        ref_dict = {}
        for dp in ref_data:
            k = key_pair(dp)
            if k in ref_dict:
                # you can warn if there are duplicates, but we just keep the first
                pass
            ref_dict[k] = dp

        int_dict = {}
        for dp in int_data:
            k = key_pair(dp)
            if k in int_dict:
                pass
            int_dict[k] = dp

        common_keys = sorted(set(ref_dict.keys()) & set(int_dict.keys()), key=str)

        if not common_keys:
            print("No overlapping (id, run_idx) keys between reference and intervention runs.")
            exit(0)

        pair_rows = []
        deltas = []

        for k in common_keys:
            dp_ref = ref_dict[k]
            dp_int = int_dict[k]

            y_ref = float(bool(eval_fn(dp_ref)))
            y_int = float(bool(eval_fn(dp_int)))
            delta = y_int - y_ref

            deltas.append(delta)

            i, run_idx = k
            pair_rows.append(
                {
                    "id": i,
                    "run_idx": run_idx,
                    "seed_ref": dp_ref.get("seed", None),
                    "seed_int": dp_int.get("seed", None),
                    "target_agent_ref": dp_ref.get("target_agent", None),
                    "target_agent_int": dp_int.get("target_agent", None),
                    "Y_ref": y_ref,
                    "Y_int": y_int,
                    "delta": delta,
                }
            )

        deltas = np.array(deltas)
        est_effect = float(np.mean(deltas))

        print(f"Pairwise counterfactual effect (mean Y_int - Y_ref over pairs): {est_effect:.4f}")
        print(f"Number of matched pairs: {len(pair_rows)}")

        if args.res_path:
            df_pairs = pd.DataFrame(pair_rows)
            df_pairs.to_csv(args.res_path, index=False)
            print(f"Saved per-pair counterfactual effects to {args.res_path}")

        # also save raw per-sample Y if you want
        samples_path = (args.res_path or "counterfactual_pairs.csv").replace(".csv", "_samples.csv")
        all_samples = []
        for k in common_keys:
            dp_ref = ref_dict[k]
            dp_int = int_dict[k]
            y_ref = float(bool(eval_fn(dp_ref)))
            y_int = float(bool(eval_fn(dp_int)))
            all_samples.append(
                {
                    "which": "ref",
                    "id": dp_ref.get("id", None),
                    "run_idx": dp_ref.get("run_idx", None),
                    "seed": dp_ref.get("seed", None),
                    "target_agent": dp_ref.get("target_agent", None),
                    "Y": y_ref,
                }
            )
            all_samples.append(
                {
                    "which": "int",
                    "id": dp_int.get("id", None),
                    "run_idx": dp_int.get("run_idx", None),
                    "seed": dp_int.get("seed", None),
                    "target_agent": dp_int.get("target_agent", None),
                    "Y": y_int,
                }
            )
        df_samples = pd.DataFrame(all_samples)
        df_samples.to_csv(samples_path, index=False)
        print(f"Saved per-sample Y values to {samples_path}")

        exit(0)

    # ---------------------------------------------------------
    # 2B. ORIGINAL ASE MODE: group-by id or id+target (unchanged)
    # ---------------------------------------------------------
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
