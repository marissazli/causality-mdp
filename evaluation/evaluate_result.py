from argparse import ArgumentParser
from evaluation_functions import *
import json
import pandas as pd

if __name__=="__main__":
    args = ArgumentParser()

    args.add_argument("path", type=str)
    args.add_argument("environment", type=str, choices=["travel_planning", "financial_article_writing", "code_generation", "multi_agent_debate"])
    args.add_argument("--res-path", type=str)

    args = args.parse_args()

    # load data
    with open(args.path) as f:
        data = json.load(f)

    # select correct evaluation function
    eval_fn = {
        "travel_planner" : evaluate_travel_planning,
        "financial_article_writing" : evaluate_financial_article_writing,
        "code_generation" : evaluate_code_generation,
        "multi_agent_debate" : evaluate_MAD
    }[args.environment]

    success = []
    for datapoint in data:
        success.append(eval_fn(datapoint))
    
    print(f"Attack Success Rate: {sum(success)/len(success)}")

    # Store success of all actions in a csv
    if args.res_path:
        results = pd.read_csv(f"datasets/BAD-ACTS.csv")
        results = results[results["Environment"] == args.environment]
        results["Success"] = success
        results.to_csv(args.res_path)