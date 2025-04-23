import json
import os
import argparse
def loop_json(method):
    if method == "rg":
        bool_base_folder_0 = False
        deltas = [1, 1.5, 2, 3, 4, 5, 6]
        base_folder_template = "pred/llama2-7b-chat-4k_old_g0.5_d{:.1f}_temp1.0_fresh_/eval/ce.json"
    elif method == "lc":
        deltas = [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
        base_folder_template = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_{}_fresh_/eval/ce.json"
        bool_base_folder_0 = True
        base_folder_0 = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_fresh_/eval/ce.json"
    elif method == "cc":
        bool_base_folder_0 = True
        base_folder_0 = "pred/llama2-7b-chat-4k_cc-k_g0.5_d5.0_temp1.0_k_2_d_tile_1.0_fresh_/eval/ce.json"
        deltas = []
    elif method == "baseline":
        bool_base_folder_0 = True
        base_folder_0 = "pred/llama2-7b-chat-4k_old_g0.5_d0.0_temp1.0_fresh_/eval/ce.json"
        deltas = []
    # Dictionary to hold the results
    finance_qa_scores = {}

    # Loop over each delta and load the corresponding JSON file
    if bool_base_folder_0:
        with open(base_folder_0, 'r') as f:
            data = json.load(f)
            # to 4 decimal places
            finance_qa_scores[0.0] = round(data.get("finance_qa", None),4)
    for delta in deltas:
        path = base_folder_template.format(delta)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # to 4 decimal places
                finance_qa_scores[delta] = round(data.get("finance_qa", None),4)
        except FileNotFoundError:
            print(f"Warning: File not found at {path}")
            finance_qa_scores[delta] = None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {path}")
            finance_qa_scores[delta] = None

    # Print results
    for delta, score in finance_qa_scores.items():
        print(f"Delta {delta}: finance_qa = {score}")
    print(finance_qa_scores.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and evaluate JSON files.")
    parser.add_argument("--method", type=str, choices=["rg", "lc", "cc", "baseline"], required=True, help="Method to use for evaluation.")
    args = parser.parse_args()

    loop_json(args.method)