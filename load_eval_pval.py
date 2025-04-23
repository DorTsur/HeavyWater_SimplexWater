import json
import os
import argparse
import numpy as np
import pdb
def loop_json(method, fresh=True):
    if method == "rg":
        bool_base_folder_0 = False
        deltas = [1, 2, 3, 4, 5.1]
        base_folder_template = "pred/llama2-7b-chat-4k_old_g0.5_d{:.1f}_temp1.0"
        pval_name = "p_val_teststats"
        file_name = "/z_score/finance_qa_0.5_{:.1f}_3.05_z.jsonl"
        if fresh:
            base_folder_template += "_fresh_"
    elif method == "lc":
        deltas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        base_folder_template = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_{}"
        bool_base_folder_0 = False
        pval_name = "p_val_teststats"
        file_name = "/z_score/finance_qa_0.0_0.0_3.05_z.jsonl"
        # base_folder_0 = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0"
        if fresh:
            base_folder_template += "_fresh_"
    elif method == "cc":
        bool_base_folder_0 = True
        base_folder_0 = "pred/llama2-7b-chat-4k_cc-k_g0.5_d5.0_temp1.0_k_2_d_tile_1.0"
        deltas = []
        if fresh:
            base_folder_0 += "_fresh_"
    elif method == "baseline":
        bool_base_folder_0 = True
        base_folder_0 = "pred/llama2-7b-chat-4k_old_g0.5_d0.0_temp1.0"
        deltas = []
        pval_name = "p_val_teststats"
        file_name = "/z_score/finance_qa_0.5_0.0_3.05_z.jsonl"
        if fresh:
            base_folder_0 += "_fresh_"
    # Dictionary to hold the results
    finance_qa_scores = {}
    finance_qa_std = {}

    # Loop over each delta and load the corresponding JSON file
    if bool_base_folder_0:
        base_folder_0 += file_name
        with open(base_folder_0, 'r') as f:
            data = json.load(f)
            p_val_list = data.get(pval_name, None)
            # to 4 decimal places
            finance_qa_scores[0.0] = round(np.median(p_val_list),4)
            finance_qa_std[0.0] = round(np.std(p_val_list),4)
    pdb.set_trace()
    for delta in deltas:
        path = base_folder_template.format(delta)
        temp_file_name = file_name.format(delta)
        path += temp_file_name
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                p_val_list = data.get(pval_name, None)
                # to 4 decimal places
                finance_qa_scores[delta] = round(np.median(p_val_list),4)
                finance_qa_std[delta] = round(np.std(p_val_list),4)
        except FileNotFoundError:
            print(f"Warning: File not found at {path}")
            finance_qa_scores[delta] = None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {path}")
            finance_qa_scores[delta] = None

    # Print results
    for delta, score in finance_qa_scores.items():
        print(f"Delta {delta}: finance_qa = {score}, std = {finance_qa_std.get(delta, 'N/A')}")
    print(finance_qa_scores.values())
    print(finance_qa_std.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and evaluate JSON files.")
    parser.add_argument("--method", type = str, choices=["rg", "lc", "cc", "baseline"], required=True, help="Method to use for evaluation.")
    parser.add_argument("--fresh", type =bool , default = 'True', help="Use fresh data files.")
    args = parser.parse_args()

    loop_json(args.method, args.fresh)