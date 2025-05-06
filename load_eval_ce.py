import json
import os
import argparse
def loop_json(method = None, dataset = 'finance_qa', deltas=None, folder_template = None, \
    attributes = ["CE_ave", "CE_std"], bool_base_folder_0 = False, base_folder_0 = None):
    # if deltas and folder_template:
    #     bool_base_folder_0 = False
    #     
    # else:
    #     if method == "rg":
    #         bool_base_folder_0 = False
    #         deltas = [1, 1.5, 2, 3, 4, 5, 6]
    #         base_folder_template = "pred/llama2-7b-chat-4k_old_g0.5_d{:.1f}_temp1.0_fresh_/eval/ce.json"
    #     elif method == "lc" or "heavy_tail":
    #         deltas = [0.2, 0.3, 0.5, 0.7]
    #         #deltas = [0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
    #         base_folder_template = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_{}_fresh_/eval/ce.json"
    #         bool_base_folder_0 = True
    #         base_folder_0 = "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_fresh_/eval/ce.json"
    #     elif method == "cc":
    #         bool_base_folder_0 = True
    #         base_folder_0 = "pred/llama2-7b-chat-4k_cc-k_g0.5_d5.0_temp1.0_k_2_d_tile_1.0_fresh_/eval/ce.json"
    #         deltas = []
    #     elif method == "baseline":
    #         bool_base_folder_0 = True
    #         base_folder_0 = "pred/llama2-7b-chat-4k_old_g0.5_d0.0_temp1.0_fresh_/eval/ce.json"
    #         deltas = []

    base_folder_template = folder_template
    # Dictionary to hold the results
    result_scores = {}
    result_std = {}
    # Loop over each delta and load the corresponding JSON file
    if bool_base_folder_0:
        with open(base_folder_0, 'r') as f:
            data = json.load(f)
            # to 4 decimal places
            result_scores[0.0] = round(data.get(attributes[0], None),4)
            result_std[0.0] = round(data.get(attributes[1], None),4)
    for delta in deltas:
        path = base_folder_template.format(delta)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # to 4 decimal places
                result_scores[delta] = round(data.get(attributes[0], None),4)
                result_std[delta] = round(data.get(attributes[1], None),4)
        except FileNotFoundError:
            print(f"Warning: File not found at {path}")
            result_scores[delta] = None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {path}")
            result_scores[delta] = None

    # Print results
    print("deltas = ", list(result_scores.keys()))
    print(f"{attributes[0]} = ", list(result_scores.values()))
    print(f"{attributes[1]} = ", list(result_std.values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and evaluate JSON files.")
    parser.add_argument("--method", type=str, choices=["rg", "lc", "cc", "baseline", "heavy_tail"], required=True, help="Method to use for evaluation.")
    parser.add_argument("--dataset", type=str, default="finance_qa", help="Dataset to evaluate.")
    parser.add_argument("--get_CE", action="store_true", help="Get CE values or z-scores.")
    args = parser.parse_args()

    deltas = [0.2, 0.3, 0.5, 0.7]
    folder_template = "pred/llama2-7b-chat-4k_heavy_tail_g0.5_d5.0_temp1.0_d_tile_{:.1f}_fresh_top_p_0.999" 
    # CE
    if args.get_CE:
        folder_template += f"/eval/{args.dataset}_CE.jsonl" 
        attrs = ["CE_ave", "CE_std"]
        bool_base_folder_0 = True
        base_folder_0 = f"pred/llama2-7b-chat-4k_heavy_tail_g0.5_d5.0_temp1.0_fresh_top_p_0.999/eval/{args.dataset}_CE.jsonl"
    else:
    # z_score
        folder_template += f"/z_score/{args.dataset}_0.0_0.0_3.05_z.jsonl"
        attrs = ["agg_pvals_avg", "agg_pvals_stder"]
        bool_base_folder_0 = True
        base_folder_0 = f"pred/llama2-7b-chat-4k_heavy_tail_g0.5_d5.0_temp1.0_fresh_top_p_0.999/z_score/{args.dataset}_0.0_0.0_3.05_z.jsonl"
    loop_json(method = args.method, dataset = args.dataset, deltas = deltas, \
        folder_template = folder_template, attributes = attrs, bool_base_folder_0 = bool_base_folder_0, \
        base_folder_0 = base_folder_0)