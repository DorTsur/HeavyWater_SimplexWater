import os
from datasets import load_dataset
import torch, gc
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from generate import Generator
import pdb
import time


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama2-7b-chat-4k", choices=["llama2-7b-chat-4k", "chatglm2-6b-32k", "tulu-7b", "internlm-7b-8k", "llama3-1b", "llama3-8b", "llama3-3b","gemma2-7b","mistral-7b"])
    
    # watermark args
    parser.add_argument(
        "--mode",
        type=str,
        default="heavy_tail",
        choices=["heavy_tail", "synthid", "old", "cc", "cc-k","inv_tr","lin_code", "exponential","q_lin_code"],
        help="Which version of the watermark to generate",
    )
    parser.add_argument(
        '--initial_seed',
        type=int,
        default=1234,
        help=("The initial seed to use in the blacklist randomization process.", 
        "Is unused if the process is markov generally. Can be None."),
        )

    parser.add_argument(
        "--dynamic_seed",
        type=str,
        default="markov_1",
        choices=[None, "initial", "markov_1","fresh", "agg_hash"],
        help="The seeding procedure to use when sampling the redlist at each step.",
        )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5)

    parser.add_argument(
        "--delta",
        type=float,
        default=5.0)
    
    parser.add_argument(
        "--bl_type",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="The type of redlisting being performed.",
        )
    
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.0,
        help="The temperature to use when generating using multinom sampling",
        )

    parser.add_argument(
        "--threshold",
        type=float,
        default=4.0)

    parser.add_argument(
        "--test_min_tokens",
        type=int, 
        default=2)
    
    parser.add_argument(
        "--start_point",
        type=int,
        default=0,
    )

    parser.add_argument(
    "--print",
    type=int,
    default=1,
    )

    parser.add_argument(
    "--sinkhorn_reg",
    type=float,
    default=0.05,
    )

    parser.add_argument(
    "--sinkhorn_thresh",
    type=float,
    default=1e-5,
    )
    
    parser.add_argument( # for dataset
        "--dataset",
        type=str,
        default="all",
        # default="konwledge_understanding",
        choices=["konwledge_memorization","konwledge_understanding","longform_qa",
                        "finance_qa","hotpotqa","lcc", "multi_news", "qmsum","alpacafarm", "all","short_finance_qa","poem", "alpacafarm_qa"],
    )

    parser.add_argument(
        "--cc_k",
        type=int,
        default=2,
        help="The partition size of the cc watermark.",
    )

    parser.add_argument(
        "--tilt",
        type=str2bool,
        default=False,
        help="tilting option for distortionless distributions",
    )

    parser.add_argument(
        "--tilting_delta",
        type=float,
        default=0.5,
        help="tilting value for distortionless distributions",
    )

    parser.add_argument(
        "--num_seeds",
        type=int,
        default=2,
        help="tilting value for distortionless distributions",
    )

    parser.add_argument(
    "--initial_seed_llm",
    type=int,
    default=42)

    parser.add_argument(
    "--top_p",
    type=float,
    default=1.0)

    parser.add_argument(
    "--context",
    type=int,
    default=1)

    parser.add_argument(
        "--hashing_fn",
        type=str,
        default=None,
        choices=[None, "min", "sum","prod", "repeat"],
        help="The seeding procedure to use when sampling the redlist at each step.",
        )
    
    parser.add_argument(
        "--ht_dist",
        type=str,
        default="lognormal",
        choices=["lognormal","gamma","pareto"],
        help="heavy tail watermark score distribution.",
        )
    
    parser.add_argument(
    "--q",
    type=int,
    default=2)

    parser.add_argument(
        '--print_args', action='store_true', help="Print the parsed arguments.")

    parser.add_argument(
    "--heavywater_k",
    type=int,
    default=1024)
    
    return parser.parse_args(args)

    

    

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    prompt = f"[INST]{prompt}[/INST]"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(watermark_args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name, debug: bool = False):
    preds = []
    generator = Generator(watermark_args, tokenizer, model)
    torch.cuda.empty_cache()
    CE_ave_per_prompt = []
    times = []
    for json_obj in tqdm(data[watermark_args.start_point:]):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        start_time = time.perf_counter()
        completions_text, completions_tokens, CE_prompt  = generator.generate(input_ids=input.input_ids, max_new_tokens=max_gen)
        end_time = time.perf_counter()
        times.append(end_time-start_time)
        CE_ave_per_prompt.append(CE_prompt)
  
        if debug:
            print("####################")
            
        pred = completions_text
        pred = post_process(pred, model_name)
        preds.append({"prompt":prompt, "pred": pred, "completions_tokens":completions_tokens, "answers": json_obj["outputs"], "all_classes": json_obj["all_classes"], "length":json_obj["length"]})

    print('------')    
    print(f'average run time {np.mean(np.array(times))}')
    print('------')
    return preds,CE_ave_per_prompt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device,  load_token_only=False):
    # pdb.set_trace()
    if "chatglm" in model_name or 'mistral-7b' in model_name or "gemma2-7b" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if not load_token_only:
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                                  output_scores=True, return_dict_in_generate=True, 
                                                  torch_dtype=torch.float16).to(device)
            model.eval()
    elif "llama2" in model_name:
        # replace_llama_attn_with_flash_attn()
        print('importing llama2')
        tokenizer = LlamaTokenizer.from_pretrained(path)
        # tokenizer = AutoTokenizer.from_pretrained(path)
        if not load_token_only:
            model = LlamaForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, torch_dtype=torch.float16).to(device) 
    elif "llama3" in model_name:
        # replace_llama_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer = AutoTokenizer.from_pretrained(path)
        if not load_token_only:
            #model = LlamaForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, torch_dtype=torch.float16).to(device) 
            model = LlamaForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, torch_dtype=torch.float16, device_map = 'auto') 
            # model = AutoModelForCausalLM.from_pretrained(path, output_scores=True, return_dict_in_generate=True, torch_dtype=torch.float16).to(device) 

    print('FINISHED importing model')
    if load_token_only:
        return tokenizer
    else:
        model = model.eval()
        return model, tokenizer

if __name__ == '__main__':
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"
    # pdb.set_trace()
    args = parse_args()
    if args.print:
        print(args)
    seed_everything(args.initial_seed_llm)
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    
    print(f'finished parsing model, model path {model2path[args.model]}')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device= {device}")
    
    model_name = args.model
    
    if args.model == "llama2-7b-chat-4k":
        model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf", model_name, device)
    elif args.model == "llama3-1b":
        model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.2-1B-Instruct", model_name, device)
    elif args.model == "llama3-8b":
        model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.1-8B-Instruct", model_name, device)
    elif args.model == "llama3-3b":
        model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.2-3B-Instruct", model_name, device)
    elif args.model == "gemma2-7b":
        model, tokenizer = load_model_and_tokenizer("google/gemma-7b", model_name, device)
    elif args.model == "mistral-7b":
        model, tokenizer = load_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.3", model_name, device)
    print('finished loading model and tokenzier')
    # pdb.set_trace()
    max_length = model2maxlen[model_name]
    datasets = ["konwledge_memorization","konwledge_understanding","longform_qa",
                        "finance_qa","hotpotqa","lcc", "multi_news", "qmsum","alpacafarm"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    dataset2level = json.load(open("config/dataset2level.json", "r"))
    
    # make dir for saving predictions with parameters in its path:
    if not os.path.exists("pred"):
        os.makedirs("pred")
    save_dir = f"pred/{model_name}_{args.mode}_g{args.gamma}_d{args.delta}_temp{args.sampling_temp}"
    if args.mode == 'heavy_tail':
        save_dir = f"pred/{model_name}_{args.mode}_{args.ht_dist}_g{args.gamma}_d{args.delta}_temp{args.sampling_temp}"
    if args.mode == "old":
        print("old RG watermark")
    if args.bl_type == "hard":
        save_dir = f"pred/{model_name}_{args.mode}_g{args.gamma}_d{args.delta}_hard"
    if args.mode == "cc-k":
        save_dir += f"_k_{args.cc_k}"
    if args.mode in ('lin_code','q_lin_code', 'heavy_tail') and args.tilt:
        save_dir += f"_d_tile_{args.tilting_delta}"
    if args.mode == 'cc-k' and args.tilt:
        save_dir += f"_d_tile_{args.tilting_delta}"
    # pdb.set_trace()
    if args.sinkhorn_reg != 0.05:
        save_dir += f"_reg_{args.sinkhorn_reg}"
    if args.sinkhorn_thresh != 1e-5:
        save_dir += f"_reg_{args.sinkhorn_thresh}"
    # if args.dynamic_seed == 'fresh':
    #     save_dir += f"_fresh_"
    save_dir += f"_{args.dynamic_seed}"
    if args.hashing_fn is not None and args.context != 1:
        save_dir += f"_{args.hashing_fn}_context_{args.context}"
    if args.top_p != 1.0:
        save_dir += f"_top_p_{args.top_p}"
    if args.mode == "q_lin_code":
        save_dir += f"_q_{args.q}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # predict on each dataset
    if args.dataset == "all":
        for dataset in datasets:
            # load data
            print(f"{dataset} has began.........")
            data = []
            with open("data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))       
            out_path = os.path.join(save_dir, f"{dataset}.jsonl")
            prompt_format = dataset2prompt[dataset]
            max_gen = dataset2maxlen[dataset]
            # for token accumulation:
            preds, CE_ave_per_prompt = get_pred(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
            
            outpath_ce = os.path.join(save_dir, f"eval/{dataset}_CE.jsonl")
            if not os.path.exists(os.path.dirname(outpath_ce)):
                os.makedirs(os.path.dirname(outpath_ce), exist_ok=True)

            with open(outpath_ce, "w", encoding="utf-8") as f_ce:
                dictionary = {"CE_ave": np.mean(CE_ave_per_prompt), "CE_std": np.std(CE_ave_per_prompt), "CE list": CE_ave_per_prompt}
                json.dump(dictionary, f_ce, ensure_ascii=False)
            print(f"CE ave for {dataset} is {dictionary['CE_ave']}, CE std is {dictionary['CE_std']}")
            print(f"saving results in {outpath_ce}")


            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
                    
    else:
        dataset = args.dataset
        print(f"{dataset} has began.........")
        data = []
        with open("data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))       
        out_path = os.path.join(save_dir, f"{dataset}.jsonl")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds, CE_ave_per_prompt = get_pred(args, model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name)
        
        outpath_ce = os.path.join(save_dir, f"eval/{dataset}_CE.jsonl")
        if not os.path.exists(os.path.dirname(outpath_ce)):
            os.makedirs(os.path.dirname(outpath_ce), exist_ok=True)

        with open(outpath_ce, "w", encoding="utf-8") as f_ce:
            dictionary = {"CE_ave": np.mean(CE_ave_per_prompt), "CE_std": np.std(CE_ave_per_prompt)/np.sqrt(len(CE_ave_per_prompt)), "CE list": CE_ave_per_prompt}
            json.dump(dictionary, f_ce, ensure_ascii=False)
        print(f"CE ave for {dataset} is {dictionary['CE_ave']}, CE std is {dictionary['CE_std']}")
        print(f"saving results in {outpath_ce}")


        if os.path.exists(out_path):
            with open(out_path, "a", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')

