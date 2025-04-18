from pred import parse_args, seed_everything, load_model_and_tokenizer
import json
import torch
import os
from generate import Generator
from tqdm import tqdm
import pdb



if __name__ == '__main__':
    args = parse_args()
    if args.print:
            print(args)
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    dataset2level = json.load(open("config/dataset2level.json", "r"))
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    max_length = model2maxlen[model_name]


    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf", model_name, device)
    save_dir = f"simple_pred/{model_name}/{args.mode}_temp{args.sampling_temp}"
    if args.mode == 'old':
        save_dir += f'g{args.gamma}_d{args.delta}'
    if args.mode == 'cc-k':
        save_dir += f"_k_{args.cc_k}"
    if args.mode == 'lin_code' and args.tilt:
        save_dir += f"_d_tilt_{args.tilting_delta}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = args.dataset
    print(f"Testing on prompt {dataset}")
    data = []
    with open("data/WaterBench/{}_{}.jsonl".format(dataset2level[dataset], dataset), "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line)) 
    max_gen = 300
    prompt_format = dataset2prompt[dataset]
    data_path = os.path.join(save_dir, f"{dataset}")
    initial_seed = args.initial_seed
    # loop over several seeds:
    for s in range(args.num_seeds):
        # pdb.set_trace()
        """
        the routine of watermarking. Steps:
        1. calc seed
        2. loop over datasets. for each dataset:        
        a. load dataset
        b. sed seed
        c. set up generator
        d. create response
        e. collect responsequit()
        """
        args.initial_seed_llm = initial_seed + s
        seed_everything(args.initial_seed_llm)
        preds = []
        generator = Generator(args, tokenizer, model)
        torch.cuda.empty_cache()
        # loop over the dataset:
        for json_obj in tqdm(data[args.start_point:]):
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            completions_text, completions_tokens  = generator.generate(input_ids=input.input_ids, max_new_tokens=max_gen)
            pred = completions_text
            preds.append({"prompt":prompt, "pred": pred, "completions_tokens":completions_tokens})
        out_path = data_path + f"/_seed_{args.initial_seed}.jsonl"
        if os.path.exists(out_path):
            with open(out_path, "a", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')
        else:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for pred in preds:
                    json.dump(pred, f, ensure_ascii=False)
                    f.write('\n')




