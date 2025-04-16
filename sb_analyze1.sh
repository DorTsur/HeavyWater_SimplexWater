#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-5:59
#SBATCH -p gpu_test
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
ml python/3.10.13-fasrc01
ml cuda/11.8.0-fasrc01
mamba activate nenv
python detect.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.1"
python eval.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.1"
python detect.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.3"
python eval.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.3"
python detect.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.4"
python eval.py --input_dir "pred/llama2-7b-chat-4k_lin_code_g0.5_d5.0_temp1.0_d_tile_0.4"
