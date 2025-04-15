#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-5:59
#SBATCH -p seas_gpu
#SBATCH --mem=32000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mail-type=ALL
ml python/3.10.13-fasrc01
ml cuda/11.8.0-fasrc01
mamba activate nenv
python pred.py --mode 'lin_code' --dataset 'finance_qa' --sampling_temp 0.7


