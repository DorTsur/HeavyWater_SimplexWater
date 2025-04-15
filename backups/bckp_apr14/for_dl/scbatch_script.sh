#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-5:59
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
ml python/3.10.13-fasrc01
ml cuda/11.8.0-fasrc01
mamba activate nenv
python pred.py --mode "old" --dataset "all"