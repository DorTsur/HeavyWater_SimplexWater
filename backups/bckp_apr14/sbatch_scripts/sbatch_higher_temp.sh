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
python pred.py --mode "old" --dataset "finance_qa" --delta 1.5 --gamma 0.5 --sampling_temp 1.0
python pred.py --mode "cc-k" --dataset "finance_qa" --delta 4.0 --gamma 0.5 --sampling_temp 1.0 --cc_k 2
python pred.py --mode "cc-combined" --dataset "finance_qa" --delta 2.0 --gamma 0.5 --sampling_temp 1.0 
python pred.py --mode "old" --dataset "finance_qa" --delta 5.0 --gamma 0.5 --sampling_temp 1.0
