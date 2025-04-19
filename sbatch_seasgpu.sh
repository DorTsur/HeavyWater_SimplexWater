#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-8:59
#SBATCH -p seas_gpu
#SBATCH --mem=32000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mail-type=ALL
ml python/3.10.13-fasrc01
ml cuda/11.8.0-fasrc01
mamba activate nenv
# RG:
python simple_pred.py --mode 'old' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --delta 0.0
python simple_pred.py --mode 'old' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --delta 1.0
python simple_pred.py --mode 'old' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --delta 2.0
python simple_pred.py --mode 'old' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --delta 3.0
python simple_pred.py --mode 'old' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --delta 4.0
# CC: 
python simple_pred.py --mode 'cc-k' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --cc_k 2
# LC:
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.0
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.1
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.2
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.3
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.5
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.7
python simple_pred.py --mode 'lin_code' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7 --tilt True --tilting_delta 0.9
# INV:
python simple_pred.py --mode 'inv_tr' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7
# EXP:
python simple_pred.py --mode 'exponential' --dataset 'poem' --num_seeds 10 --sampling_temp 0.7