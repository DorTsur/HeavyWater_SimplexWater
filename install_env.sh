ml python/3.10.13-fasrc01
ml cuda/11.8.0-fasrc01


mamba create --name nenv python=3.11
mamba activate nenv

pip install torch==2.0.1
pip install flash-attn==2.3.3
pip install alpaca_farm==0.1.9
pip install cpm_kernels==1.0.11
pip install -r requirements.txt

