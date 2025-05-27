# Watermark Detection for LLM Outputs

<img src="images/readme_image.png" width="300" style="vertical-align: top right;"/>

This is the official code repository for the paper **HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions**
The code is based on the pipeline in the [WaterBench paper](https://arxiv.org/abs/2311.07138).

---

## Features

- **Supports Multiple Watermarking Methods:**
  - SimplexWater - Based on Coding Theory.
  - HeavyWater - Based on Heavy tail score distributions.
  - [Red/Green Watermark]([https://example.com](https://proceedings.mlr.press/v202/kirchenbauer23a.html))
  - Gumbel Watermark
  - [Inverse Transform Watermark](https://arxiv.org/abs/2307.15593)
  - [SynthID (Tournament Smapling)](https://www.nature.com/articles/s41586-024-08025-4)
  - [Correlated Channel Watermark](https://arxiv.org/abs/2505.08878)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DorTsur/CC_WM_LLMs.git
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The watermarking pipeline is divided into 4 main components:

### Watermarking
Applying a watermark to an LLM generated text. Main script is pred.py
See the parameter choice in pred.py for the full parameter shoice.

Example of parameters:
dataset - choice of dataset to watermark
mode - determines which watermarking method is used
dynamic_seed - the seeding scheme used to generate the watermark
model - which LLM generated the text
top-p - which top-p is used (top-p=1.0 is the default)
sampling_temp - which temprature to use in the sampling procedure
tilt - tilting option for SimplexWater and HeavyWater
tilting_delta - tilting parameter value (only for HeavyWater and SimplexWater)

Example code:

```bash
python pred.py --mode 'lin_code' --dataset 'finance_qa' --sampling_temp 0.7 --tilt True --tilting_delta 0.0 --dynamic_seed 'markov_1' --top_p 0.999
```

### Detection

Detecting a watermark. Only needs the input_dir (default is in pred/...) No need to provide specific information, the exact setting is parsed from the path.

Example code:

```bash
python detect.py --input_dir <path-to-input-dir>
```

### Generation Metric Evaluation
Evalution of generation metric based on the WaterBench metrics. Requires on input_dir path.

Example code:

```bash
python eval.py --input_dir <path-to-input-dir>
```

### Text Attack

For robustness experiment. Applying a textual attack to the watermarked text. Based on the experiments from [MarkMyWords Benchmark](https://ieeexplore.ieee.org/abstract/document/10992530?casa_token=xX6MQibbApQAAAAA:NS0YTLxOx9aQ_AT9EhjVPOpbV3wgRiCgqhGjV8B73U1vpDfHScNsQbiS2w5_jBbQdrHb14jX)
Require speciyfying which attack is to be performed.

Example code:

```bash
python attack.py --input_dir <path-to-input-dir> --attack 'LowercaseAttack'
```
---

## Extending

- **Add new watermarking methods:**  
  Adding a new watermark is very simple:
  1. Implement your watermark as a new **LogitProcessor**
  2. Implement a detector.
  The new watermark script should be saved in the `watermark/` directory, and you should add a reference to your modules in `generate.py` and `detect.py`
- **Change statistical tests:**  
  Modify the relevant section in the main loop to use your preferred test or metric.

---

## Input Format

- The script expects input directories containing `.jsonl` files.
- Each line in a `.jsonl` file should be a JSON object with at least:
  - `"prompt"`: The input prompt to the LLM.
  - `"pred"`: The generated output from the LLM.

---


## Citation

TBD.


---

## Contact

For questions or contributions, don't hesitate to contact me privately.
