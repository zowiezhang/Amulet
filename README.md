<div align="center">
    <img src="images/logo.png" alt="Amulet Logo" width="100%">
</div>

# Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2502.19148-b31b1b.svg)](https://arxiv.org/abs/2502.19148) [![Paper](https://img.shields.io/badge/OpenReview-Amulet-blue)](https://openreview.net/forum?id=f9w89OY2cp) [![Project Page](https://img.shields.io/badge/🔗-Project%20Page-blue)](https://zowiezhang.github.io/projects/Amulet/)

## Overview

![method](images/method.jpg)

The figure is intersected by an axis, with each node on the axis displaying a different distribution that shows the constantly changing user personalized preferences due to factors like time, value, need, and context, as illustrated by the part (a).

The part (b) shows that existing methods mostly consider aligning LLMs with general preferences from a static dataset, which may result in misalignment in dynamically personalized scenarios.

In the part (c), we have enlarged one of the preference nodes to show the processing of our Amulet framework. We formulate the decoding process of every token as a separate online learning problem and further adapt the backbone LLMs to align with the current user preference through a real-time optimization process with the guidance of user-provided prompts. The red token means the current processing token, which will be the condition for the next token prediction.

The core algorithm is implemented in `DecodingMethodsModels` folder. We also provide all the four baseline method (Base, Pref, Beam, LA) mentioned in our paper in the folder. We only test the code on Qwen, Llama, Mistral series of LLMs, it may run for other series, but we have not tested them.

## Setup

This code has been tested on Ubuntu 20.04 with Python 3.8 or above.

Clone the source code from GitHub:

```bash
git clone https://github.com/zowiezhang/Amulet.git
cd Amulet
```

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and setting up an environment:

```bash
conda create -y --name amulet python=3.8
conda activate amulet
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

This will automatically setup all dependencies.

## Datasets

We preprocessed the following four datasets as our benchmark, which are placed in the data folder.

```
data
  ├── HelpSteer_train.json
  ├── UltraFeedback_truthful_qa.json
  ├── UltraFeedback_ultrachat.json
  └── personal_preference_eval_preference_data.json
```

You can also use your own dataset by defining a json file of the following format:

```
{
    "index": 0,
    "question": "What is the best mobile phone brand currently?"
},
...
```

## Usage

```bash
export PYTHONPATH=$(pwd)
python main.py \
   --method amulet \
   --model_name meta-llama/Llama-3.1-8B-Instruct \
   --eval_data UltraFeedback_truthful_qa \
   --pref_name creative
```

The memory cost of Amuet is nearly the same as the inference. For 7B/8B model size or below, Amulet can run with only one 24G Nvidia GPU. This code is set to run on a single GPU by default, if you want to perform the code on multi-GPUs, please add `--multi_gpu` to the command.

Please refer to `config.py` for documentation on what each configuration does.

## Evaluation

We implement two types of evaluation, GPT win rate and RM score.

### GPT win rate

If you want to use our GPT win rate evaluation, you should first create and add your API_KEY in `openai_key_info.py`, then follow the demo:

```bash
python evals/gpt_evals.py \
   --pref_name creative \
   --model_name Llama-3.1-8B-Instruct \
   --eval_data UltraFeedback_truthful_qa
```

### RM score

We also provide an implemrntation of RM evals with [ArmoRM](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1), here is a quick start demo:

```bash
python evals/rm_evals.py \
   --method amulet \
   --pref_name creative \
   --model_name Llama-3.1-8B-Instruct \
   --eval_data UltraFeedback_truthful_qa
```

## BibTex

If you find our work useful, please consider citing:

```
@inproceedings{zhang2025amulet,
    title={Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of {LLM}s},
    author={Zhaowei Zhang and Fengshuo Bai and Qizhi Chen and Chengdong Ma and Mingzhi Wang and Haoran Sun and Zilong Zheng and Yaodong Yang},
    booktitle={The Thirteenth International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://openreview.net/forum?id=f9w89OY2cp}
}
```
