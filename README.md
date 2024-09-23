## Private Finetuning for LLMs (LLM-PFT)

<p>
    <a href="https://www.python.org/downloads/">
            <img alt="Build" src="https://img.shields.io/badge/3.10-Python-blue">
    </a>
    <a href="https://pytorch.org">
            <img alt="Build" src="https://img.shields.io/badge/2.0-PyTorch-orange">
    </a>
    <a href="https://github.com/jyhong836/fast-differential-privacy">
            <img alt="Build" src="https://img.shields.io/badge/2.0-fastDP-orange">
    </a>
</p>

The codebase for LLM DP/scrubbing/undefended finetuning in [LLM-PBE](https://llm-pbe.github.io/home) [![arXiv](https://img.shields.io/badge/arXiv-2408.12787-green)](https://arxiv.org/abs/2408.12787).

This code is modified from the [code](https://github.com/microsoft/analysing_pii_leakage) of [pii-leakage](https://arxiv.org/abs/2302.00539).
This repository supports fine-tuning latest LLMs, Flair Named Entity Recognition (NER) models, and Private AI API (for scrubbing).
It allows fine-tuning (i) undefended, (ii) differentially-private and (iii) scrubbed language models on ECHR and Enron.

The repository differs [pii-leakage](https://github.com/microsoft/analysing_pii_leakage) in these ways:
1. We replace opacus with a customized version of [fast-dp](https://github.com/jyhong836/fast-differential-privacy) which is more memory efficient and is compatible with latest pytorch 2.0, cuda and distributed training (e.g., deepspeed).
2. We can support latest LLMs, e.g., LLAMA.
3. (WIP) We extend the scrubbing tool from Flair to Private AI.
4. We exclude PII analysis tools but focus on fine-tuning.


## Build & Run

We recommend setting up a conda environment for this project.
```shell
conda create -n llm-pft python=3.10 -y
conda activate llm-pft

pip install torch
# if running `nvcc -V` yields empty, do this:
conda install cuda -c nvidia -y
# pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

**Troubleshooting**:
* `FlashAttention only support Ampere and newer.`
  - update `transformer` to the latest
  - Add below if still not working.
```
import torch
# to disable flash attn?
torch.backends.cuda.enable_mem_efficient_sdp(False)
```
  - Remove fp16 and bf16 from both deepspeed and config if still see this.
* cannot find symbollink for model (typically due to flash attn)
  - install nvcc, e.g. by `conda install cuda -c nvidia -y`
* `RuntimeError: 'weight' must be 2-D` with llama2 and zero3. Try to update below packages to the machted versions:
  ```shell
  pip install transformers==4.29.0
  pip install pydantic==1.10
  pip install deepspeed~=0.8.3
  # if fast tokenization is used
  pip install tokenizers==0.13.3
  pip install fastDP@git+https://github.com/jyhong836/fast-differential-privacy.git  # for `zero grad DP stage3()` error.
  ```
* `PrivacyEngine_Distributed_stage_2_and_3.__init__.<local>.zero_grad_DP_stage3() got an unexpected keyword argument 'set_to_none'`. Install a fixed version of fast-dp by `pip install fastDP@git+https://github.com/jyhong836/fast-differential-privacy.git`.
* If you encounter the following error message when running the attack:
```
if self.pad_token_id is not None and self.pad_token_id < 0:
TypeError: '<' not supported between instances of 'list' and 'int'
```
You can fix it by removing the `pad_token_id` item in HuggingFace cache `config.json` (e.g., the path may be like `~/.cache/huggingface/hub/models--LLM-PBE--Llama3.1-8b-instruct-LLMPC-Red-Team/snapshots/xxx/config.json`) and run again.


## Usage

We explain the following functions. The scripts are in the ```./examples``` folder and
run configurations are in the ```./configs``` folder.
* **Fine-Tune**: Fine-tune a pre-trained LM on a dataset (optionally with DP or scrubbing).


## Fine-Tuning

We demonstrate how to fine-tune a ```LLaMA2``` model on the [ECHR](https://huggingface.co/datasets/ecthr_cases) dataset
(i) without defenses, (ii) with scrubbing and (iii) with differentially private training (ε=8).

**No Defense**
```shell
export CUDA_VISIBLE_DEVICES=2,3,4,5
deepspeed fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-undefended.yml
```

**With Scrubbing**

_Note_: All PII will be scrubbed from the dataset. Scrubbing is a one-time operation that requires tagging all PII in the dataset first
which can take many hours depending on your setup. We do not provide tagged datasets.
```shell
export CUDA_LAUNCH_BLOCKING=1
deepspeed fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-scrubbed.yml
```

**With DP (ε=8.0)**

_Note_: We use the [dp-transformers](https://github.com/microsoft/dp-transformers) wrapper around PyTorch's [opacus](https://github.com/pytorch/opacus) library.
```shell
# if device ID's are not 0,1,2,3, then do below
export CUDA_VISIBLE_DEVICES=2,3,4,5
deepspeed --num_gpus=4 fine_tune.py --config_path=configs/fine-tune/echr-llama2-7b-dp8.yml
```
NOTE: Don't directly run the script with `python fine_tune.py ...` which does not apply DP with ZERO.

## Datasets

The provided ECHR dataset wrapper already tags all PII in the dataset.
The PII tagging is done using the Flair NER modules and can take several hours depending on your setup, but is a one-time operation
that will be cached in subsequent runs.

## Citation

Please consider citing the following paper if you found our work useful.

```
@article{li2024llm,
  title={LLM-PBE: Assessing Data Privacy in Large Language Models},
  author={Li, Qinbin and Hong, Junyuan and Xie, Chulin and Tan, Jeffrey and Xin, Rachel and Hou, Junyi and Yin, Xavier and Wang, Zhun and Hendrycks, Dan and Wang, Zhangyang and others},
  journal={Proceedings of the VLDB Endowment},
  volume={17},
  number={11},
  pages={3201--3214},
  year={2024},
  publisher={VLDB Endowment}
}
@InProceedings{lukas2023analyzing,
  title      = {Analyzing Leakage of Personally Identifiable Information in Language Models},
  author     = {Lukas, Nils and Salem, Ahmed and Sim, Robert and Tople, Shruti and Wutschitz, Lukas and Zanella-B{\'e}guelin, Santiago},
  booktitle  = {2023 IEEE Symposium on Security and Privacy (SP)},
  year       = {2023},
  publisher  = {IEEE Computer Society},
  pages      = {346-363},
  doi        = {10.1109/SP46215.2023.00154}
}
```
