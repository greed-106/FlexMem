# [CVPR 2026 Highlight] FlexMem: Scaling the Long Video Understanding of MLLMs via Visual Memory Mechanism 

[![arXiv](https://img.shields.io/badge/Arxiv-2603.29252-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2603.29252)

## 👣Introduction

This repository implements FlexMem, a novel and training-free visual memory mechanism for Multimodal Large language Models (MLLMs). FlexMem can help MLLMs continually watch video content and recall the most relevant memory fragments to answer the question.

![overview](images/overview.png)

### Key advantages:

- **Video Understanding of Infinite Lengths:** We study the long video understanding of MLLMs from the perspective of visual memory mechanism, and propose a novel approached termed FlexMem to scale up the input of video frames.
- **Outstanding Model Performance with Low Resource Requirement:** On a set of benchmarks, our FelxMem can greatly improve the capabilities of base MLLMs and outperform a set of SOTA methods using only one 3090 GPU.

## 🛠️ Usage

### Repository Layout
```
FlexmMem
├── flexmem/                  
│   ├── modeling_memory/     # Memory Storage and Retrieval 
│   ├── modeling_qwen2/      # Dual-Pathway Compression
│   └── stream_llava_qwen/   # FlexMem-patched LLaVa-video Inference Pipline 
├── llava/                   # Evaluation Process
└── scripts/                 # Assessment Tools
```
### Installation
```bash
uv sync
```

`uv sync` creates a Python 3.10 environment, obtains ordinary Python packages from the Tsinghua PyPI mirror, and obtains `torch==2.9.1+cu128` and `torchvision==0.24.1+cu128` from the official CUDA 12.8 PyTorch wheel index. The latter is necessary because CUDA-enabled PyTorch wheels are not published on PyPI. To run commands inside the environment, prefix them with `uv run`.

### lmms-eval

The repository installs two lmms-eval model plugins: `flexmem` for the standard FlexMem implementation and `flexmem_fast` for the repository's MLVU-oriented FlexMem-fast implementation. No changes to lmms-eval's own source tree are needed. Both preserve FlexMem's streaming video sampling and reset visual memory for every example. `batch_size` must remain `1`.

For LongVideoBench's video variant:

```bash
uv run lmms-eval \
  --model flexmem \
  --model_args pretrained=/path/to/LLaVA-Video-7B-Qwen2,batch_size=1 \
  --tasks longvideobench_val_v \
  --batch_size 1 \
  --output_path ./outputs/longvideobench
```

For MLVU:

```bash
uv run lmms-eval \
  --model flexmem_fast \
  --model_args pretrained=/path/to/LLaVA-Video-7B-Qwen2,batch_size=1 \
  --tasks mlvu_dev \
  --batch_size 1 \
  --output_path ./outputs/mlvu
```

lmms-eval downloads the benchmark data according to each task definition. Set `HF_HOME` before running if its dataset cache should live outside the default location. `flexmem` defaults to [`flexmem_lmms_eval/configs/flexmem.yaml`](flexmem_lmms_eval/configs/flexmem.yaml), while `flexmem_fast` uses the original `FlexMem-fast/config.yaml`; pass `config_path=/path/to/config.yaml` in `--model_args` to override either. `longvideobench_*_v` is the compatible LongVideoBench task family because FlexMem consumes each video as one visual stream.
### Long Video Benchmark Evaluation
For **LongVideoBench** evaluation, you can use the following script to evaluate.

First, download the **LongVideoBench** dataset and **LLaVA-Video-7B-Qwen2** model weights to your local machine, assume their root are **data_root** and **model_root**, and replace **CKPT** and **DATA_ROOT** in FlexMem/scripts/video/lvbench/lvbench_eval_stream.sh with your local **model_root** and **data_root**.  

Then, you can use the following script:  
```bash
bash FlexMem/scripts/video/lvbench/lvbench_eval_stream.sh
```
For **FlexMem-Fast** evaluation on **MLVU**.

First download the **MLVU** dataset and the **LLaVA-Video-7B-Qwen2** model weights to your local machine, assume their roots are **data_root** and **model_root**, and replace **CKPT** and **DATA_ROOT** in FlexMem-fast/scripts/video/lvbench/lvbench_eval_stream.sh with your local **model_root** and **data_root**. 

Then, you can use the following script:  
```bash
bash FlexMem-fast/scripts/video/mlvu/mlvu_eval_stream.sh
```

## 🤝 Acknowledgements

- **LLaVA-NeXT**: the codebase we used for evaluation.
- **Video-XL**: the codebase we built upon. 


## Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:

```BibTeX
@article{chen2026flexmem,
  title={Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism},
  author={Tao Chen and Kun Zhang and Qiong Wu and Xiao Chen and Chao Chang and Xiaoshuai Sun and Yiyi Zhou and Rongrong Ji},
  journal={arXiv preprint arXiv:2603.29252},
  year={2026}
}
```
