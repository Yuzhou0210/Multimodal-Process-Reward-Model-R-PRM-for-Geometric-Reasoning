# Multimodal R-PRM

**Process-Reward Verification for Geometric Reasoning and Test-Time Compute Scaling**

Multimodal R-PRM is a multimodal process reward model for fine-grained geometric reasoning verification. It evaluates intermediate reasoning steps, extracts continuous Soft-Rewards, and reranks multiple candidate solutions at test time.

## Overview

For each reasoning step, R-PRM performs four-stage verification:

1. Visual Analysis
2. Now Step Analysis
3. Data Source Analysis
4. Calculation Analysis

The verifier ends with:

```text
Verification: Is the step correct? Yes/No
```

Instead of using only the binary output, we convert the `Yes/No` logits into a continuous Soft-Reward.

For step $t$,

$$
p_t =
P(\mathrm{Yes}\mid I,Q,S_{<t},s_t)
=
\frac{
\exp(z_{\mathrm{Yes}}-z_{\max})
}{
\exp(z_{\mathrm{Yes}}-z_{\max})
+
\exp(z_{\mathrm{No}}-z_{\max})
}.
$$

where

$$
z_{\max}=\max(z_{\mathrm{Yes}},z_{\mathrm{No}}).
$$

For a trajectory $\tau=\{s_1,\ldots,s_K\}$,

$$
\mathcal{S}(\tau)
=
\frac{1}{K}\sum_{t=1}^{K}p_t
+
\lambda \min_{1\leq t\leq K}(p_t),
\qquad
\lambda\rightarrow0^+.
$$

## Training Pipeline

```text
Geo170K
   ↓
Now-Step Construction
   ↓
Multimodal API Annotation
   ↓
Hard Filtering
   ↓
1,788 Seed Examples
   ↓
Stage-I R-PRM SFT
   ↓
Large-Scale Data Expansion
   ↓
13,935 Process-Supervision Examples
   ↓
Final R-PRM
```

Final training setup:

```text
Backbone: Qwen3-VL-8B-Instruct
Precision: bfloat16
Optimizer: AdamW
Learning rate: 2e-5
Scheduler: Cosine decay
Warmup: 3%
Global batch size: 32
Epochs: 2
```

Validation performance:

| Metric | Result |
|---|---:|
| Step Classification Accuracy | 93.37% |
| ROC-AUC | 0.961 |

## Test-Time Inference

The reported experiments use:

```text
Greedy Solution (T=0)
        +
N Sampled Trajectories (T=0.7)
        ↓
Candidate Pool
        ↓
R-PRM Soft-Reward Scoring
        ↓
Trajectory Reranking
        ↓
Final Answer
```

Current evaluation:

> **Trajectory-level Best-of-N + Greedy Guard**

Step-level PRM-guided beam search and MCTS are future extensions and are not used in the reported main results.

## Results

Evaluation on a 30-example MathVista geometry subset:

| Method | Accuracy |
|---|---:|
| Greedy Baseline | 50.00% |
| Random Pick @ N=8 | 60.00% |
| PRM Best-of-8 | 63.33% |
| **PRM Best-of-(8+1) + Guard** | **66.67%** |
| Oracle @ N=8 | 93.33% |
| Qwen3.8-Flash (reported) | 90.00% |

## Repository Structure

```text
multimodal-rprm/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── scripts/
│   └── data/
├── distill/
├── training/
├── inference/
├── evaluation/
├── demo/
├── scripts/
├── configs/
└── assets/
```

## Installation

```bash
git clone https://github.com/Yuzhou0210/Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning.git
cd Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning
pip install -r requirements.txt
```

## Main Scripts

Training:

```bash
bash training/run_stage2_sft.sh
```

Best-of-N reranking:

```bash
python inference/run_best_of_n_rerank.py
```

R-PRM inference:

```bash
python inference/test_rprm_infer.py
```

Gradio demo:

```bash
python demo/app_gradio.py
```

## Citation

```bibtex
@article{multimodalrprm2026,
  title  = {Multimodal R-PRM: Process-Reward Verification for Geometric Reasoning and Test-Time Compute Scaling},
  author = {Hu, Wenshuang and Ren, Xinyuan and Shi, Yuzhou},
  year   = {2026}
}
```

## License

Please follow the original licenses of Qwen3-VL, Geo170K, MathVista, and other third-party resources used in this project.
