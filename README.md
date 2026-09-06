# Multimodal R-PRM

**Process-Reward Verification for Geometric Reasoning and Test-Time Compute Scaling**

Multimodal R-PRM is a process reward model designed for fine-grained verification of multimodal geometric reasoning. Instead of judging only the final answer, R-PRM evaluates intermediate reasoning steps through structured visual, logical, data-source, and numerical verification.

The framework combines:

- **Step-level multimodal process verification**
- **Four-stage structured reasoning diagnostics**
- **Continuous Soft-Reward extraction from verifier logits**
- **Hard-negative process-supervision construction**
- **Trajectory-level Best-of-\(N\) reranking**
- **Greedy Guard for robust test-time scaling**
- **Interactive Gradio diagnostics**

> **Current evaluation:** trajectory-level Best-of-\(N\) reranking + Greedy Guard.  
> **Future extension:** step-level PRM-guided beam search / MCTS.

---

## Overview

Given a geometric image \(I\), question \(Q\), reasoning history \(S_{<t}\), and current reasoning step \(s_t\), R-PRM performs four-stage verification:

1. **Visual Analysis** — verifies geometric primitives, annotations, and spatial relations.
2. **Now Step Analysis** — checks whether the current deduction follows from previous reasoning.
3. **Data Source Analysis** — traces geometric axioms, theorems, and numerical quantities.
4. **Calculation Analysis** — verifies algebraic transformations and arithmetic operations.

The verifier terminates with:

```text
Verification: Is the step correct? Yes/No
```

Instead of using this binary output directly, we extract the logits associated with `Yes` and `No` and convert them into a continuous **Soft-Reward**.

For step \(t\),

\[
p_t =
P(\mathrm{Yes}\mid I,Q,S_{<t},s_t)
=
\frac{\exp(z_{\mathrm{Yes}}-z_{\max})}
{\exp(z_{\mathrm{Yes}}-z_{\max})+
 \exp(z_{\mathrm{No}}-z_{\max})},
\]

where

\[
z_{\max}=\max(z_{\mathrm{Yes}},z_{\mathrm{No}}).
\]

The max-logit shift prevents numerical overflow during exponentiation.

For a trajectory

\[
\tau=\{s_1,\ldots,s_K\},
\]

we compute

\[
\mathcal{S}(\tau)
=
\frac{1}{K}\sum_{t=1}^{K}p_t
+
\lambda\min_{1\leq t\leq K}(p_t),
\qquad
\lambda\rightarrow0^+.
\]

The mean Soft-Reward is the primary ranking signal, while the minimum step score acts as a bottleneck-based tie breaker.

---

## Training Pipeline

The R-PRM training pipeline consists of four stages:

```text
Geo170K / Geometry QA
        │
        ▼
Positive / Negative
Now-Step Construction
        │
        ▼
Multimodal API Annotation
(Image + Question + History + Step)
        │
        ▼
Structured Four-Stage Verification
        │
        ▼
Hard Filtering
Prediction == Constructed Label
        │
        ▼
1,788 High-Quality Seed Examples
        │
        ▼
Stage-I R-PRM SFT
(Qwen3-VL-8B-Instruct + LoRA)
        │
        ▼
Large-Scale Offline
Process-Data Construction
        │
        ▼
Filtering + Hard-Negative Balancing
        │
        ▼
13,935 Process-Supervision Examples
        │
        ▼
Final R-PRM
        │
        ▼
Soft-Reward Verification
        │
        ▼
Best-of-N + Greedy Guard
```

### Stage 1 — Seed Process-Supervision Construction

We construct step-level verification instances from geometry reasoning examples.

Each instance contains:

```text
Image
Question
Previous Reasoning Steps
Current Step (Now Step)
Human / Constructed Label
```

Positive steps are derived from reasoning consistent with the correct solution.

Negative steps are constructed through controlled perturbations, including:

- incorrect parallel / perpendicular / tangent relations;
- unsupported congruence or similarity arguments;
- incorrect theorem applications;
- algebraic sign errors;
- equation transposition errors;
- arithmetic mistakes.

A multimodal API annotator based on `Qwen3-VL-8B-Instruct` generates the structured four-stage verification trace.

We apply **Hard Filtering** and retain an annotation only when

```text
Annotator Prediction == Constructed / Human Label
```

This produces **1,788 high-quality seed examples**.

### Stage 2 — Stage-I R-PRM SFT

The 1,788 seed examples are used to bootstrap the structured verifier.

Backbone:

```text
Qwen3-VL-8B-Instruct
```

Training:

```text
Method: LoRA SFT
Framework: ms-swift
Per-device batch size: 2
Gradient accumulation: 8
Epochs: 3
Approx. optimization steps: 300
Best checkpoint: step 200
```

Observed training statistics:

| Metric | Value |
|---|---:|
| Final training loss | 0.0737 |
| Average training loss | 0.1252 |
| Evaluation loss | 0.1276 |
| Training token accuracy | 97.35% |
| Evaluation token accuracy | 95.19% |

The token-level accuracy measures acquisition of the structured SFT output format and should **not** be interpreted as step-verification accuracy.

The selected LoRA checkpoint is merged with the original backbone to produce a standalone Stage-I R-PRM for offline inference.

### Stage 3 — Large-Scale Process-Data Expansion

We begin with **40,000 candidate examples** from Geo170K.

Samples containing only direct `Answer: X` responses are removed, leaving:

```text
40,000 candidates
        ↓
30,175 examples with intermediate reasoning
```

The Stage-I verifier supports offline structured verification-trace generation.

The expanded dataset is processed using:

- structured-format validation;
- trivial-step filtering;
- hard filtering;
- dynamic negative sampling;
- hard-negative balancing.

The original candidate pool contains more than 80% positive examples. After filtering and balancing, the final dataset contains:

```text
Total: 13,935 examples

Yes:  44.3%
No:   55.7%
```

### Stage 4 — Final R-PRM Training

The final R-PRM is trained on the 13,935-example process-supervision dataset.

```text
Backbone: Qwen3-VL-8B-Instruct
Precision: bfloat16
Optimizer: AdamW
Learning rate: 2e-5
Scheduler: cosine decay
Warmup ratio: 3%
Gradient accumulation: 4
Global batch size: 32
Epochs: 2
```

Validation performance:

| Metric | Value |
|---|---:|
| Step Classification Accuracy | **93.37%** |
| ROC-AUC | **0.961** |

---

## Test-Time Inference

The experiments reported in the paper use **trajectory-level test-time scaling**.

The policy generates:

- one deterministic greedy trajectory at \(T=0\);
- \(N\) stochastic trajectories at \(T=0.7\).

The candidate pool is

\[
\mathcal{C}
=
\{\tau_0\}
\cup
\{\tau_1,\ldots,\tau_N\},
\]

where \(\tau_0\) is the greedy trajectory.

R-PRM evaluates every candidate and selects

\[
\tau^*
=
\arg\max_{\tau\in\mathcal{C}}
\mathcal{S}(\tau).
\]

### Greedy Guard

Pure stochastic sampling may occasionally replace a correct deterministic solution with a degraded sampled trajectory.

We therefore retain the greedy solution as an additional candidate:

```text
Greedy (T=0) ───────────────┐
                            │
Sampling (T=0.7, N paths) ──┤
                            ▼
                    Candidate Pool
                            │
                            ▼
                     R-PRM Scoring
                            │
                            ▼
                       Best Path
```

This mechanism is referred to as **Greedy Guard**.

---

## Current Evaluation vs. Future Search

The distinction between the currently reported system and future extensions is important.

### Current Evaluation

The results reported in the paper use:

```text
Complete Trajectory Generation
          ↓
Best-of-N Candidate Pool
          ↓
R-PRM Step Verification
          ↓
Trajectory Aggregation
          ↓
Greedy Guard
          ↓
Final Reranking
```

That is:

> **Trajectory-level Best-of-\(N\) + Greedy Guard**

### Future / Extended System

Online search algorithms that query R-PRM during intermediate generation are **not used for the main reported results**.

Potential extensions include:

```text
PRM-Guided Step-Level Beam Search
```

and

```text
Monte Carlo Tree Search (MCTS)
```

where Soft-Rewards could guide branch expansion and pruning before complete trajectories are generated.

---

## Results

We evaluate on a 30-example geometry subset from MathVista `testmini`.

| Method | Correct / 30 | Accuracy |
|---|---:|---:|
| Policy Baseline (Greedy) | 15 | 50.00% |
| Random Pick @ N=4 | 14.25 | 47.50% |
| Random Pick @ N=8 | 18 | 60.00% |
| PRM Best-of-4 (Pure) | 15 | 50.00% |
| PRM Best-of-8 (Pure) | 19 | 63.33% |
| **PRM Best-of-(8+1) + Guard** | **20** | **66.67%** |
| Oracle Upper Bound (N=4) | 18 | 60.00% |
| **Oracle Upper Bound (N=8)** | **28** | **93.33%** |
| Qwen3.8-Flash (reported) | 27 | 90.00% |

The N=8 oracle result indicates that a correct solution is present in the candidate pool for 28 of the 30 examples. The gap between the **93.33% oracle coverage** and **66.67% realized R-PRM accuracy** suggests that verifier selection remains an important bottleneck.

> The oracle result is a candidate-coverage upper bound and should not be interpreted as an end-to-end model accuracy directly comparable to the reported commercial baseline.

---

## Repository Structure

```text
multimodal-rprm/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── scripts/
│       ├── convert_to_sft_format.py
│       ├── fix_sft_format.py
│       ├── make_large_pool.py
│       └── prepare_final_rprm_data.py
│
├── distill/
│   └── ...
│
├── training/
│   ├── train_final_rprm.sh
│   ├── merge_final_rprm_lora.py
│   └── merge_final_rprm_lora.sh
│
├── inference/
│   ├── run_best_of_n_rerank.py
│   ├── run_policy_prm_pipeline.py
│   ├── test_rprm_infer.py
│   └── test_soft_reward.py
│
├── evaluation/
│   ├── parse_mathvista_log.py
│   ├── diagnostics.py
│   └── step_diagnostics.py
│
├── demo/
│   └── app_gradio.py
│
├── scripts/
│   ├── deploy_pipeline.sh
│   └── master_pipeline.sh
│
├── configs/
│
└── assets/
    ├── figures/
    └── examples/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-rprm.git
cd multimodal-rprm
```

Create a Python environment:

```bash
conda create -n rprm python=3.10 -y
conda activate rprm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses Qwen3-VL as the multimodal backbone and `ms-swift` for LoRA-based supervised fine-tuning.

> Exact package versions should be matched to the training environment before reproducing the experiments.

---

## Data Preparation

Prepare the process-supervision data:

```bash
python data/scripts/convert_to_sft_format.py
```

Construct the larger candidate pool:

```bash
python data/scripts/make_large_pool.py
```

Prepare the final R-PRM training set:

```bash
python data/scripts/prepare_final_rprm_data.py
```

Large-scale datasets and generated images are not stored directly in this repository.

---

## Training

Run final R-PRM supervised fine-tuning:

```bash
bash training/train_final_rprm.sh
```

After training, merge the LoRA adapter into the backbone:

```bash
python training/merge_final_rprm_lora.py
```

or:

```bash
bash training/merge_final_rprm_lora.sh
```

The merged model can then be served as a standalone multimodal verifier.

---

## R-PRM Inference

Test the trained verifier:

```bash
python inference/test_rprm_infer.py
```

Test Soft-Reward extraction:

```bash
python inference/test_soft_reward.py
```

Run trajectory-level Best-of-\(N\) reranking:

```bash
python inference/run_best_of_n_rerank.py
```

---

## Evaluation

Parse MathVista evaluation logs:

```bash
python evaluation/parse_mathvista_log.py
```

Run step-level diagnostics:

```bash
python evaluation/step_diagnostics.py
```

The main reported metric is answer-level accuracy after trajectory reranking. R-PRM itself is additionally evaluated using step-level classification accuracy and ROC-AUC.

---

## Interactive Demo

A Gradio interface is provided for visualizing R-PRM verification results.

Run:

```bash
python demo/app_gradio.py
```

The diagnostic interface can display:

- step-level verification;
- Soft-Reward confidence;
- four-stage reasoning analysis;
- bottleneck steps;
- candidate trajectory comparison;
- Best-of-\(N\) reranking results.

---

## Model and Data Files

Model checkpoints, full datasets, and experiment outputs are intentionally excluded from Git.

Recommended `.gitignore` entries:

```gitignore
# Model checkpoints
output/
output_rprm_sft/
checkpoints/
*.safetensors
*.bin
*.pt
*.pth

# Large datasets
multimodal_data/
data/raw/*
data/processed/*

!data/raw/README.md
!data/processed/README.md

# Logs
logs/
*.log
wandb/

# Python
__pycache__/
*.pyc
.venv/
venv/

# Cache
.cache/
huggingface/

# macOS
.DS_Store
```

---

## Reproducibility Notes

The current repository distinguishes between two forms of test-time search:

**Reported experiments**

```text
Trajectory-Level Best-of-N + Greedy Guard
```

**Planned extensions**

```text
Step-Level PRM-Guided Beam Search / MCTS
```

Results from these two settings should not be mixed when reproducing the experiments reported in the paper.

The reported evaluation currently uses a relatively small 30-example MathVista geometry subset. The results should therefore be interpreted as a small-scale evaluation of multimodal process verification rather than a comprehensive benchmark comparison.

---

## Citation

If you find this project useful, please cite:

```bibtex
@article{multimodalrprm2026,
  title   = {Multimodal R-PRM: Process-Reward Verification for
             Geometric Reasoning and Test-Time Compute Scaling},
  author  = {Hu, Wenshuang and Ren, Xinyuan and Shi, Yuzhou},
  year    = {2026}
}
```

> The citation entry will be updated after publication.

---

## Acknowledgements

This project builds upon the Qwen3-VL model family, Geo170K / G-LLaVA, MathVista, LoRA, and related work on process reward models and test-time compute scaling.

Please refer to the accompanying paper for complete citations and experimental details.

---

## License

Please add the appropriate open-source license before public release.

The licenses and usage restrictions of the underlying models and datasets remain governed by their respective original licenses.
