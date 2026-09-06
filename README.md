# Multimodal R-PRM

**Process-Reward Verification for Geometric Reasoning and Test-Time Compute Scaling**

Multimodal R-PRM is a multimodal process reward model for fine-grained verification of geometric reasoning. It evaluates intermediate reasoning steps, extracts continuous Soft-Rewards, and reranks multiple candidate solutions at test time.

---

## Overview

For each reasoning step, R-PRM performs four-stage verification:

1. **Visual Analysis** — checks geometric primitives, annotations, and spatial relations.
2. **Now Step Analysis** — verifies whether the current step follows from the previous reasoning.
3. **Data Source Analysis** — checks the source of geometric axioms, theorems, and numerical values.
4. **Calculation Analysis** — verifies algebraic transformations and arithmetic calculations.

The verifier ends with a binary decision:

```text
Verification: Is the step correct? Yes / No
```

Instead of directly using this discrete prediction, we extract the logits of `Yes` and `No` and convert them into a continuous **Soft-Reward**.

For reasoning step `t`:

```text
p_t = P(Yes | I, Q, S_<t, s_t)

    = exp(z_Yes - z_max)
      -----------------------------------------------
      exp(z_Yes - z_max) + exp(z_No - z_max)

where:

z_max = max(z_Yes, z_No)
```

The max-logit shift improves numerical stability and prevents overflow during exponentiation.

For a reasoning trajectory:

```text
tau = {s_1, s_2, ..., s_K}
```

the trajectory score is:

```text
S(tau) = Mean Soft-Reward + lambda * Bottleneck Soft-Reward

       = (1/K) * sum(p_t) + lambda * min(p_t)

where lambda -> 0+
```

The **mean Soft-Reward** is the primary ranking signal, while the minimum step score serves as a bottleneck-based tie breaker.

---

## Training Pipeline

The R-PRM training pipeline consists of four main stages:

```text
Geo170K
   |
   v
Now-Step Construction
   |
   v
Multimodal API Annotation
   |
   v
Hard Filtering
   |
   v
1,788 High-Quality Seed Examples
   |
   v
Stage-I R-PRM SFT
   |
   v
Large-Scale Data Expansion
   |
   v
13,935 Process-Supervision Examples
   |
   v
Final R-PRM
```

### Stage 1: Seed Process-Supervision Construction

Positive and negative reasoning steps are constructed from geometry problems.

Negative examples include:

- incorrect geometric relations;
- unsupported theorem applications;
- algebraic sign errors;
- equation transposition errors;
- arithmetic mistakes.

A multimodal API annotator based on **Qwen3-VL-8B-Instruct** generates structured four-stage verification traces.

We apply **Hard Filtering**:

```text
Keep sample only if:

Annotator Prediction == Constructed / Human Label
```

This produces **1,788 high-quality seed examples**.

### Stage 2: Stage-I R-PRM SFT

The seed examples are used to fine-tune Qwen3-VL-8B-Instruct with LoRA.

```text
Backbone: Qwen3-VL-8B-Instruct
Framework: ms-swift
Method: LoRA SFT
Per-device batch size: 2
Gradient accumulation: 8
Epochs: 3
Best checkpoint: Step 200
```

Stage-I training results:

| Metric | Value |
|---|---:|
| Final Training Loss | 0.0737 |
| Average Training Loss | 0.1252 |
| Evaluation Loss | 0.1276 |
| Training Token Accuracy | 97.35% |
| Evaluation Token Accuracy | 95.19% |

Token accuracy measures acquisition of the structured SFT output format and should not be interpreted as step-verification accuracy.

### Stage 3: Large-Scale Data Expansion

The large-scale data construction starts from **40,000 Geo170K candidates**.

```text
40,000 candidates
        |
        | Remove answer-only examples
        v
30,175 reasoning examples
        |
        | Structured filtering
        | Hard-negative construction
        | Dynamic negative balancing
        v
13,935 final examples
```

Final label distribution:

```text
Yes: 44.3%
No:  55.7%
```

### Stage 4: Final R-PRM Training

```text
Backbone: Qwen3-VL-8B-Instruct
Precision: bfloat16
Optimizer: AdamW
Learning Rate: 2e-5
Scheduler: Cosine Decay
Warmup Ratio: 3%
Global Batch Size: 32
Gradient Accumulation: 4
Epochs: 2
```

Final validation performance:

| Metric | Result |
|---|---:|
| Step Classification Accuracy | **93.37%** |
| ROC-AUC | **0.961** |

---

## Test-Time Inference

The experiments reported in the paper use **trajectory-level Best-of-N reranking with Greedy Guard**.

```text
             Input
               |
       +-------+-------+
       |               |
       v               v
 Greedy T=0       Sampling T=0.7
       |               |
       v               v
    tau_0       {tau_1, ..., tau_N}
       |               |
       +-------+-------+
               |
               v
        Candidate Pool
               |
               v
       R-PRM Verification
               |
               v
      Soft-Reward Scoring
               |
               v
      Trajectory Reranking
               |
               v
          Best Answer
```

The candidate pool is:

```text
C = {tau_0, tau_1, ..., tau_N}
```

and the final solution is:

```text
tau* = argmax S(tau), for tau in C
```

### Current Evaluation

The reported results use:

```text
Trajectory-Level Best-of-N + Greedy Guard
```

### Future Extensions

The following online search methods are **not used in the current reported experiments**:

```text
Step-Level PRM-Guided Beam Search
Monte Carlo Tree Search (MCTS)
```

They are considered future extensions of the current system.

---

## Results

Evaluation on a 30-example MathVista geometry subset:

| Method | Accuracy |
|---|---:|
| Greedy Baseline | 50.00% |
| Random Pick @ N=8 | 60.00% |
| PRM Best-of-8 | 63.33% |
| **PRM Best-of-(8+1) + Greedy Guard** | **66.67%** |
| Oracle @ N=8 | **93.33%** |
| Qwen3.8-Flash (Reported) | 90.00% |

The gap between the **93.33% oracle candidate coverage** and the **66.67% realized R-PRM accuracy** indicates that verifier selection remains an important bottleneck.

> **Note:** Oracle @ N=8 represents candidate-space coverage rather than end-to-end system accuracy and should not be interpreted as a direct comparison with the commercial baseline.

---

## Repository Structure

```text
Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning/
|
├── demo/
|
├── scripts/
│   ├── Data_preparation/
│   ├── Data_distillation/
│   ├── Gradio/
│   ├── Training/
│   ├── Inference/
│   └── Evaluation/
|
├── README.md
└── .gitignore
```

### Directory Description

| Directory | Description |
|---|---|
| `scripts/Data_preparation/` | Dataset filtering, conversion, and process-step construction |
| `scripts/Data_distillation/` | API annotation, structured CoT generation, and hard filtering |
| `scripts/Training/` | Stage-I and final R-PRM training / LoRA merging |
| `scripts/Inference/` | Soft-Reward inference and Best-of-N reranking |
| `scripts/Evaluation/` | MathVista evaluation and diagnostic scripts |
| `scripts/Gradio/` | Interactive R-PRM visualization interface |
| `demo/` | Demo resources and examples |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Yuzhou0210/Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning.git

cd Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> Package versions should be matched to the training environment when reproducing the experiments.

---

## Usage

### Data Preparation

Scripts for constructing and processing the training data are located in:

```text
scripts/Data_preparation/
```

### Data Distillation

API-based annotation and hard filtering:

```text
scripts/Data_distillation/
```

### Training

Stage-I and final R-PRM training scripts:

```text
scripts/Training/
```

### Inference

R-PRM inference, Soft-Reward extraction, and Best-of-N reranking:

```text
scripts/Inference/
```

### Evaluation

MathVista evaluation and step-level diagnostics:

```text
scripts/Evaluation/
```

### Gradio Demo

Interactive visualization:

```text
scripts/Gradio/
```

---

## Citation

If you find this project useful, please cite:

```bibtex
@article{multimodalrprm2026,
  title  = {Multimodal R-PRM: Process-Reward Verification for Geometric Reasoning and Test-Time Compute Scaling},
  author = {Hu, Wenshuang and Ren, Xinyuan and Shi, Yuzhou},
  year   = {2026}
}
```

The citation information will be updated after publication.

---

## Acknowledgements

This project builds upon Qwen3-VL, Geo170K / G-LLaVA, MathVista, LoRA, and related work on process reward models and test-time compute scaling.

Please refer to the accompanying paper for complete references and experimental details.

---

## License

Please follow the licenses and usage restrictions of the underlying models and datasets used in this project.
