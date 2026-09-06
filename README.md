# Multimodal R-PRM

**Multimodal Process Reward Model for Geometric Reasoning**

Multimodal R-PRM is a process reward model designed for fine-grained verification of multimodal geometric reasoning. Instead of evaluating only the final answer, R-PRM verifies intermediate reasoning steps and converts the verifier's Yes/No logits into continuous Soft-Rewards for trajectory-level reranking.

The current system combines:

- Multimodal step-level verification
- Four-stage structured reasoning analysis
- Continuous Soft-Reward extraction
- Hard-negative process-supervision construction
- Best-of-N trajectory reranking
- Greedy Guard for robust test-time scaling

> **Current evaluation:** Trajectory-level Best-of-N + Greedy Guard  
> **Future extension:** Step-level PRM-guided beam search / MCTS

---

## Overview

Given a geometric image, question, previous reasoning history, and current reasoning step, R-PRM performs four-stage verification:

1. **Visual Analysis** — verifies geometric primitives, annotations, and spatial relations.
2. **Now Step Analysis** — checks whether the current deduction follows from the previous reasoning.
3. **Data Source Analysis** — traces geometric axioms, theorems, and numerical quantities.
4. **Calculation Analysis** — verifies algebraic transformations and arithmetic operations.

The verifier terminates with:

```text
Verification: Is the step correct? Yes / No
```

Instead of directly using the binary decision, we extract the logits associated with `Yes` and `No` and convert them into a continuous Soft-Reward.

For reasoning step `t`:

```text
p_t = P(Yes | I, Q, S_<t, s_t)

    = exp(z_Yes - z_max)
      ---------------------------------------------
      exp(z_Yes - z_max) + exp(z_No - z_max)

where

z_max = max(z_Yes, z_No)
```

The max-logit shift improves numerical stability and prevents overflow during exponentiation.

For a trajectory:

```text
tau = {s_1, s_2, ..., s_K}
```

the trajectory score is:

```text
S(tau) = (1/K) * sum(p_t) + lambda * min(p_t)

where lambda -> 0+
```

The mean Soft-Reward is the primary ranking signal, while the minimum step reward acts as a bottleneck-based tie breaker.

---

## Training Pipeline

The R-PRM training pipeline consists of four main stages:

```text
Geo170K Candidate Pool
        |
        v
Now-Step Construction
        |
        v
Positive / Negative Step Generation
        |
        v
Multimodal Structured Annotation
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
Filtering + Hard-Negative Balancing
        |
        v
13,935 Process-Supervision Examples
        |
        v
Final R-PRM Training
        |
        v
Soft-Reward Verification
        |
        v
Best-of-N + Greedy Guard
```

### Stage 1: Seed Process-Supervision Construction

Positive steps are constructed from reasoning consistent with the correct solution.

Negative steps are generated through controlled perturbations, including:

- corrupted parallel, perpendicular, or tangent relations;
- unsupported congruence or similarity arguments;
- incorrect theorem applications;
- algebraic sign errors;
- equation transposition errors;
- arithmetic mistakes.

A multimodal annotator generates structured verification traces.

We apply **Hard Filtering** and retain an annotation only when:

```text
Annotator Prediction == Constructed / Human Label
```

This produces **1,788 high-quality seed examples**.

---

### Stage 2: Stage-I R-PRM SFT

The seed examples are used to bootstrap the structured verifier.

```text
Backbone: Qwen3-VL-8B-Instruct
Framework: ms-swift
Method: LoRA SFT
Per-device batch size: 2
Gradient accumulation: 8
Epochs: 3
Best checkpoint: Step 200
```

The Stage-I model learns the structured multimodal verification format and is subsequently used to support large-scale offline process-data construction.

---

### Stage 3: Large-Scale Data Expansion

The data-expansion pipeline begins with **40,000 candidate examples** from Geo170K.

Samples containing only direct answers without intermediate reasoning are removed:

```text
40,000 candidate examples
        |
        | Remove answer-only samples
        v
30,175 reasoning examples
        |
        | Syntax validation
        | Trivial-step filtering
        | Hard-negative construction
        | Dynamic negative sampling
        v
13,935 final process-supervision examples
```

Final label distribution:

```text
Yes: 44.3%
No:  55.7%
```

The final dataset contains challenging negative examples involving both visual-topological and mathematical reasoning errors.

---

### Stage 4: Final R-PRM Training

The final R-PRM is trained on the expanded process-supervision dataset using Qwen3-VL-8B-Instruct as the backbone.

The released Stage-2 training logs report a final validation token accuracy of approximately:

```text
eval_token_acc = 0.96576
               = 96.58%
```

> `eval_token_acc` is a token-level SFT metric and should not be interpreted as step-level verification accuracy or ROC-AUC.

The complete Stage-I and Final R-PRM training configurations are provided in the `configs/` directory.

---

## Test-Time Inference

The main experiments use **trajectory-level Best-of-N reranking with Greedy Guard**.

```text
                 Input
                   |
          +--------+--------+
          |                 |
          v                 v
     Greedy Search     Random Sampling
        (T=0)             (T=0.7)
          |                 |
          v                 v
        tau_0        {tau_1, ..., tau_N}
          |                 |
          +--------+--------+
                   |
                   v
            Candidate Pool
                   |
                   v
        R-PRM Step Verification
                   |
                   v
          Soft-Reward Extraction
                   |
                   v
         Trajectory Aggregation
                   |
                   v
              Reranking
                   |
                   v
             Best Answer
```

The candidate pool is:

```text
C = {tau_0, tau_1, ..., tau_N}
```

where `tau_0` is the deterministic greedy trajectory.

The final trajectory is selected by:

```text
tau* = argmax S(tau), for tau in C
```

### Greedy Guard

Pure stochastic sampling may occasionally replace a correct deterministic solution with a degraded sampled trajectory.

To reduce this failure mode, the greedy trajectory generated at `T=0` is always retained in the candidate pool as a **Greedy Guard**.

---

## Current Evaluation vs. Future Search

### Current Evaluation

The experiments reported in this project use:

```text
Complete Trajectory Generation
          |
          v
Best-of-N Candidate Pool
          |
          v
R-PRM Verification
          |
          v
Soft-Reward Aggregation
          |
          v
Greedy Guard
          |
          v
Final Reranking
```

In short:

> **Trajectory-Level Best-of-N + Greedy Guard**

### Future Extensions

The following online search methods are not used in the reported main experiments:

- Step-level PRM-guided beam search
- Monte Carlo Tree Search (MCTS)

These methods are considered extensions in which R-PRM rewards could guide intermediate branch expansion and pruning.

---

## Results

We evaluate the trajectory-selection framework on a 30-example MathVista geometry subset.

| Method | Correct / 30 | Accuracy |
|---|---:|---:|
| Policy Baseline (Greedy) | 15 | 50.00% |
| Random Pick @ N=4 | 14.25 (avg.) | 47.50% |
| Random Pick @ N=8 | 18 | 60.00% |
| PRM Best-of-4 (Pure) | 15 | 50.00% |
| PRM Best-of-8 (Pure) | 19 | 63.33% |
| **PRM Best-of-(8+1) + Guard** | **20** | **66.67%** |
| Oracle Upper Bound (N=4) | 18 | 60.00% |
| **Oracle Upper Bound (N=8)** | **28** | **93.33%** |
| Qwen3.8-Flash (reported) | 27 | 90.00% |

The best realized R-PRM configuration improves the greedy baseline from **50.00% to 66.67%**.

The Oracle @ N=8 result shows that at least one correct candidate exists for 28 of the 30 evaluated examples. The gap between **93.33% candidate coverage** and **66.67% realized R-PRM accuracy** indicates that verifier-based candidate selection remains an important bottleneck.

> **Note:** Oracle accuracy measures candidate-space coverage and is not an end-to-end model result.

Detailed result files are available under `results/`.

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
│
├── configs/
│   ├── stage1_sft_args.json
│   └── final_rprm_args.json
│
├── examples/
│   └── process_supervision_example.jsonl
│
├── results/
│   ├── README.md
│   ├── main_results.csv
│   └── stage2_training_metrics.json
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Directory Description

| Directory | Description |
|---|---|
| `scripts/Data_preparation/` | Dataset filtering, conversion, and process-step construction |
| `scripts/Data_distillation/` | Structured annotation and hard filtering |
| `scripts/Training/` | Stage-I and Final R-PRM training / LoRA merging |
| `scripts/Inference/` | R-PRM inference, Soft-Reward extraction, and Best-of-N reranking |
| `scripts/Evaluation/` | MathVista evaluation and diagnostic scripts |
| `scripts/Gradio/` | Interactive R-PRM visualization |
| `configs/` | Training configurations for Stage-I and Final R-PRM |
| `examples/` | Small process-supervision examples |
| `results/` | Summarized experimental results |
| `demo/` | Demo resources |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Yuzhou0210/Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning.git

cd Multimodal-Process-Reward-Model-R-PRM-for-Geometric-Reasoning
```

Create an environment:

```bash
conda create -n rprm python=3.10 -y
conda activate rprm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
torch
transformers
accelerate
peft
qwen-vl-utils
ms-swift
vllm
datasets
numpy
pandas
Pillow
tqdm
scikit-learn
gradio
requests
```

> Exact versions of the original training environment are being reconstructed. The released dependencies specify the core software stack required by the project.

---

## Usage

### Data Preparation

Dataset processing and process-step construction scripts are located in:

```text
scripts/Data_preparation/
```

### Data Distillation

Structured annotation and hard-filtering scripts are located in:

```text
scripts/Data_distillation/
```

### Training

Stage-I and Final R-PRM training scripts are located in:

```text
scripts/Training/
```

### Inference

Soft-Reward extraction and trajectory-level Best-of-N reranking are located in:

```text
scripts/Inference/
```

### Evaluation

MathVista evaluation and diagnostic scripts are located in:

```text
scripts/Evaluation/
```

### Gradio Demo

The interactive R-PRM visualization interface is located in:

```text
scripts/Gradio/
```

---

## Model Weights

The final R-PRM is initialized from **Qwen3-VL-8B-Instruct** and fine-tuned using multimodal process supervision.

| Model | Description | Weights |
|---|---|---|
| Stage-I R-PRM | Seed-SFT verifier trained on 1,788 examples | Coming soon |
| **Final R-PRM** | Final verifier trained on the expanded process-supervision dataset | **Coming soon** |

The **Final R-PRM** is the model used for the main R-PRM inference and reranking experiments.

Model weights will be hosted separately rather than stored directly in this GitHub repository.

---

## Data

The complete upstream datasets are not redistributed directly through this repository.

The project uses geometry reasoning data derived from resources including **Geo170K** and evaluates on a geometry subset of **MathVista**.

Small examples of the process-supervision format are provided under:

```text
examples/
```

The full constructed process-supervision dataset may be released separately subject to the licenses and redistribution terms of the original data sources.

---

## Reproducibility Notes

Please distinguish the following metrics:

### Token Accuracy

Produced during SFT training and measures token-level prediction accuracy over the structured verification output.

### Step Verification Accuracy

Measures whether R-PRM correctly predicts the Yes/No correctness label of an individual reasoning step.

### Downstream Problem Accuracy

Measures whether the final trajectory selected by the complete system produces the correct answer to the geometry problem.

These metrics should not be conflated.

The currently released Stage-2 training logs support a validation token accuracy of approximately **96.58%**.

The main downstream evaluation contains 30 MathVista geometry examples and should therefore be interpreted as a small-scale evaluation of the proposed verification and test-time reranking framework.

---

## Citation

If you find this project useful, please cite:

```bibtex
@article{multimodalrprm2026,
  title  = {Multimodal R-PRM: Process-Reward Verification for
            Geometric Reasoning and Test-Time Compute Scaling},
  author = {Hu, Wenshuang and Ren, Xinyuan and Shi, Yuzhou},
  year   = {2026}
}
```

Citation information will be updated after publication.

---

## Acknowledgements

This project builds upon Qwen3-VL, Geo170K / G-LLaVA, MathVista, LoRA, ms-swift, vLLM, and prior work on process reward models and test-time compute scaling.

Please refer to the accompanying paper for complete references and experimental details.

---

## License

Please follow the licenses and usage restrictions of all underlying models, datasets, and third-party resources used in this project.

The release of derived data and model weights is subject to the applicable licenses of the original resources.
