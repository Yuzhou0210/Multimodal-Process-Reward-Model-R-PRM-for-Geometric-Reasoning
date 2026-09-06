# Experimental Results

This directory contains summarized experimental results for the Multimodal R-PRM project.

## Files

### `stage2_training_metrics.json`

Contains the final recorded validation token accuracy for the Stage-2 R-PRM SFT run.

```text
eval_token_acc = 0.965761793725626
               ≈ 96.58%
```

This is a **token-level SFT metric**. It should not be interpreted as step-level verification accuracy or ROC-AUC.

### `main_results.csv`

Contains the main downstream results on the 30-example MathVista geometry evaluation subset.

The primary evaluated system is:

```text
Trajectory-Level Best-of-N
        +
Greedy Guard
        +
R-PRM Soft-Reward Reranking
```

The best realized R-PRM result is **66.67% (20/30)** with Best-of-(8+1) + Greedy Guard.

The Oracle @ N=8 result is **93.33% (28/30)** and represents candidate-space coverage rather than end-to-end system accuracy.

## Metric Clarification

- **Token Accuracy:** token-level prediction accuracy during SFT.
- **Step Verification Accuracy:** Yes/No correctness classification of a reasoning step.
- **Downstream Problem Accuracy:** final answer accuracy after trajectory selection.

These metrics should not be conflated.

## Notes

- The reported downstream evaluation contains 30 MathVista geometry examples.
- Step-level PRM-guided beam search and MCTS are not used in the reported main evaluation.
- `Random Pick @ N=4` uses an averaged correct count (14.25), rather than a single-run integer count.
