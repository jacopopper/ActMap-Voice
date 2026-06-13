# ActMap UQ Report Notes

Source: `/home/jacopodardini/uni/EinAI/white_box_uq/doc/report.pdf`

## Core Claim

ActMap is a white-box uncertainty quantification method built from local LLM runtime activations. It records hidden states during answer generation, compresses them into a fixed `12 x 32 x 128` activation map, and uses a computer-vision classifier to predict whether the generated answer is correct.

## Why It Matters For ActMap Voice

The voice demo is not starting from a generic safety heuristic. It is adapting an already measured UQ signal:

```text
QA report: generated answer -> ActMap -> correct / incorrect
Voice demo: transcript turn -> ActMap -> ANSWER / VERIFY / ESCALATE
```

The report supports the argument that structured runtime activations contain information about model reliability that simpler probes miss.

## Numbers To Use

TriviaQA in-domain benchmark:

- Dataset: 90,000 generated QA rows.
- Generator models: `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`.
- ActMap ViT2D ensemble: `0.8856` AUROC.
- ActMap ViT2D single model: `0.8788` AUROC.
- DRIFT baseline: `0.7593` AUROC.
- Best-layer linear probe: `0.7342` AUROC.

TriviaQA to balanced NQ-Open transfer:

- ActMap ViT2D ensemble: `0.8438` AUROC.
- DRIFT baseline: `0.6732` AUROC.
- Best-layer linear probe: `0.6319` AUROC.

Per-generator TriviaQA AUROC:

- Qwen3-8B: ActMap `0.8860`, DRIFT `0.8098`, linear `0.7561`.
- Llama-3.1-8B-Instruct: ActMap `0.8856`, DRIFT `0.7694`, linear `0.7582`.
- Mistral-7B-Instruct-v0.3: ActMap `0.8790`, DRIFT `0.6733`, linear `0.6847`.

Black-box baselines from the UQ project README and `data/black_results_30k_sota_gpu.json` on the shared 9,000-row TriviaQA test split:

- MTE: `0.8132` AUROC, `0.8471` AUPRC.
- Mean log probability / PPL: `0.8073` AUROC, `0.8443` AUPRC.
- Regular entropy: `0.7913` AUROC, `0.8318` AUPRC.
- Semantic entropy, paper-style: `0.7629` AUROC, `0.7633` AUPRC.
- `P(True)`: `0.7523` AUROC, `0.7858` AUPRC.

This supports the stronger project claim: ActMap is the best UQ method evaluated in the project, beating both the strongest black-box baseline and the reproduced white-box activation baselines.

## Governance Angle

ActMap produces an auditable intermediate object, not just a scalar confidence score. The report frames this as useful for:

- human review triggers;
- activation-derived risk logging;
- calibration monitoring after dataset or generator changes;
- channel ablations or saliency diagnostics;
- post-deployment audits.

This maps well to voice agents because a bad spoken answer is public, immediate, and hard to retract. The useful product behavior is therefore to decide before speech whether the agent should answer, verify, or escalate.

## Careful Submission Wording

Use:

> ActMap is the best uncertainty quantification method evaluated in this project benchmark, beating the strongest black-box baseline and reproduced white-box activation baselines on the shared TriviaQA test split.

Avoid:

> ActMap is universally the best uncertainty method on every benchmark.

The broader universal claim would require benchmarking every major UQ implementation across multiple datasets and generation settings. For the hackathon pitch, the project-benchmark claim is strong and defensible.
