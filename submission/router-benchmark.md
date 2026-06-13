# Router Benchmark

## Result

This benchmark compares the final ActMap vision router against a text-only LM router on the same held-out test split.

| Router | Accuracy | Macro F1 | Speed |
| --- | ---: | ---: | ---: |
| ActMap + ViT | 96.18% | 0.9619 | 390.7 rows/s |
| Text-only LM router | 91.01% | 0.9110 | 86.4 rows/s inference-only |

Delta:

- Accuracy gain: `+5.17` percentage points.
- Macro F1 gain: `+0.0509`.
- Speed: about `4.5x` faster for the final routing/classification step.

## Confusion Matrices

Rows are true labels and columns are predictions in `[ANSWER, VERIFY, ESCALATE]` order.

ActMap + ViT:

```text
[[139, 8,   0],
 [8,   142, 0],
 [0,   1,   147]]
```

Text-only LM router:

```text
[[123, 24, 0],
 [4,   145, 1],
 [3,   8,   137]]
```

## Caveat

This speed comparison is for routing once ActMaps exist. The current ActMap extraction pass is still unoptimized and dominates end-to-end latency.

Clean demo claim:

> ActMap gives a more accurate and much cheaper/faster final router than asking an LM to classify the transcript directly.

Local generated artifact:

- `runs/router_benchmark/test_qwen3_lm_vs_vit.json`

