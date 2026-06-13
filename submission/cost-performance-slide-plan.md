# Cost And Performance Slides

The goal is to show a credible advantage without building a full production ElevenLabs Workflow benchmark. Use three lightweight pieces of evidence:

1. a small routing-quality comparison;
2. a simple latency/cost model;
3. a qualitative "same topic, different action" precision demo.

## Slide 1: Routing Quality Pilot

Claim:

> Activation routing beats a transcript-only text baseline on the current 1,000-example ActMap Voice pilot.

Use the shared split from `runs/actmap_vit2d_incremental_1000/split_indices.json`.

| Router | Input signal | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| ActMap ViT2D | Local Qwen activations, `12 x 32 x 128` map | 0.9338 | 0.9312 |
| Text n-gram Naive Bayes | Transcript text only | 0.9139 | 0.9135 |

Delta:

- Accuracy: `+1.99` percentage points for ActMap.
- Macro F1: `+1.77` percentage points for ActMap.

Important caveat:

> This is a pilot benchmark on synthetic route labels, not the final product benchmark. The purpose is to show that activation maps add signal beyond a simple transcript-only classifier.

Do not use the smaller `runs/actmap_vit2d_incremental_current` checkpoint in slides. It is a 450-row smoke model and is weaker than the text baseline.

Reproduce:

```bash
python3 -m src.text_baseline
```

Artifacts:

- `runs/text_baseline_actmap1000/text_baseline_results.json`
- `runs/text_baseline_actmap1000/summary.md`

## Slide 2: Where Text Routing Fails

Use this to explain why text-only routing is not enough.

Text-only conditions are good at topic detection:

```text
"This is about refunds."
"This is about billing."
"This is about data deletion."
```

ActMap targets reliability and operational action:

```text
"Can the model safely answer this now?"
"Does this require current account or policy data?"
"Should this go directly to a human?"
```

Examples from text-baseline mistakes:

| Query | True route | Text baseline |
| --- | --- | --- |
| Someone used my card without permission. | `ESCALATE` | `VERIFY` |
| You charged a card I removed from my account. | `ESCALATE` | `VERIFY` |
| Critical documents have been permanently deleted. | `ESCALATE` | `VERIFY` |
| What are the results of experiment pricing-page-test? | `VERIFY` | `ANSWER` |
| How many team members use the desktop app vs web? | `VERIFY` | `ANSWER` |

Speaker note:

> The text baseline often recognizes the topic but misses the operational severity. ActMap is designed to separate "this topic is about billing" from "this turn should not be answered by the voice agent."

## Slide 3: Latency Model Without Full Benchmarking

Show the decision paths:

```text
Native LLM-condition path:
STT -> text LLM route -> maybe RAG/tool -> response LLM -> TTS

ActMap path:
STT -> local activation route -> ANSWER or VERIFY or ESCALATE -> TTS
```

Use measured and documented components:

- ElevenLabs dry run on this machine:
  - TTS: about `2.7s` for a short demo sentence.
  - STT: about `1.2s` on the generated demo clip.
- ElevenLabs docs say RAG adds slight response latency, around `250ms`.
- ElevenLabs Soft timeout exists because LLM response latency can require filler speech.

Avoid pretending we measured full production latency. Use this phrasing:

> We are not claiming a complete end-to-end ElevenAgents benchmark. We are showing that ActMap can remove whole branches from the critical path: no text LLM routing, no RAG, and no tool call on turns classified as `ANSWER`.

## Slide 4: Cost Model

Use a formula instead of made-up prices:

```text
Always-verify cost per N turns:
  N * (retrieval_or_tool_cost + extra_llm_route_cost)

ActMap cost per N turns:
  N * local_actmap_cost
  + N * P(VERIFY) * retrieval_or_tool_cost
  + N * P(ESCALATE) * handoff_cost
```

The savings driver:

```text
Avoided external calls = N * P(ANSWER)
```

Example scenario for a slide:

If `60%` of turns are stable support questions:

```text
Always RAG/tools: 100 external lookups per 100 turns
ActMap routed:   40 external lookups per 100 turns
Avoided:         60 external lookups per 100 turns
```

Speaker note:

> The exact dollar amount depends on the customer's LLM, vector database, tools, and support system. The structural cost advantage is simple: ActMap only pays the expensive external path when the route is `VERIFY` or `ESCALATE`.

Concrete pilot estimate:

```text
Pilot held-out split:
  ANSWER + ESCALATE = 102 / 151 = 67.5% of turns

At 1M voice turns/day and $0.005 per unnecessary RAG/tool path:
  1,000,000 * 67.5% * $0.005 = ~$3.4k/day
  annualized = ~$1.2M/year
```

ActMap vs the transcript-only router:

```text
ActMap triggers RAG on 29.8% of pilot turns.
Text n-gram router triggers RAG on 33.1%.
Delta = 3.31 percentage points fewer RAG/tool paths.

At 1M voice turns/day and $0.005 per RAG/tool path:
  1,000,000 * 3.31% * $0.005 = ~$166/day
```

The bigger ActMap-vs-text value is correctness:

```text
ActMap accuracy: 93.38%
Text router accuracy: 91.39%
Delta: +1.99 points

At 1M voice turns/day:
  1,000,000 * 1.99% = ~19.9k more correct routing decisions/day
```

For the actual hackathon slide, use the bigger scenario:

```text
Assume 10M voice turns/day and $0.02 per avoidable RAG/tool path.

Always verifying everything:
  10M * 67.5% unnecessary RAG * $0.02 = ~$135k/day
  annualized = ~$49M/year

ActMap vs text-only router:
  10M * 3.31% fewer RAG calls * $0.02 = ~$6.6k/day
  annualized = ~$2.4M/year

Accuracy impact:
  10M * 1.99% better routing = ~199k more correct routes/day
```

This is the memorable slide number. Keep the smaller `1M turns/day` calculation in speaker notes if challenged.

## Slide 5: Why This Helps ElevenLabs

Positioning:

> ElevenLabs already has strong workflow orchestration. ActMap gives those workflows a better routing signal.

Native Workflows:

- LLM Conditions route from transcript text.
- Soft timeout and tool sounds mask latency.
- RAG and tools improve correctness but add delay.

ActMap:

- Routes from local model activations.
- Sends easy turns directly to answer.
- Sends live-data turns to verification.
- Sends risky turns to escalation before the agent says the wrong thing.

## Minimal Benchmarking Checklist

Enough for hackathon slides:

- Run `python3 submission/elevenlabs_dry_run.py` and show STT/TTS traces.
- Run `python3 -m src.text_baseline` and show ActMap vs text-router table.
- Show three same-topic route examples: `ANSWER`, `VERIFY`, `ESCALATE`.
- Use the formula-based cost model instead of claiming exact production savings.
