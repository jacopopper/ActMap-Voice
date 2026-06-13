# RAG Cost Estimate

## What We Can Claim

Public sources do not appear to disclose daily ElevenLabs call volume. Do not invent that number.

Instead, use a scenario model:

- Public scale context: ElevenLabs reportedly crossed `$500M ARR` in early 2026.
- Product context: ElevenLabs documents RAG as adding slight response latency, around `250ms`.
- Pricing context: ElevenLabs publishes plan/credit pricing and low-latency TTS as low as `$0.05/minute` on Business, but the customer's external RAG/tool/LLM stack has its own cost.
- Our pilot: ActMap determines when RAG is unnecessary.

## Pilot Rates From The 1,000-Example Split

Source artifacts:

- `runs/actmap_vit2d_incremental_1000/test_results.json`
- `runs/text_baseline_actmap1000/text_baseline_results.json`

Held-out test set: `151` turns.

True labels:

- `ANSWER`: `48`
- `VERIFY`: `49`
- `ESCALATE`: `54`

If a baseline runs RAG for every turn, RAG is unnecessary for:

```text
ANSWER + ESCALATE = 102 / 151 = 67.5% of turns
```

ActMap predicted route distribution from its confusion matrix:

```text
Predicted VERIFY = 45 / 151 = 29.8%
Predicted non-VERIFY = 106 / 151 = 70.2%
```

Text n-gram baseline predicted route distribution:

```text
Predicted VERIFY = 50 / 151 = 33.1%
```

ActMap vs text-only router:

- ActMap accuracy: `93.38%`
- Text baseline accuracy: `91.39%`
- Delta: `+1.99` percentage points
- ActMap triggers RAG `3.31` percentage points less often than the text baseline in this split.

## Slide Table: Wasted RAG If You Verify Everything

Assumption: a RAG/tool lookup costs a blended `$0.005` per turn. This is not an ElevenLabs price; it is a placeholder for the customer's vector search, context construction, extra LLM tokens, and backend API overhead.

| Daily voice turns | RAG calls that were not needed | Wasted/day at $0.005 |
| ---: | ---: | ---: |
| 100,000 | 67,550 | $338 |
| 1,000,000 | 675,497 | $3,377 |
| 10,000,000 | 6,754,967 | $33,775 |

Sensitivity:

| Daily voice turns | $0.002/RAG | $0.005/RAG | $0.010/RAG | $0.020/RAG |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | $135 | $338 | $675 | $1,351 |
| 1,000,000 | $1,351 | $3,377 | $6,755 | $13,510 |
| 10,000,000 | $13,510 | $33,775 | $67,550 | $135,099 |

Slide wording:

> In our pilot label mix, two thirds of turns should not go through RAG. At 1M voice turns/day, even a half-cent unnecessary RAG/tool path burns about `$3.4k/day`, or about `$1.2M/year`.

Annualized:

```text
$3,377/day * 365 = ~$1.23M/year
```

## Slide Table: ActMap Vs Text Router

This table isolates the delta between ActMap and a simple transcript-only router, not the delta against always-RAG.

ActMap triggers RAG `3.31` percentage points less often than the text n-gram baseline while also improving route accuracy by `1.99` points.

| Daily voice turns | Fewer RAG calls/day vs text router | Saved/day at $0.005 |
| ---: | ---: | ---: |
| 100,000 | 3,311 | $17 |
| 1,000,000 | 33,113 | $166 |
| 10,000,000 | 331,126 | $1,656 |

This is the conservative cost-only delta. The larger value is safety and correctness:

| Daily voice turns | More correct routes/day vs text router |
| ---: | ---: |
| 100,000 | 1,987 |
| 1,000,000 | 19,868 |
| 10,000,000 | 198,675 |

Slide wording:

> Against a transcript-only router, ActMap is not only more accurate; it also invokes the expensive verification path less often in the pilot split. At 1M turns/day, that is roughly `33k` fewer RAG/tool calls and `20k` more correct routing decisions per day.

## Recommended Money Slide

Title:

**The routing decision has real unit economics**

Body:

```text
Always verify:
  1M turns/day * 67.5% unnecessary RAG * $0.005 = $3.4k/day wasted

ActMap vs text router:
  1M turns/day * 3.31% fewer RAG calls * $0.005 = $166/day saved
  1M turns/day * 1.99% better routing = 19.9k more correct routes/day
```

Footer caveat:

> Scenario model. ElevenLabs call volume is not public; RAG cost depends on customer stack. The pilot result shows the direction and order of magnitude, not a production cost audit.

## Bigger Hackathon Slide

For the live presentation, use a more aggressive but still understandable platform-scale assumption:

```text
Assume:
  10M voice turns/day
  $0.02 blended cost per unnecessary RAG/tool path
  67.5% of turns do not need RAG in our pilot label mix

Always-verify waste:
  10,000,000 * 67.5% * $0.02 = ~$135k/day
  ~$49M/year
```

ActMap vs transcript-only routing at the same scale:

```text
ActMap invokes RAG 3.31 points less often than text-only routing.

10,000,000 * 3.31% * $0.02 = ~$6.6k/day
Annualized = ~$2.4M/year

ActMap is also +1.99 points more accurate:
10,000,000 * 1.99% = ~199k more correct route decisions/day
```

Slide wording:

> At 10M voice turns/day, always verifying everything can waste about `$49M/year` in avoidable RAG/tool calls. Even against a transcript-only router, ActMap's small pilot delta is worth about `$2.4M/year` in avoided calls, plus roughly `199k` more correct routing decisions every day.

If you want a bigger "platform scale" number in backup:

```text
100M voice turns/day * 67.5% unnecessary RAG * $0.02
= ~$1.35M/day
= ~$493M/year
```

Use this only as a sensitivity point, not the main claim.

## Why This Is Still Useful

Judges do not need exact internal ElevenLabs traffic to understand the business case. The important point is that voice agents operate at high turn volume, and every unnecessary RAG/tool branch adds:

- latency;
- vector search or tool cost;
- extra LLM context tokens;
- more chances for bad tool latency;
- a need for filler speech or tool sounds.

ActMap's business claim:

> Better routing compounds at scale. Even tiny per-turn savings become real money when voice agents handle hundreds of thousands or millions of turns per day.

## Sources To Cite

- ElevenLabs pricing: https://elevenlabs.io/pricing
- ElevenLabs RAG docs: https://elevenlabs.io/docs/eleven-agents/customization/knowledge-base/rag
- ElevenLabs ARR report: https://economictimes.indiatimes.com/tech/funding/voice-ai-firm-elevenlabs-tops-500-million-arr-announces-additional-funding/articleshow/130833329.cms
