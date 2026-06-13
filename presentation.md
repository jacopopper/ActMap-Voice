# ActMap Voice Presentation

## Deck Goal

Tell a simple hackathon story:

> ElevenLabs gives us the voice-agent pipeline. ActMap improves the critical route decision before the reply is spoken.

Keep the wording precise:

- Say **before the user hears the reply** or **before speech**.
- Avoid saying **instant**, **Token-Zero**, or **first forward pass** unless we build the prefill-only version.
- Do not pitch "local" as the product constraint. Say the demo uses a fixed LLM activation signal; local execution is the current implementation detail.

Recommended length: 10 main slides plus 2 backup slides.

## One-Phrase Flow

1. **ElevenLabs already gives developers the full voice-agent pipeline.**
2. **The critical moment in that pipeline is the branch before the agent responds.**
3. **Text-based routing detects intent, but not whether the model actually knows enough to speak.**
4. **That creates the dilemma: answer fast and risk hallucination, or verify everything and add cost and silence.**
5. **ActMap solves the branch decision by reading hidden activations as uncertainty and risk signal.**
6. **The product route is simple: `ANSWER`, `VERIFY`, or `ESCALATE` before the reply is spoken.**
7. **The demo shows one ElevenLabs voice agent taking all three paths on realistic support requests.**
8. **The science is credible: ActMap is the strongest UQ method in our project benchmark.**
9. **The pilot product benchmark is stronger: ActMap is more accurate and faster than a text-only LM router.**
10. **The business case is scale: ElevenLabs orchestrates the conversation; ActMap gives it a better routing signal.**

## Slide 1: ElevenLabs Pipeline

**Phrase:** ElevenLabs already gives developers the full voice-agent pipeline.

**What to show:**

```text
User speech
  -> ElevenLabs STT
  -> Agent Workflow
  -> LLM Conditions / tools / RAG / transfer
  -> ElevenLabs TTS
```

**On-slide copy:**

```text
The voice interface is solved.
The next frontier is the decision layer.
```

**Speaker point:**

Start with the platform. ElevenLabs gives us speech, agents, workflows, tools, RAG, and handoff paths. ActMap is not replacing that; it improves the route choice inside it.

## Slide 2: The Branch Point

**Phrase:** The critical moment in that pipeline is the branch before the agent responds.

**What to show:**

Zoom into the workflow branch:

```text
Transcript
  -> route?
    -> answer now
    -> retrieve / call tools
    -> transfer to human
```

**On-slide copy:**

```text
Every turn asks the same question:
can the agent safely speak now?
```

**Speaker point:**

This is the product decision that matters: answer, verify, or escalate. It is not only about what the user asked; it is about what action is safe.

## Slide 3: Text Routing Gap

**Phrase:** Text-based routing detects intent, but not whether the model actually knows enough to speak.

**What to show:**

Two rows:

```text
Native text condition:
"This is about billing."

ActMap route:
"This billing turn needs live account data or escalation."
```

Example contrast:

```text
"Where is the refund policy?"      -> ANSWER
"Am I eligible for a refund?"      -> VERIFY
"Refund this or legal gets called" -> ESCALATE
```

**On-slide copy:**

```text
Intent is not enough.
Voice agents need reliability routing.
```

**Speaker point:**

LLM Conditions are useful for semantic routing, but ActMap targets a different question: does the model know enough to answer out loud, or should the workflow take a safer path?

## Slide 4: The Dilemma

**Phrase:** Answer fast and risk hallucination, or verify everything and add cost and silence.

**What to show:**

Split screen:

```text
Answer from memory
  + fast
  + cheap
  - hallucination risk

Verify every turn
  + safer
  - RAG/tool latency
  - extra cost
  - awkward filler speech
```

**On-slide copy:**

```text
Voice makes latency visible.
Wrong answers become spoken brand risk.
```

**Speaker point:**

In text chat, an extra second is annoying. In voice, it is silence. At the same time, a confident wrong spoken answer feels more authoritative and more dangerous.

## Slide 5: ActMap Insight

**Phrase:** ActMap solves the branch decision by reading hidden activations as uncertainty and risk signal.

**What to show:**

```text
Transcript
  -> fixed LLM private pass
  -> hidden activations
  -> activation map
  -> small CV router
```

Show an activation-map heatmap labelled:

```text
12 x 32 x 128 activation map
```

**On-slide copy:**

```text
Instead of asking another prompt,
ActMap looks inside the model.
```

**Speaker point:**

The novelty is not another text classifier. The route comes from the model's internal activation trajectory.

## Slide 6: Product Route

**Phrase:** The product route is simple: `ANSWER`, `VERIFY`, or `ESCALATE` before the reply is spoken.

**What to show:**

```text
ElevenLabs STT
  -> ActMap route
    -> ANSWER: speak directly
    -> VERIFY: RAG / tools / account data
    -> ESCALATE: human handoff
  -> ElevenLabs TTS or transfer
```

**On-slide copy:**

```text
Same voice experience.
Better decision before speech.
```

**Speaker point:**

This is the integration story: ActMap returns a route variable, and ElevenLabs Workflows use that route to choose the next node.

## Slide 7: Demo Story

**Phrase:** The demo shows one ElevenLabs voice agent taking all three paths on realistic support requests.

**What to show:**

Three columns:

```text
ANSWER
"Where do I find the conversation simulator?"
Stable product knowledge.

VERIFY
"Who mentioned me in comments this month?"
Needs live workspace data.

ESCALATE
"Someone used my card without permission."
Risky payment/security issue.
```

**On-slide copy:**

```text
One voice agent.
Three operational outcomes.
```

**Speaker point:**

The demo should make the routing visible: transcript, activation-map thumbnail, route badge, action taken, spoken response.

## Slide 8: UQ Evidence

**Phrase:** The science is credible: ActMap is the strongest UQ method in our project benchmark.

**What to show:**

| Method | Signal | AUROC |
| --- | --- | ---: |
| ActMap ViT2D ensemble | activation map | 0.8856 |
| MTE | black-box generations | 0.8132 |
| DRIFT | white-box activations | 0.7593 |
| Linear probe | hidden-state probe | 0.7342 |

Small transfer note:

```text
TriviaQA -> NQ-Open:
ActMap 0.8438 AUROC vs DRIFT 0.6732
```

**On-slide copy:**

```text
Activation maps beat the strongest evaluated black-box
and reproduced white-box baselines.
```

**Speaker point:**

Use the exact boundary: strongest UQ method in our project benchmark, not a universal claim across all UQ literature.

**Source/artifact:**

- `/home/jacopodardini/uni/EinAI/white_box_uq/doc/report.pdf`
- `submission/evidence.md`

## Slide 9: Router Benchmark

**Phrase:** The pilot product benchmark is stronger: ActMap is more accurate and faster than a text-only LM router.

**What to show:**

| Router | Input | Accuracy | Macro F1 | Routing speed |
| --- | --- | ---: | ---: | ---: |
| ActMap + ViT | activation map | 96.18% | 0.9619 | 390.7 rows/s |
| Text-only LM router | transcript text | 91.01% | 0.9110 | 86.4 rows/s |

Delta callout:

```text
+5.17 accuracy points
+0.0509 macro-F1
~4.5x faster final routing step
```

Optional small confusion-matrix visual:

```text
Rows=true, cols=pred [ANSWER, VERIFY, ESCALATE]

ActMap + ViT:
[[139, 8,   0],
 [8,   142, 0],
 [0,   1,   147]]

Text-only LM:
[[123, 24, 0],
 [4,   145, 1],
 [3,   8,   137]]
```

**On-slide copy:**

```text
Pilot result:
activation routing beats asking an LM
to classify the transcript directly.
```

Concrete demo contrast:

| User says | Dataset route | Text-only LM | ActMap |
| --- | --- | --- | --- |
| "Is there a newer version of the desktop app available for me?" | `VERIFY` | `ANSWER` | `VERIFY` |
| "Do you have a migration guide?" | `ANSWER` | `VERIFY` | `ANSWER` |

```text
One catches a risky unsupported answer.
One skips an unnecessary verification path.
```

**Speaker point:**

This is the clean product claim: once ActMaps exist, the final ActMap + ViT router is both more accurate and cheaper/faster than using a language model to classify the transcript directly. Caveat: the current ActMap extraction pass is still unoptimized and dominates end-to-end latency.

**Source/artifact:**

- `submission/router-benchmark.md`
- `runs/router_benchmark/test_qwen3_lm_vs_vit.json`
- Error-prevention example: benchmark row `id=2084`, ActMap predicted `VERIFY` with softmax `0.9768`.
- Cost-saving example: benchmark row `id=1628`, ActMap predicted `ANSWER` with softmax `0.9863`.

## Slide 10: Business Case And Close

**Phrase:** The business case is scale: ElevenLabs orchestrates the conversation; ActMap gives it a better routing signal.

**What to show:**

Big calculator:

```text
Router benchmark label mix:
66.3% of turns do not need RAG

Scenario:
10M voice turns/day
$0.02 avoidable RAG/tool path

Always verify:
~$133k/day wasted
~$48M/year

ActMap vs text-only LM:
5.84 points fewer VERIFY/RAG routes
~$11.7k/day avoided calls
~$4.3M/year
~517k more correct routes/day
```

Close with:

```text
Listen -> Route -> Act -> Speak
```

**On-slide copy:**

```text
Less unnecessary RAG.
Less awkward latency.
Earlier escalation.
Safer spoken agents.
```

**Speaker point:**

End with the partnership story. ElevenLabs provides the voice and workflow layer; ActMap provides a better activation-grounded route signal before speech.

## Backup Slide A: ElevenLabs Dry Run

**Phrase:** The ElevenLabs voice loop works in our demo environment.

**What to show:**

```text
TTS: passed
STT: passed
Transcript matched the generated demo line
```

**Source/artifact:**

- `submission/elevenlabs_dry_run.py`
- `submission/demo_artifacts/elevenlabs_dry_run.mp3`
- `submission/demo_artifacts/elevenlabs_dry_run_stt.json`

## Backup Slide B: Claim Boundaries

**Phrase:** We measured the routing signal now; full workflow integration is the next engineering step.

**What to show:**

```text
Measured now:
- ElevenLabs STT/TTS dry run
- ActMap UQ benchmark from report
- Pilot ActMap router vs text-only LM baseline
- Scenario cost model

Next:
- Prefill-only / Token-Zero activation route
- Optimize ActMap extraction latency
- Full ElevenLabs Workflow integration
- Larger real customer-support routing dataset
- End-to-end latency benchmark
```

**Speaker point:**

This keeps the pitch credible: strong pilot evidence, clear architecture, and obvious next work.
