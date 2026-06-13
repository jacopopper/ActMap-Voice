# ActMap Voice: 90-Second Presentation

## Review Constraint

All material must be reviewable in **1:30 total**, so the submission should use:

- a `60-75s` demo video;
- a `5-slide` presentation;
- one short notes block with links and caveats.

Do not make judges read the long evidence files during review. Those files are only backup.

## One-Sentence Story

ElevenLabs gives us the voice-agent pipeline; ActMap improves the critical route decision before the reply is spoken by using LLM activation maps instead of transcript-only routing.

## Slide 1: ElevenLabs Pipeline

**Phrase:** ElevenLabs already gives us the voice-agent loop.

**Show:**

```text
User speech -> ElevenLabs STT -> Agent Workflow -> ElevenLabs TTS
```

Then zoom into the workflow branch:

```text
route?
  -> answer
  -> verify with RAG/tools
  -> escalate
```

**On-slide copy:**

```text
The voice interface is solved.
The risky part is choosing the path before speech.
```

**Say in video, ~10s:**

ElevenLabs already gives builders speech-to-text, agents, workflows, tools, RAG, and text-to-speech. ActMap focuses on the branch inside that pipeline: should the agent answer, verify, or escalate before the user hears a reply?

## Slide 2: The Problem

**Phrase:** Transcript routing sees intent, but not whether the model should speak.

**Show:**

```text
Text route:
"This is about billing."

Operational route:
"Can the agent safely answer this billing request out loud?"
```

Examples:

```text
"Where is the refund policy?"      -> ANSWER
"Am I eligible for a refund?"      -> VERIFY
"Refund this or legal gets called" -> ESCALATE
```

**On-slide copy:**

```text
Fast memory can hallucinate.
Always-RAG adds cost and awkward silence.
Late escalation creates spoken brand risk.
```

**Say in video, ~15s:**

Native text conditions are good at intent. But support routing is more than intent. A refund question might be a simple policy answer, an account-specific lookup, or a legal escalation. If we verify everything, the conversation gets slower and more expensive. If we answer everything from memory, the voice agent can confidently say the wrong thing.

## Slide 3: ActMap Solution

**Phrase:** ActMap routes from hidden activations, not only transcript text.

**Show:**

```text
Transcript
  -> fixed LLM private pass
  -> hidden activations
  -> 12 x 32 x 128 activation map
  -> ViT router
  -> ANSWER | VERIFY | ESCALATE
```

**On-slide copy:**

```text
ElevenLabs handles voice.
ActMap handles judgment before speech.
```

**Say in video, ~15s:**

ActMap runs a fixed model privately, converts hidden activations into an image-like activation map, and classifies the operational route with a small vision model. The result can feed ElevenLabs Workflows as a route variable: answer directly, verify with tools or RAG, or hand off to a human.

## Slide 4: Demo And Results

**Phrase:** In the pilot, ActMap is more accurate and faster than a text-only LM router.

**Show one demo row per route:**

```text
ANSWER:   "Where do I find the conversation simulator?"
VERIFY:   "Who mentioned me in comments this month?"
ESCALATE: "Someone used my card without permission."
```

**Show results table:**

| Router | Accuracy | Macro F1 | Routing speed |
| --- | ---: | ---: | ---: |
| ActMap + ViT | 96.18% | 0.9619 | 390.7 rows/s |
| Text-only LM router | 91.01% | 0.9110 | 86.4 rows/s |

**On-slide copy:**

```text
+5.17 accuracy points
+0.0509 macro-F1
~4.5x faster final routing step
```

**Say in video, ~20s:**

The demo shows one ElevenLabs voice agent taking three paths: answer, verify, and escalate. On our pilot router benchmark, ActMap plus ViT reaches 96.18% accuracy, compared with 91.01% for a text-only LM router. The final routing step is also about 4.5 times faster once ActMaps exist.

**Caveat for notes, not main slide:**

The speed comparison is for routing once ActMaps exist. Current ActMap extraction is still unoptimized and dominates end-to-end latency.

## Slide 5: Why It Matters

**Phrase:** Better routing saves money, latency, and bad handoffs at voice-agent scale.

**Show:**

```text
Scenario:
10M voice turns/day
$0.02 avoidable RAG/tool path
66.3% turns do not need RAG in benchmark mix

Always verify:
~$133k/day wasted
~$48M/year

ActMap vs text-only LM:
~517k more correct routes/day
~$4.3M/year avoided RAG/tool calls
```

**On-slide copy:**

```text
Listen -> Route -> Act -> Speak

ElevenLabs orchestrates the conversation.
ActMap gives it a better route signal.
```

**Say in video, ~15s:**

At scale, routing compounds. Better route decisions mean fewer unnecessary RAG calls, less waiting, and earlier human handoff on risky turns. The pitch is simple: ElevenLabs orchestrates the conversation; ActMap gives those workflows a better signal before speech.

## 75-Second Video Script

```text
ElevenLabs already gives developers the full voice-agent loop: speech-to-text, agent workflows, tools, RAG, transfer paths, and text-to-speech.

ActMap Voice focuses on the critical branch inside that loop. Before the agent speaks, should it answer, verify external information, or escalate to a human?

Text-based routing can tell that a user is asking about billing or refunds, but it cannot see whether the model actually knows enough to answer out loud. That creates the voice-agent dilemma: answer fast and risk hallucination, or verify everything and add cost and awkward silence.

ActMap routes from the model's hidden activations. We run a fixed model privately, convert its hidden states into a 12 by 32 by 128 activation map, and use a small vision router to choose ANSWER, VERIFY, or ESCALATE.

In the demo, ElevenLabs handles STT and TTS, while ActMap changes the workflow path. A stable product question gets answered immediately. A live account question goes to verification. A payment or security issue escalates instead of being improvised out loud.

On our pilot benchmark, ActMap plus ViT reaches 96.18% accuracy, versus 91.01% for a text-only LM router, and the final router is about 4.5 times faster once ActMaps exist.

At voice-agent scale, that means fewer unnecessary RAG calls, less latency, and safer handoffs. ElevenLabs orchestrates the conversation; ActMap gives it a better route signal before speech.
```

## Submission Form Copy

**Project name:** ActMap Voice

**Elevator pitch:**

```text
ActMap Voice routes ElevenLabs agents using LLM activation signals, deciding whether to answer, verify with RAG/tools, or escalate before the user hears an unsafe or slow reply.
```

**Notes:**

```text
Review path under 1:30: watch the short demo video, then skim the 5-slide deck. GitHub contains the runnable ElevenLabs STT/TTS dry run, router benchmark code, and backup evidence notes. Caveat: the 4.5x speed result is for the final routing step once ActMaps exist; ActMap extraction latency is still unoptimized in the prototype.
```

