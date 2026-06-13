# Presentation Outline

## 1. Title

**ActMap Voice: a safety router before AI voice agents speak**

One-liner: Voice agents should not retrieve for everything, but they should not confidently improvise when the request touches policies, account data, pricing, refunds, incidents, or risky actions.

## 2. Problem

Voice agents face a routing problem on every turn:

- Answering directly is fast and cheap, but can hallucinate on account-specific or high-risk requests.
- Retrieving every time is safer, but slower, more expensive, and often unnecessary.
- Escalating too late is dangerous when the user reports legal, security, compliance, payment, or business-critical issues.

## 3. Insight

The model's internal activations carry useful signals about uncertainty and operational risk before the agent speaks.

ActMap turns a short private local generation into an image-like activation map. A small classifier uses that map to choose `ANSWER`, `VERIFY`, or `ESCALATE`.

## 4. Product

ActMap Voice sits between transcription and speech:

```text
Transcript -> ActMap router -> route-specific action -> spoken response
```

The user still experiences a voice agent. The system gets a decision gate that can reduce unnecessary retrieval while preventing risky unsupported answers.

## 5. ElevenLabs Usage

ElevenLabs handles the voice experience:

- Speech-to-text for the user's audio.
- Text-to-speech or streaming TTS for the spoken reply.
- Optional ElevenAgents surface for the deployed voice UI.

ActMap does not replace ElevenLabs. It is the decision layer inserted before the agent emits speech.

Native ElevenLabs routing uses Agent Workflows: a graph of nodes and edges where LLM Conditions can evaluate transcript text to choose the next path. ActMap complements this by producing an activation-grounded route that can drive those same workflow paths.

## 6. Local Model Usage

The routing signal comes from a fixed local LLM:

- Model: `Qwen/Qwen3-8B`.
- Runtime: local vLLM with forward hooks.
- Captured signal: generated-token hidden activations.
- ActMap shape: `12 x 32 x 128`.
- Output: `ANSWER`, `VERIFY`, or `ESCALATE`.

This matters for the submission because the technical novelty is local activation introspection, not a hosted LLM prompt asking itself whether it is confident.

Report-backed credibility:

- ActMap has already been evaluated as a white-box uncertainty signal for generated QA.
- TriviaQA, 90,000 rows across Qwen3, Llama 3.1, and Mistral: ActMap ViT2D ensemble reaches 0.8856 AUROC.
- Best black-box baseline on the same split: MTE 0.8132 AUROC.
- Reproduced activation baselines on the same split: DRIFT 0.7593 AUROC, best-layer linear probe 0.7342 AUROC.
- TriviaQA to balanced NQ-Open transfer: ActMap 0.8438 AUROC, DRIFT 0.6732, linear 0.6319.

## 7. Demo Cases

Case 1, `ANSWER`:

> Where do I find the conversation simulator in the app?

Expected behavior: no retrieval, concise spoken answer.

Case 2, `VERIFY`:

> Who has mentioned me in comments this month?

Expected behavior: route to account/workspace lookup before speaking.

Case 3, `ESCALATE`:

> The CTO is furious about the data breach and says we cannot renew unless this is fixed now.

Expected behavior: no invented answer, immediate human handoff.

## 8. Why It Wins

ActMap Voice is cheaper than always-on retrieval, safer than blind parametric answering, and more inspectable than a black-box policy prompt. It uses ElevenLabs where voice quality matters and local activations where the routing signal matters.

Compared with native text-level LLM Conditions, ActMap optimizes a different decision:

- Native workflow condition: "What does the user intend?"
- ActMap route: "Does the model know enough to answer, or should this be verified/escalated?"

That is why the demo can claim both speed and precision: easy turns skip retrieval/tool latency, while risky turns are caught before the voice agent improvises.

Concrete benchmark contrast:

- Error prevented: "Is there a newer version of the desktop app available for me?" Dataset route `VERIFY`; text-only LM predicted `ANSWER`; ActMap predicted `VERIFY` with softmax `0.9768`.
- Cost saved: "Do you have a migration guide?" Dataset route `ANSWER`; text-only LM predicted `VERIFY`; ActMap predicted `ANSWER` with softmax `0.9863`.

## 9. Evidence

The literature supports the pain point:

- The local ActMap results show that activation maps are the best UQ method evaluated in the project, outperforming the strongest black-box baseline and reproduced white-box probes on the same benchmark.
- Adaptive RAG papers argue that always retrieving creates unnecessary overhead on simple queries.
- Spoken-dialogue papers treat cascaded ASR-LLM-TTS latency as a core user-experience problem.
- Customer-service AI studies show that escalation timing and transparency affect service quality and adoption.

Pilot ActMap Voice router benchmark:

- Same 1,000-example split.
- ActMap ViT2D: 0.9338 accuracy, 0.9312 macro F1.
- Text-only n-gram Naive Bayes: 0.9139 accuracy, 0.9135 macro F1.
- Slide claim: activation maps add signal beyond a simple transcript-only router.

Unit economics:

- In the pilot test split, `ANSWER + ESCALATE` is `102 / 151 = 67.5%` of turns, so always running RAG would be unnecessary for roughly two thirds of turns.
- Hackathon slide assumption: `10M` voice turns/day and `$0.02` per avoidable RAG/tool path.
- Always verifying everything: about `$135k/day` wasted, or about `$49M/year`.
- ActMap vs text-only router: about `$2.4M/year` in fewer RAG/tool calls and about `199k` more correct route decisions per day.

Positioning for ElevenLabs: ActMap helps decide when an ElevenLabs voice agent should use knowledge base/RAG, call tools, or transfer/escalate before it speaks.

## 10. Closing

The agent should not decide what to say only after it starts saying it. ActMap Voice gives the agent a pre-speech decision layer.
