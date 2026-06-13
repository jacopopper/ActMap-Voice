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

## 6. Local Model Usage

The routing signal comes from a fixed local LLM:

- Model: `Qwen/Qwen3-8B`.
- Runtime: local vLLM with forward hooks.
- Captured signal: generated-token hidden activations.
- ActMap shape: `12 x 32 x 128`.
- Output: `ANSWER`, `VERIFY`, or `ESCALATE`.

This matters for the submission because the technical novelty is local activation introspection, not a hosted LLM prompt asking itself whether it is confident.

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

## 9. Closing

The agent should not decide what to say only after it starts saying it. ActMap Voice gives the agent a pre-speech decision layer.
