# ActMap Voice Submission

This folder contains the hackathon submission structure for ActMap Voice: the pitch, the demo flow, and the integration plan that uses ElevenLabs for the voice interface while keeping the activation model local.

## Reviewer Path

The hackathon review limit is `1:30` total, so the primary materials are intentionally short:

1. Watch the `60-75s` demo video.
2. Skim the `5-slide` presentation generated from `../presentation.md`.
3. Use `review-package.md` for the exact GitHub/video/presentation links and short notes.

Everything else in this folder is backup evidence for judges who want more detail.

## Submission Claim

ActMap Voice is a pre-speech decision layer for AI voice agents. Before the agent talks, it routes the user request into one of three actions:

- `ANSWER`: answer immediately without retrieval.
- `VERIFY`: check external information before speaking.
- `ESCALATE`: hand off to a human for high-risk or business-critical cases.

The core differentiator is that the route is not based only on keywords or an LLM self-report. ActMap runs a fixed local LLM, extracts hidden activations during a short private generation, converts those activations into image-like ActMaps, and classifies operational risk from that internal signal.

## Folder Structure

- `review-package.md`: short submission-form source of truth.
- `../presentation.md`: 5-slide, 90-second presentation plan and video script.
- `presentation-outline.md`: longer backup slide narrative.
- `pipeline.md`: technical architecture for ElevenLabs plus local ActMap routing.
- `demo-script.md`: live demo sequence, prompts, and fallback plan.
- `evidence.md`: literature and product rationale for why this is a real pain point.
- `actmap-uq-report-notes.md`: benchmark numbers and careful claims from the local ActMap UQ report.
- `elevenlabs-native-vs-actmap.md`: comparison between native ElevenLabs workflow routing and ActMap activation routing.
- `cost-performance-slide-plan.md`: lightweight slide plan for cost, latency, and text-baseline comparison.
- `rag-cost-estimate.md`: scenario-based dollar estimates for unnecessary RAG/tool routing.
- `router-benchmark.md`: compact ActMap-vs-text-only-LM router benchmark.
- `elevenlabs_dry_run.py`: repeatable live API smoke test for ElevenLabs TTS and STT.

## Required Positioning

The demo should make two facts explicit:

- ElevenLabs is the speech layer: audio input, transcript, spoken response, and optional ElevenAgents UI/deployment.
- ActMap is the local decision layer: `Qwen/Qwen3-8B` runs locally through vLLM hooks, produces `12 x 32 x 128` activation maps, and decides whether speech should proceed.

Because this demo is being built on an ElevenLabs Creator plan, keep the integration Creator-plan-safe:

- Use direct ElevenLabs API calls for speech-to-text and text-to-speech.
- Treat ElevenAgents, telephony, SIP transfer, enterprise retention controls, and production deployment features as optional stretch paths.
- Do not depend on premium-only audio formats or enterprise-only zero-retention settings.
- Record visible API traces so the ElevenLabs usage is obvious even if the demo is a local script.

The shortest diagram:

```text
User voice
  -> ElevenLabs STT
  -> local ActMap router over local Qwen activations
  -> ANSWER | VERIFY | ESCALATE
  -> response text or handoff text
  -> ElevenLabs TTS
  -> user hears the result
```

## Submission Proof Points

During the demo, show:

- The incoming transcript came from ElevenLabs.
- The local model name and ActMap tensor shape.
- The route decision, confidence, and action.
- The final spoken audio generated through ElevenLabs.

## Core Differentiation

Native ElevenLabs Workflows can route conversations through LLM conditions over transcript text. ActMap adds a different signal: the local model's hidden activations during a short private generation. The pitch is that ElevenLabs provides the voice and orchestration, while ActMap makes the pre-speech route faster on easy turns and more precise on risky turns.

## ElevenLabs Dry Run

Run this before the live demo:

```bash
python3 submission/elevenlabs_dry_run.py
```

It loads `.env`, generates a short TTS clip, transcribes that clip with ElevenLabs STT, and saves local artifacts under `submission/demo_artifacts/`.

## Official ElevenLabs References

Checked on 2026-06-13:

- Speech to Text: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- Streaming Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/stream
- ElevenAgents overview: https://elevenlabs.io/docs/eleven-agents/overview
