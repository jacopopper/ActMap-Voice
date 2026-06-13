# ActMap Voice Submission

This folder contains the hackathon submission structure for ActMap Voice: the pitch, the demo flow, and the integration plan that uses ElevenLabs for the voice interface while keeping the activation model local.

## Submission Claim

ActMap Voice is a pre-speech decision layer for AI voice agents. Before the agent talks, it routes the user request into one of three actions:

- `ANSWER`: answer immediately without retrieval.
- `VERIFY`: check external information before speaking.
- `ESCALATE`: hand off to a human for high-risk or business-critical cases.

The core differentiator is that the route is not based only on keywords or an LLM self-report. ActMap runs a fixed local LLM, extracts hidden activations during a short private generation, converts those activations into image-like ActMaps, and classifies operational risk from that internal signal.

## Folder Structure

- `presentation-outline.md`: slide-by-slide submission narrative.
- `pipeline.md`: technical architecture for ElevenLabs plus local ActMap routing.
- `demo-script.md`: live demo sequence, prompts, and fallback plan.

## Required Positioning

The demo should make two facts explicit:

- ElevenLabs is the speech layer: audio input, transcript, spoken response, and optional ElevenAgents UI/deployment.
- ActMap is the local decision layer: `Qwen/Qwen3-8B` runs locally through vLLM hooks, produces `12 x 32 x 128` activation maps, and decides whether speech should proceed.

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

## Official ElevenLabs References

Checked on 2026-06-13:

- Speech to Text: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- Streaming Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/stream
- ElevenAgents overview: https://elevenlabs.io/docs/eleven-agents/overview
