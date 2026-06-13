# Pipeline

## Non-Negotiable Architecture

ActMap Voice should be presented as a cascaded voice pipeline:

```text
Audio input
  -> ElevenLabs Speech to Text
  -> local ActMap routing service
  -> route-specific backend action
  -> response text
  -> ElevenLabs Text to Speech
  -> audio output
```

ElevenLabs is responsible for speech I/O. The local ActMap service is responsible for extracting hidden activations and deciding whether the voice agent is allowed to answer immediately.

## ElevenLabs Touchpoints

Use these API surfaces in the demo plan:

- Speech-to-text: `POST /v1/speech-to-text` with `model_id=scribe_v2` and the recorded audio file.
- Text-to-speech: `POST /v1/text-to-speech/:voice_id` for simple audio generation.
- Streaming text-to-speech: use the current streaming TTS endpoint when the demo UI needs faster first audio.
- Optional ElevenAgents integration: use ElevenAgents as the voice UI or deployment surface, with ActMap exposed as an external routing service or tool.

The submission should explicitly log both the STT and TTS calls so judges can see ElevenLabs is part of the live loop.

## Local ActMap Service

The local service should expose one narrow contract:

```json
{
  "transcript": "Who has mentioned me in comments this month?",
  "context": {
    "channel": "voice",
    "tenant": "demo_workspace"
  }
}
```

Response:

```json
{
  "route": "VERIFY",
  "confidence": 0.87,
  "local_model": "Qwen/Qwen3-8B",
  "actmap_shape": [12, 32, 128],
  "action": "check_workspace_activity_before_speaking",
  "draft_response": "Let me check the current workspace activity before I answer that."
}
```

The route must be computed from local activations:

1. Format the transcript as a short support-agent prompt.
2. Run local `Qwen/Qwen3-8B` through vLLM.
3. Capture generated-token hidden states with hooks from `src/activations.py`.
4. Build and normalize the ActMap with shape `12 x 32 x 128`.
5. Score the ActMap with the router classifier.
6. Return the route before any user-facing speech is generated.

## Route Actions

`ANSWER`:

- Use the local model's safe, concise answer or a deterministic canned answer.
- Skip retrieval.
- Send the final text to ElevenLabs TTS.

`VERIFY`:

- Do not answer from parametric memory.
- Call the relevant lookup path: account data, policy docs, pricing table, refund policy, workspace activity, or ticket history.
- Only then generate the spoken response through ElevenLabs.

`ESCALATE`:

- Do not attempt to resolve the request in the voice agent.
- Generate a handoff message and create or display a human escalation record.
- Use ElevenLabs TTS only for the handoff acknowledgement.

## Demo Logging

Print one trace per turn:

```text
[elevenlabs:stt] transcript="..."
[actmap:local] model=Qwen/Qwen3-8B actmap_shape=12x32x128
[actmap:route] route=VERIFY confidence=0.87 action=retrieve_before_answer
[elevenlabs:tts] voice_id=... chars=...
```

This makes the prize-relevant architecture visible without requiring judges to inspect source code.

## Latency Strategy

Keep the local activation pass short:

- Warm the local model before the demo.
- Use a private generation of 16 to 32 tokens for routing.
- Keep ActMap classifier inference in-process.
- For the live demo, start with recorded audio clips if microphone capture is unstable.

The important story is not that ActMap replaces low-latency voice infrastructure. The story is that it adds a small pre-speech gate that avoids expensive retrieval on easy turns and avoids unsafe speech on risky turns.

## Implementation Order

1. Finish the local route CLI from transcript text to `ANSWER | VERIFY | ESCALATE`.
2. Wrap it as a local HTTP service.
3. Add a small script that records or loads audio, calls ElevenLabs STT, calls local ActMap, and calls ElevenLabs TTS.
4. Add visible trace output for the demo.
5. Optionally move the same service behind an ElevenAgents tool/custom integration.

## Official References

- Speech to Text: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
- Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
- Streaming Text to Speech: https://elevenlabs.io/docs/api-reference/text-to-speech/stream
- ElevenAgents overview: https://elevenlabs.io/docs/eleven-agents/overview
