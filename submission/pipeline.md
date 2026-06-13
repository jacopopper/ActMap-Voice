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

## Native ElevenLabs Baseline

The native ElevenLabs path for sophisticated routing is Agent Workflows:

```text
Transcript
  -> workflow graph
  -> LLM Condition evaluates edge from text
  -> subagent, RAG, tool, transfer, or response node
```

This is useful orchestration, but it is transcript-level routing. It decides from the semantic content of the user's words. When the LLM or tool path takes time, ElevenLabs can maintain voice UX with features such as soft timeout filler speech or tool-call sounds with pre-speech behavior.

ActMap should be framed as an upstream routing signal for this workflow:

```text
Transcript
  -> ActMap route from local activations
  -> deterministic workflow edge on route
  -> ANSWER node, VERIFY node, or ESCALATE node
```

The product difference is that ActMap asks whether the model appears to know enough to speak, not just what topic the user mentioned.

## ElevenLabs Touchpoints

Use these API surfaces in the demo plan:

- Speech-to-text: `POST /v1/speech-to-text` with `model_id=scribe_v2` and the recorded audio file.
- Text-to-speech: `POST /v1/text-to-speech/:voice_id` for simple audio generation.
- Streaming text-to-speech: use the current streaming TTS endpoint when the demo UI needs faster first audio.
- Optional ElevenAgents integration: use ElevenAgents as the voice UI or deployment surface, with ActMap exposed as an external routing service or tool.

The submission should explicitly log both the STT and TTS calls so judges can see ElevenLabs is part of the live loop.

## Creator Plan Constraints

The current demo should assume a Creator plan and avoid features that may require a higher tier:

- Primary path: local script or local web app calling ElevenLabs STT and TTS by API key.
- Safe output format: use the default TTS audio response unless the account confirms higher-quality formats are available.
- No hard dependency on ElevenAgents, SIP, phone transfer, workspace SSO, data residency, or zero-retention controls.
- Optional path: if ElevenAgents is available in the account, connect ActMap as a server-side tool or custom middleware. If not, the direct API demo still satisfies the ElevenLabs requirement.

This makes the demo robust: the prize-relevant claim is that ElevenLabs carries the speech loop while the local model carries the activation-routing loop.

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

## Why ActMaps Are The Signal

The project already has evidence that ActMaps are a strong white-box uncertainty representation. In the local UQ report, the same `12 x 32 x 128` activation-map idea is used to predict generated-answer correctness:

- ActMap ViT2D ensemble: 0.8856 AUROC on TriviaQA across Qwen3, Llama 3.1, and Mistral generations.
- Best black-box baseline, MTE: 0.8132 AUROC on the same shared test split.
- Mean log probability / PPL: 0.8073 AUROC on the same shared test split.
- Regular entropy: 0.7913 AUROC on the same shared test split.
- DRIFT reproduction: 0.7593 AUROC on the same shared split.
- Best-layer linear probe: 0.7342 AUROC on the same shared split.
- TriviaQA to balanced NQ-Open transfer: ActMap remains the strongest ranker at 0.8438 AUROC.

For ActMap Voice, the label changes from factual correctness to operational route. The value proposition stays the same: the best UQ signal from the project becomes a pre-speech decision signal for whether the agent should answer, verify, or escalate.

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

## Why This Helps ElevenLabs

ElevenLabs agents already expose knowledge bases, RAG, tools, and transfer-style workflows. ActMap is complementary: it decides which of those paths to invoke before speech begins.

The value proposition for a voice platform is:

- Lower latency on simple turns by skipping unnecessary LLM-conditioned workflow branches, RAG, and tools.
- Lower backend cost for customers running high-volume voice agents.
- Better user trust by avoiding confident spoken answers on policy, billing, refund, security, compliance, or account-specific requests.
- Cleaner handoffs because the escalation decision happens before the user hears a bad answer.
- Less need to mask avoidable latency with filler speech because easy turns take the direct answer path.

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
- Workflows and LLM conditions: https://elevenlabs.io/docs/eleven-agents/customization/agent-workflows
- Soft timeout and conversation flow: https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow
- Tool call sounds and pre-speech behavior: https://elevenlabs.io/docs/eleven-agents/customization/tools/tool-configuration/tool-call-sounds
- RAG latency and knowledge-base routing: https://elevenlabs.io/docs/eleven-agents/customization/knowledge-base/rag
