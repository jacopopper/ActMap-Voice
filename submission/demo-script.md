# Demo Script

## Setup Checklist

- Set `ELEVENLABS_API_KEY`.
- Set `ELEVENLABS_VOICE_ID`.
- Warm the local `Qwen/Qwen3-8B` model.
- Confirm the ActMap extractor prints `actmap_shape=12x32x128`.
- Prepare three audio clips or microphone prompts: one `ANSWER`, one `VERIFY`, one `ESCALATE`.
- Keep the first version Creator-plan-safe: direct STT API call, local ActMap route, direct TTS API call.

ElevenLabs live API dry run:

```bash
python3 submission/elevenlabs_dry_run.py
```

Useful local smoke command:

```bash
python -m src.extract_actmap_voice --input actmap_dataset.jsonl --output data/actmap_voice_qwen3_smoke.pt --limit-per-class 2 --max-new-tokens 32
```

Validator:

```bash
python -m src.check_actmap_voice data/actmap_voice_qwen3_smoke.pt
```

## 90-Second Narration

ActMap Voice is a decision layer for AI voice agents. It decides before the agent speaks whether the agent should answer, verify external information, or escalate to a human.

ElevenLabs handles the voice interface: speech in, speech out. ActMap sits in the middle. It runs a fixed local Qwen model, extracts hidden activations from a short private generation, converts them into an activation map, and routes the request.

ElevenLabs Workflows can already route conversations with LLM Conditions over transcript text. ActMap adds a more specific routing signal: it looks at whether the local model's internal activation trajectory suggests the agent actually knows enough to speak.

The result is a voice agent that does not retrieve documents for every basic question, but also does not invent answers when the request depends on current policy, account data, refunds, pricing, security, or urgent business impact.

## Live Demo Flow

### Turn 1: Answer

Prompt:

> Where do I find the conversation simulator in the app?

Show:

```text
[elevenlabs:stt] transcript="Where do I find the conversation simulator in the app?"
[actmap:local] model=Qwen/Qwen3-8B actmap_shape=12x32x128
[actmap:route] route=ANSWER confidence=...
[elevenlabs:tts] speaking direct answer
```

Spoken behavior: quick answer, no retrieval.

### Turn 2: Verify

Prompt:

> Who has mentioned me in comments this month?

Show:

```text
[elevenlabs:stt] transcript="Who has mentioned me in comments this month?"
[actmap:local] model=Qwen/Qwen3-8B actmap_shape=12x32x128
[actmap:route] route=VERIFY confidence=...
[backend] checking workspace activity
[elevenlabs:tts] speaking verified result or lookup acknowledgement
```

Spoken behavior: the agent checks current workspace data before answering.

Benchmark-backed contrast:

> Is there a newer version of the desktop app available for me?

Show:

```text
[baseline:text-lm] route=ANSWER
[actmap:route] route=VERIFY confidence=0.9768
[demo] ActMap blocks an unsupported spoken account-specific answer and verifies first
```

Use this when the presentation needs one clean example where the language-model router would answer from text alone, but ActMap takes the safer path.

### Turn 3: Escalate

Prompt:

> The CTO is furious about the data breach and says we cannot renew unless this is fixed now.

Show:

```text
[elevenlabs:stt] transcript="The CTO is furious about the data breach..."
[actmap:local] model=Qwen/Qwen3-8B actmap_shape=12x32x128
[actmap:route] route=ESCALATE confidence=...
[handoff] creating urgent human escalation
[elevenlabs:tts] speaking handoff acknowledgement
```

Spoken behavior: no invented remediation, immediate handoff.

### Optional Cost-Saving Contrast

Prompt:

> Do you have a migration guide?

Show:

```text
[baseline:text-lm] route=VERIFY
[actmap:route] route=ANSWER confidence=0.9863
[demo] ActMap skips an unnecessary lookup and speaks immediately
```

Use this if the judges ask how the router saves money rather than only how it reduces risk.

## Fallback Plan

If live microphone capture fails, use prerecorded audio files and still call ElevenLabs STT.

If local extraction is slow, show the warmed service trace and use a small prepared ActMap route cache for the three demo prompts. Be explicit that the cache was produced by the same local Qwen activation pipeline.

If TTS streaming is unstable, use the non-streaming text-to-speech endpoint and play the returned audio file.

If an ElevenAgents feature is unavailable on the current plan, do not block the demo. Run the same loop as a local voice script and mention ElevenAgents as the production integration target.

## Final Line

Voice agents need a moment of judgment before they speak. ActMap Voice gives them that moment.
