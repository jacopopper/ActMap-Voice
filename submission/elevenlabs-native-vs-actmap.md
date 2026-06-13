# ElevenLabs Native Routing Vs ActMap Routing

## Native ElevenLabs Routing

ElevenLabs Agent Workflows are a visual graph system. Conversations move through nodes such as subagents, dispatch tools, agent transfer, phone transfer, or end nodes. Edges can be unconditional, expression-based, or LLM-conditioned.

The important baseline for ActMap is the **LLM Condition** edge:

```text
User speech
  -> ElevenLabs STT
  -> transcript text
  -> LLM evaluates natural-language condition
  -> workflow edge fires
  -> subagent, RAG, tool, transfer, or response
```

This is powerful because it routes semantically from natural language. The cost is that the routing decision is still a text-level LLM judgment. It decides from what the user said, not from what the answering model actually knows internally.

ElevenLabs also has latency-management features:

- **Soft timeout:** plays a filler phrase if the LLM response takes longer than the configured timeout.
- **Tool call sounds / pre-speech behavior:** can provide audio feedback while tool calls run.
- **RAG:** gives the agent document grounding, but the docs state that RAG adds slight response latency, around 250ms.

These are good voice UX features. They mask waiting. ActMap tries to remove avoidable waiting earlier.

## ActMap Routing

ActMap routes after transcription and before any user-facing response:

```text
User speech
  -> ElevenLabs STT
  -> transcript text
  -> local Qwen private generation
  -> hidden activations
  -> ActMap router
  -> ANSWER | VERIFY | ESCALATE
  -> ElevenLabs TTS or workflow/tool/handoff
```

The route is not just intent classification. It asks a different question:

> Is this a turn the model can safely answer from its own knowledge, or is this a turn that needs live data, retrieval, policy verification, or a human?

## Why Faster

Native workflow routing can require a text-level LLM condition, followed by RAG or tool execution, followed by response generation. When that takes too long, the platform can use soft timeout or tool-call audio to cover the wait.

ActMap aims to reduce the number of turns that enter the expensive path:

- `ANSWER` skips RAG/tools entirely for stable, low-risk questions.
- `VERIFY` uses retrieval/tools only when the activation signal says the turn depends on live or external information.
- `ESCALATE` avoids wasted agent/tool loops on cases that should go directly to a human path.

The implementation requirement for the demo is a warmed local model and a short private generation, around 16 to 32 tokens. The router classifier itself should be negligible compared with STT, TTS, RAG, or external API calls.

## Why More Precise

Text-level workflow conditions can identify intent:

```text
"The user is asking about refunds."
```

ActMap targets model reliability:

```text
"The local model's activation trajectory looks risky for answering this refund question directly."
```

That distinction matters because two user turns can look similar in text but require different operational actions:

- "Where is the refund policy page?" -> `ANSWER`
- "Am I eligible for a refund on my last invoice?" -> `VERIFY`
- "Refund this now or our legal team is getting involved." -> `ESCALATE`

The native text classifier sees the refund topic. ActMap is designed to decide whether the agent should trust its internal knowledge, fetch current account/policy data, or hand off.

## Best Integration Story

Do not pitch ActMap as competing with ElevenLabs. Pitch it as an external decision layer that can feed ElevenLabs Workflows:

```text
ActMap route=ANSWER
  -> continue in lightweight answer node

ActMap route=VERIFY
  -> enter RAG/tool node

ActMap route=ESCALATE
  -> enter agent-transfer or human-transfer node
```

In production, the ActMap result can be stored as a dynamic variable or returned by a server tool. Workflow edges can then use deterministic expression conditions on `route`, instead of asking an LLM to infer the same decision from transcript text.

That is the key technical contribution:

> ActMap converts the routing problem from text-level semantic guessing into activation-grounded reliability routing.

## Demo Line

Use this in the presentation:

> ElevenLabs already has excellent orchestration for what to do once a route is chosen. ActMap improves how the route is chosen: instead of only reading the transcript, it looks at the local model's hidden activations and asks whether the agent actually knows enough to speak.

## Official References

- Workflows and LLM conditions: https://elevenlabs.io/docs/eleven-agents/customization/agent-workflows
- Soft timeout and conversation flow: https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow
- Tool call sounds and pre-speech behavior: https://elevenlabs.io/docs/eleven-agents/customization/tools/tool-configuration/tool-call-sounds
- RAG latency and knowledge-base routing: https://elevenlabs.io/docs/eleven-agents/customization/knowledge-base/rag
