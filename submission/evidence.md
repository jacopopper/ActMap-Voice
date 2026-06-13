# Evidence And Rationale

## Bottom Line

Yes, this is a real pain point. The literature supports three separate claims that make ActMap Voice credible:

1. Always-on retrieval is often wasteful because many requests do not need external context.
2. Voice agents are especially sensitive to latency because the user is waiting for audible response.
3. Customer-service automation needs earlier and cleaner human handoff when the AI is outside its capability or the user is upset.

ActMap Voice connects those claims into one product idea: use a fast local activation signal to choose `ANSWER`, `VERIFY`, or `ESCALATE` before ElevenLabs speaks.

## ActMap's Existing UQ Evidence

The local report at `/home/jacopodardini/uni/EinAI/white_box_uq/doc/report.pdf` gives the strongest ActMap-specific value proposition.

It evaluates ActMap as a white-box uncertainty quantification method for generated answers. The setup converts generated-token hidden states into a fixed `12 x 32 x 128` activation map and trains a vision classifier to predict answer correctness.

Main results from the report:

- On a 90,000-row TriviaQA benchmark across `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`, and `mistralai/Mistral-7B-Instruct-v0.3`, a two-seed ActMap ViT2D ensemble reaches **0.8856 AUROC**.
- On the same test split, reproduced white-box activation baselines reach **0.7593 AUROC** for DRIFT and **0.7342 AUROC** for a best-layer linear probe.
- Under transfer from TriviaQA to balanced NQ-Open, ActMap reaches **0.8438 AUROC**, while DRIFT reaches **0.6732** and the linear probe reaches **0.6319**.
- The report also notes that ActMap transfers well as a ranker, but its probabilities need recalibration under dataset shift because NQ ECE rises.

The broader UQ project also includes black-box baselines on the same 9,000-row TriviaQA test split. The best black-box baseline is MTE at **0.8132 AUROC** and **0.8471 AUPRC**; mean log probability reaches **0.8073 AUROC**, regular entropy reaches **0.7913 AUROC**, semantic entropy reaches about **0.7629 AUROC**, and `P(True)` reaches **0.7523 AUROC**. ActMap's two-seed ensemble at **0.8856 AUROC** and **0.9127 AUPRC** is therefore the best UQ method evaluated in this project benchmark.

Submission claim:

> ActMap is the best uncertainty quantification method evaluated in this project: it beats the strongest black-box baseline, MTE, and the reproduced white-box activation baselines, while also producing an auditable activation artifact rather than only a scalar confidence score.

The precise boundary is "best evaluated in this project benchmark," not a universal literature claim across every possible dataset and UQ implementation.

## Adaptive Retrieval Is Already A Known Cost Problem

The strongest related work is adaptive RAG. These papers are useful because they validate the problem while leaving room for ActMap's novelty.

- **Adaptive-RAG** argues that retrieval-augmented LLM systems can impose unnecessary computational overhead on simple queries and should dynamically choose between no retrieval, single-step retrieval, and more complex retrieval.  
  Source: https://arxiv.org/abs/2403.14403

- **Self-RAG** says fixed, indiscriminate retrieval can hurt usefulness when retrieval is unnecessary or retrieved passages are irrelevant, and proposes retrieval on demand.  
  Source: https://arxiv.org/abs/2310.11511

- **LLM-Independent Adaptive RAG** states directly that RAG mitigates hallucination but can carry high computational cost and can still risk misinformation. It studies lightweight adaptive retrieval features.  
  Source: https://arxiv.org/abs/2505.04253

- **L-RAG** frames "retrieve-always" RAG as a latency and overhead issue for high-throughput production systems, reporting retrieval reductions of 8% to 26% and latency savings of 80ms to 210ms in its tested setup when retrieval latency exceeds 500ms.  
  Source: https://arxiv.org/abs/2601.06551

Implication for ActMap: the route decision is not a gimmick. It is a known systems problem, but ActMap uses local hidden activations rather than only prompt-level self-reflection, query text features, or entropy.

## Voice Makes The Pain Sharper

In text chat, the user can tolerate a little extra delay. In voice, delay is felt as silence.

- A 2026 spoken-dialogue paper describes the conventional ASR-LLM-TTS pipeline as sequential and latency-heavy because transcription, reasoning, and speech synthesis happen one after another. Its proposed streaming architecture reduces response latency by 19% to 51% on two benchmarks.  
  Source: https://arxiv.org/abs/2602.23266

- Work on predictive ASR for voice assistants studies prefetching responses from partial ASR hypotheses, explicitly trading latency gains against the cost of failed predictions.  
  Source: https://arxiv.org/abs/2305.13794

- A 2024 paper on spoken avatar systems argues that LLM-driven spoken dialogue has a response-time problem and suggests classifiers to decide when a system can respond within human turn-taking constraints.  
  Source: https://arxiv.org/abs/2404.16053

Implication for ActMap: for ElevenLabs-style voice agents, "should I retrieve or escalate?" is not only an accuracy decision. It is also a latency and user-experience decision before audio starts.

## Escalation Is A Customer-Service Quality Problem

The handoff path matters because wrong spoken answers are worse than slow text answers: they feel authoritative, immediate, and branded.

- A 2026 field experiment on Alibaba customer service found that agentic AI reduced average chat duration but substantially lowered ratings for AI-eligible chats. It also found that intervention timing matters and early intervention is important for sustaining high post-escalation effort.  
  Source: https://arxiv.org/abs/2605.14830

- A 2025 customer-service chatbot study identifies "gatekeeper aversion": users underuse chatbot channels when an imperfect first stage may force transfer to an expert second stage. The paper recommends transparency about what chatbots can handle and faster live-agent access after chatbot failure.  
  Source: https://arxiv.org/abs/2504.06145

- A 2023 human-AI customer-support system paper notes common feedback that bots lack personal touch and fail to understand real user intent, motivating real-time human-AI collaboration.  
  Source: https://arxiv.org/abs/2301.12158

Implication for ActMap: escalation should not be a late apology after the agent fails. It should be one of the first routing outcomes.

## Why This Is Useful To ElevenLabs

ElevenLabs documentation positions the platform around speech APIs and conversational AI agents, including STT, TTS, knowledge bases, RAG, tools, and agent transfer-style workflows.  
Source index: https://elevenlabs.io/docs/llms.txt

The relevant native routing primitive is Agent Workflows: visual conversation graphs whose edges can use LLM Conditions evaluated in real time. Conversation Flow settings include Soft timeout, which speaks a filler phrase when the LLM takes longer than the configured timeout. Tool call sounds can also provide audio feedback during tool execution, and RAG is documented as adding slight response latency, around 250ms.

ActMap is useful to ElevenLabs because it can sit just before TTS or just before an ElevenAgents response:

```text
ElevenLabs transcript
  -> ActMap route
  -> answer now, retrieve/tool-call, or transfer/escalate
  -> ElevenLabs speech
```

That gives ElevenLabs builders a practical control layer:

- **For simple questions:** lower latency and lower customer backend cost.
- **For policy/account questions:** verify before speech.
- **For risky events:** hand off before the voice agent says the wrong thing.
- **For platform differentiation:** an internal-signal router is more novel than another prompt-only guardrail.

The sharper positioning is:

> ElevenLabs Workflows are strong at orchestrating paths. ActMap improves the path-selection signal by routing from model activations, not only from transcript text.

Specific ElevenLabs references:

- Workflows and LLM conditions: https://elevenlabs.io/docs/eleven-agents/customization/agent-workflows
- Soft timeout and conversation flow: https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow
- Tool call sounds and pre-speech behavior: https://elevenlabs.io/docs/eleven-agents/customization/tools/tool-configuration/tool-call-sounds
- RAG latency and knowledge-base routing: https://elevenlabs.io/docs/eleven-agents/customization/knowledge-base/rag

## Pitch Wording

Use this in the deck:

> Voice agents do not just need better voices. They need judgment before speech. ActMap gives an ElevenLabs voice agent a pre-speech decision layer: answer immediately when the turn is safe, verify when the answer depends on live data, and escalate before a risky answer is spoken.

More technical version:

> ActMap turns the local model's hidden-state trajectory into an uncertainty image. In our project benchmark, it is the best evaluated UQ method, beating the strongest black-box baseline and reproduced white-box activation baselines. ActMap Voice applies the same signal before speech: if the activation map looks risky, the ElevenLabs agent verifies or escalates instead of improvising out loud.
