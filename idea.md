# ActMap Voice

ActMap Voice is a decision layer for AI voice agents that determines, before the agent speaks, whether it should answer immediately, verify external information, or escalate to a human.

The core idea is that a voice agent should not retrieve documents for every request, but it also should not rely blindly on its parametric knowledge when the answer depends on policies, account data, pricing, refunds, or risky actions.

ActMap uses the hidden activations of a fixed local LLM, converted into image-like "activation maps," as an internal signal of uncertainty and operational risk.

In the demo, ElevenLabs handles the speech interface, while ActMap routes each user request into `ANSWER`, `VERIFY`, or `ESCALATE`, making the agent cheaper, faster, and safer.
