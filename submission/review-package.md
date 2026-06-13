# Review Package

Use this as the submission-form source of truth.

## Time Budget

Total review time must be under `1:30`.

Recommended reviewer path:

```text
0:00-1:15  Watch demo video.
1:15-1:30  Skim 5-slide presentation and notes.
```

Do not ask reviewers to read the long supporting docs unless they want backup.

## Required Links

GitHub repository:

```text
https://github.com/jacopopper/ActMap-Voice
```

Video:

```text
TODO: add YouTube or Vimeo link
```

Presentation:

```text
TODO: add PDF/Canva/Google Slides link generated from presentation.md
```

External links:

```text
ElevenLabs docs: https://elevenlabs.io/docs
ActMap Voice repository: https://github.com/jacopopper/ActMap-Voice
```

## Project Name

```text
ActMap Voice
```

## Elevator Pitch

```text
ActMap Voice routes ElevenLabs agents using LLM activation signals, deciding whether to answer, verify with RAG/tools, or escalate before the user hears an unsafe or slow reply.
```

## Short Notes

```text
Review path under 1:30: watch the short demo video, then skim the 5-slide deck. GitHub contains the runnable ElevenLabs STT/TTS dry run, router benchmark code, and backup evidence notes. Caveat: the 4.5x speed result is for the final routing step once ActMaps exist; ActMap extraction latency is still unoptimized in the prototype.
```

## What To Include In The Video

Keep the video between `60s` and `75s`.

Show only:

1. ElevenLabs voice-agent pipeline and the route branch.
2. ActMap activation-map route: `ANSWER`, `VERIFY`, `ESCALATE`.
3. One quick demo row per route.
4. Benchmark table:

| Router | Accuracy | Macro F1 | Routing speed |
| --- | ---: | ---: | ---: |
| ActMap + ViT | 96.18% | 0.9619 | 390.7 rows/s |
| Text-only LM router | 91.01% | 0.9110 | 86.4 rows/s |

5. Closing line:

```text
ElevenLabs orchestrates the conversation; ActMap gives it a better route signal before speech.
```

## What Not To Include In The Main Review Path

- Long literature discussion.
- Full confusion matrices.
- Full cost sensitivity table.
- Raw benchmark JSON.
- Full implementation walkthrough.

Those are useful backup materials, but they will break the `1:30` review window.

