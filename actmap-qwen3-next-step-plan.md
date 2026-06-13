# Plan: ActMap Qwen3 Local Extraction

**Generated**: 2026-06-13
**Estimated Complexity**: Medium

## Overview
Use the copied ActMap reference code to extract `12 x 32 x 128` activation maps from local `Qwen/Qwen3-8B` for the current `actmap_dataset.jsonl`, then train a small 3-way router for `ANSWER`, `VERIFY`, and `ESCALATE`.

## Copied Reference Files
- `src/activations.py`: vLLM forward hooks, Qwen3 no-think chat input, generated hidden-state collection, ActMap++ tensor construction, normalization.
- `src/check_actmaps.py`: QA-oriented ActMap tensor validator. Needs schema adaptation for ActMap Voice.
- `src/extract_white_features.py`: Hugging Face hidden-state feature extractor for DRIFT/linear-style baselines. Useful as a reference baseline, not the main demo path.

## Sprint 1: Dataset-to-ActMap Extraction
**Goal**: Produce a local `.pt` file with rows shaped for ActMap Voice.

### Task 1.1: Add ActMap Voice Extractor
- **Location**: `src/extract_actmap_voice.py`
- **Description**: Read `actmap_dataset.jsonl`, load `Qwen/Qwen3-8B` with vLLM, attach hooks from `src.activations`, generate a short response for each query, build and normalize ActMaps, and save rows to `data/actmap_voice_qwen3.pt`.
- **Acceptance Criteria**:
  - Supports `--limit`, `--offset`, `--save-every`, `--max-new-tokens`, `--temperature`, and `--output`.
  - Saves rows with `id`, `query`, `label`, `route_id`, `model`, `answer`, and `actmap`.
  - Resumes from existing output without duplicating rows.
- **Validation**:
  - Run a 10-row smoke extraction.
  - Verify every `actmap` has shape `(12, 32, 128)`.

### Task 1.2: Adapt Validator
- **Location**: `src/check_actmap_voice.py`
- **Description**: Validate ActMap Voice row schema and tensor quality.
- **Acceptance Criteria**:
  - Checks schema, label distribution, shape, dtype, finite values, zeroish maps, and sample norms.
  - Prints sample rows by label.
- **Validation**:
  - Passes on the 10-row smoke output.

## Sprint 2: Router Baseline
**Goal**: Train a fast 3-way classifier over extracted ActMaps.

### Task 2.1: Add Lightweight Router Trainer
- **Location**: `src/train_router.py`
- **Description**: Train a simple CNN or compact ViT-style classifier on ActMaps with stratified train/val/test split.
- **Acceptance Criteria**:
  - Reports accuracy, macro F1, per-class precision/recall, and confusion matrix.
  - Saves checkpoint and label mapping.
- **Validation**:
  - Train on a small subset first, then full 5,100 rows.

### Task 2.2: Add Inference Helper
- **Location**: `src/route_query.py`
- **Description**: Given one text query, extract Qwen3 ActMap, run the router, and print `ANSWER`, `VERIFY`, or `ESCALATE` with confidence.
- **Acceptance Criteria**:
  - Single-command local demo path works.
  - Latency is measured separately for extraction and router scoring.

## Sprint 3: Demo Integration
**Goal**: Make the hackathon loop visible.

### Task 3.1: Add Demo CLI
- **Location**: `src/demo_voice_router.py`
- **Description**: Simulate the pre-speech decision layer: transcript in, route out, mock response path.
- **Acceptance Criteria**:
  - Shows query, route, confidence, generated answer, and ActMap tensor stats.
  - Can be wired to ElevenLabs transcript input later.

## Risks & Gotchas
- Qwen3 8B activation extraction is GPU-memory heavy; start with `--limit 10` and low `--max-new-tokens`.
- vLLM hooks require eager mode and may break if vLLM internals change.
- The dataset is synthetic and surface-form separable, so keep a small human-written holdout set for demo credibility.
- The copied `extract_white_features.py` is not the main ActMap path; it is only useful for hidden-state baselines.

## Immediate Next Command
After adding `src/extract_actmap_voice.py`, run:

```bash
python -m src.extract_actmap_voice --input actmap_dataset.jsonl --output data/actmap_voice_qwen3_smoke.pt --limit 10 --max-new-tokens 32
```
