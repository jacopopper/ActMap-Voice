from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.vit import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    ProbeDataset,
    build_model,
    checkpoint_model_config,
    checkpoint_state_dict,
    load_checkpoint_payload,
    load_rows,
    load_split_indices,
    select_device,
    _metrics,
    _run_epoch,
)


DEFAULT_DATA = Path("data/actmap_voice_qwen3_full_48tok.pt")
DEFAULT_SPLIT = Path("runs/actmap_vit2d_final_2967/split_indices.json")
DEFAULT_VIT_CHECKPOINT = Path("runs/actmap_vit2d_final_2967/best_model.pt")
DEFAULT_OUTPUT = Path("runs/router_benchmark/benchmark_results.json")
DEFAULT_LM_MODEL = "Qwen/Qwen3-8B"

LM_ROUTER_SYSTEM_PROMPT = """You are the routing classifier for a customer support voice agent.
The product is an AI software subscription platform with Individual, Pro, and Enterprise plans, customer accounts, billing, refund policies, privacy/security settings, and support workflows.

Choose exactly one route:
ANSWER: The voice agent can answer immediately from stable general product knowledge, navigation help, definitions, setup steps, or high-level feature and plan explanations.
VERIFY: The agent must check external or account-specific information before answering, including current pricing, plan limits, invoices, refunds, billing, account state, usage, analytics, policies, privacy/security settings, support tickets, or customer-specific data.
ESCALATE: A human should take over because the request is urgent, risky, adversarial, legal/compliance-related, security-incident-related, outage/business-impacting, asks for risky account action, or shows severe customer frustration.

Return only one label: ANSWER, VERIFY, or ESCALATE."""


LABEL_RE = re.compile(r"\b(ANSWER|VERIFY|ESCALATE)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ActMap ViT routing against a text-only LM router baseline."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--vit-checkpoint", type=Path, default=DEFAULT_VIT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lm-model", default=DEFAULT_LM_MODEL)
    parser.add_argument("--lm-max-new-tokens", type=int, default=8)
    parser.add_argument("--lm-temperature", type=float, default=0.0)
    parser.add_argument("--lm-batch-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def split_rows(rows: list[dict[str, Any]], split_file: Path, split: str) -> list[dict[str, Any]]:
    train_idx, val_idx, test_idx = load_split_indices(split_file)
    indices_by_split = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
    return [rows[index] for index in indices_by_split[split]]


def confusion_from_predictions(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in LABEL_TO_ID] for _ in LABEL_TO_ID]
    for truth, pred in zip(y_true, y_pred):
        matrix[LABEL_TO_ID[truth]][LABEL_TO_ID[pred]] += 1
    return matrix


def metrics_from_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    confusion = confusion_from_predictions(y_true, y_pred)
    total = len(y_true)
    correct = sum(truth == pred for truth, pred in zip(y_true, y_pred))
    per_class = {}
    f1s = []
    for label_id, label_name in ID_TO_LABEL.items():
        tp = confusion[label_id][label_id]
        fp = sum(row[label_id] for row in confusion) - tp
        fn = sum(confusion[label_id]) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_class[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": correct / max(total, 1),
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": confusion,
        "per_class": per_class,
    }


def evaluate_vit(
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    *,
    device_name: str,
    require_cuda: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    device = select_device(require_cuda=require_cuda, requested=device_name)
    payload = load_checkpoint_payload(checkpoint_path, device)
    model_config = checkpoint_model_config(payload, checkpoint_path)
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint_state_dict(payload))
    loader = DataLoader(
        ProbeDataset(rows, augment=False),
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    criterion = torch.nn.CrossEntropyLoss()
    loss, logits, labels = _run_epoch(
        model,
        loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
    )
    metrics = _metrics(labels, logits)
    metrics["loss"] = loss
    metrics["elapsed_seconds"] = time.perf_counter() - started_at
    metrics["rows_per_second"] = len(rows) / max(metrics["elapsed_seconds"], 1e-9)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["model_config"] = model_config
    return metrics


def apply_chat_template(tokenizer: Any, *, query: str) -> str:
    messages = [
        {"role": "system", "content": LM_ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify this automatic speech-to-text transcript for the next voice-agent action.\n\n"
                f"Transcript: {query}\n\nRoute:"
            ),
        },
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def parse_lm_label(text: str) -> str | None:
    match = LABEL_RE.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def evaluate_lm_router(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    import multiprocessing as mp
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from vllm import LLM, SamplingParams

    started_at = time.perf_counter()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        logprobs=0,
    )

    y_true: list[str] = []
    y_pred: list[str] = []
    raw_outputs: list[dict[str, Any]] = []
    invalid_count = 0
    inference_started_at = time.perf_counter()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [apply_chat_template(tokenizer, query=row["query"]) for row in batch]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            for row, output in zip(batch, outputs):
                text = output.outputs[0].text.strip() if output.outputs else ""
                parsed = parse_lm_label(text)
                if parsed is None:
                    invalid_count += 1
                    parsed = "VERIFY"
                y_true.append(row["label"])
                y_pred.append(parsed)
                raw_outputs.append(
                    {
                        "id": row["id"],
                        "query": row["query"],
                        "label": row["label"],
                        "prediction": parsed,
                        "raw_output": text,
                    }
                )
    finally:
        try:
            for attr_path in (
                ("llm_engine", "shutdown"),
                ("llm_engine", "engine_core", "shutdown"),
                ("llm_engine", "engine_core", "close"),
            ):
                obj = llm
                for attr in attr_path[:-1]:
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        break
                method = getattr(obj, attr_path[-1], None) if obj is not None else None
                if callable(method):
                    method()
        finally:
            del tokenizer
            del llm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    metrics = metrics_from_predictions(y_true, y_pred)
    metrics["invalid_count"] = invalid_count
    metrics["model"] = model_name
    metrics["elapsed_seconds_total"] = time.perf_counter() - started_at
    metrics["elapsed_seconds_inference"] = time.perf_counter() - inference_started_at
    metrics["rows_per_second_inference"] = len(rows) / max(metrics["elapsed_seconds_inference"], 1e-9)
    metrics["raw_outputs"] = raw_outputs
    return metrics


def summarize_errors(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    errors = []
    for pred in predictions:
        if pred["label"] == pred["prediction"]:
            continue
        errors.append(pred)
        if len(errors) >= limit:
            break
    return errors


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data)
    rows = split_rows(rows, args.split_file, args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    labels = {label: sum(row["label"] == label for row in rows) for label in LABEL_TO_ID}
    print(f"benchmark_rows={len(rows)} split={args.split} labels={labels}")

    print("evaluating_actmap_vit...")
    vit_metrics = evaluate_vit(
        rows,
        args.vit_checkpoint,
        device_name=args.device,
        require_cuda=args.require_cuda,
    )
    print(
        f"actmap_vit accuracy={vit_metrics['accuracy']:.4f} "
        f"macro_f1={vit_metrics['macro_f1']:.4f} "
        f"rps={vit_metrics['rows_per_second']:.1f}"
    )

    print("evaluating_text_only_lm_router...")
    lm_metrics = evaluate_lm_router(
        rows,
        model_name=args.lm_model,
        max_new_tokens=args.lm_max_new_tokens,
        temperature=args.lm_temperature,
        batch_size=args.lm_batch_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(
        f"text_lm accuracy={lm_metrics['accuracy']:.4f} "
        f"macro_f1={lm_metrics['macro_f1']:.4f} "
        f"invalid={lm_metrics['invalid_count']} "
        f"inference_rps={lm_metrics['rows_per_second_inference']:.1f}"
    )

    output = {
        "data": str(args.data),
        "split_file": str(args.split_file),
        "split": args.split,
        "n_rows": len(rows),
        "label_counts": labels,
        "actmap_vit": vit_metrics,
        "text_only_lm_router": lm_metrics,
        "text_lm_error_examples": summarize_errors(rows, lm_metrics["raw_outputs"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_results={args.output}")


if __name__ == "__main__":
    main()
