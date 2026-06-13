from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


LABEL_TO_ID = {
    "ANSWER": 0,
    "VERIFY": 1,
    "ESCALATE": 2,
}

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_INPUT = Path("actmap_dataset.jsonl")
DEFAULT_OUTPUT = Path("data/actmap_voice_qwen3.pt")
DEFAULT_SYSTEM_PROMPT = (
    "You are a customer support voice agent for an AI software subscription "
    "platform. The user input is an automatic speech-to-text transcript and "
    "may contain minor transcription errors. The platform has Individual, Pro, "
    "and Enterprise plans; workspaces and projects; team seats and roles; API "
    "keys, webhooks, integrations, prompt tools, model playgrounds, analytics, "
    "usage dashboards, exports, and support tickets. You can answer stable "
    "general questions about features, navigation, terminology, setup steps, "
    "and high-level plan differences. For account-specific status, current "
    "pricing, invoices, refunds, payment issues, security incidents, privacy "
    "requests, compliance documents, contract terms, or urgent business-impact "
    "problems, do not invent details; say that you need to check the account, "
    "policy, or support record. Keep responses concise and natural for voice."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract decode-token ActMaps for ActMap Voice routing examples. "
            "The prompt prefill is intentionally skipped by the vLLM hooks; "
            "ActMaps are built from the generated private draft trajectory."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Select up to N examples for each route label before extraction.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--actmap-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--normalize-actmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-sample per-channel z-score normalization before saving.",
    )
    parser.add_argument(
        "--no-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Qwen3 thinking mode in the chat template when supported.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output before extraction.",
    )
    parser.add_argument(
        "--capture-mode",
        choices=["decode"],
        default="decode",
        help="Currently only decode-token ActMaps are supported.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used in the Qwen chat template before the STT transcript.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="Optional file whose contents override --system-prompt.",
    )
    return parser.parse_args()


def iter_jsonl_rows(path: Path, *, offset: int = 0, limit: int | None = None):
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= offset:
                continue
            if limit is not None and yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            validate_input_row(row, path=path, line_number=line_number)
            yielded += 1
            yield row


def select_input_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.limit_per_class is None:
        return list(iter_jsonl_rows(args.input, offset=args.offset, limit=args.limit))

    if args.limit is not None:
        raise ValueError("Use either --limit or --limit-per-class, not both.")
    if args.limit_per_class < 1:
        raise ValueError("--limit-per-class must be positive.")

    selected: list[dict[str, Any]] = []
    counts = {label: 0 for label in LABEL_TO_ID}
    for row in iter_jsonl_rows(args.input, offset=args.offset):
        label = row["label"]
        if counts[label] >= args.limit_per_class:
            continue
        selected.append(row)
        counts[label] += 1
        if all(count >= args.limit_per_class for count in counts.values()):
            break

    short = {
        label: count
        for label, count in counts.items()
        if count < args.limit_per_class
    }
    if short:
        raise ValueError(
            f"Not enough rows for --limit-per-class={args.limit_per_class}: {short}"
        )
    return selected


def validate_input_row(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    expected = {"id", "query", "label"}
    if set(row) != expected:
        raise ValueError(
            f"{path}:{line_number}: expected keys {sorted(expected)}, got {sorted(row)}"
        )
    if row["label"] not in LABEL_TO_ID:
        raise ValueError(f"{path}:{line_number}: unknown label {row['label']!r}")
    if not isinstance(row["id"], int):
        raise TypeError(f"{path}:{line_number}: id must be int")
    if not isinstance(row["query"], str) or not row["query"].strip():
        raise TypeError(f"{path}:{line_number}: query must be a non-empty string")


def load_existing_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import torch

    rows = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(rows, list):
        raise TypeError(f"Expected {path} to contain a list, found {type(rows).__name__}")
    return rows


def save_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(rows, tmp_path)
    tmp_path.replace(path)


def load_model(args: argparse.Namespace):
    from vllm import LLM

    llm = LLM(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    return llm, llm.get_tokenizer()


def release_model(llm: Any, tokenizer: Any | None = None) -> None:
    import gc

    import torch

    try:
        from src.activations import probe_remove_hooks

        llm.apply_model(probe_remove_hooks)
    except Exception as exc:
        print(f"[warn] could not remove vLLM hooks cleanly: {exc}")

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
            try:
                method()
            except Exception as exc:
                print(f"[warn] vLLM teardown method {'.'.join(attr_path)} failed: {exc}")

    del tokenizer
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_private_draft_prompt(query: str) -> str:
    return f"STT transcript: {query}"


def resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file is None:
        return args.system_prompt
    return args.system_prompt_file.read_text(encoding="utf-8").strip()


def chat_template_model_name(model_name: str) -> str:
    """Return a registry model id only when Qwen3-specific chat args are needed."""
    if "qwen3" in model_name.lower():
        return DEFAULT_MODEL
    return ""


def extract_rows(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import torch

    from src.activations import (
        build_actmappp,
        extract_generation_hidden_states_vllm,
        normalize_actmap,
        probe_setup_hooks,
    )

    if args.overwrite and args.output.exists():
        args.output.unlink()
    system_prompt = resolve_system_prompt(args)

    selected_rows = select_input_rows(args)
    dataset = load_existing_dataset(args.output)
    existing_keys = {
        (int(row["id"]), str(row["model"]))
        for row in dataset
        if "id" in row and "model" in row
    }
    pending_rows = [
        row for row in selected_rows if (int(row["id"]), args.model) not in existing_keys
    ]

    print(
        f"input_rows={len(selected_rows)} existing_rows={len(dataset)} "
        f"pending_rows={len(pending_rows)} output={args.output}"
    )
    if not pending_rows:
        return

    actmap_dtype = torch.float16 if args.actmap_dtype == "float16" else torch.float32
    llm = None
    tokenizer = None
    completed = 0
    failures = 0

    try:
        llm, tokenizer = load_model(args)
        num_layers = llm.apply_model(probe_setup_hooks)[0]
        print(
            f"vLLM ready for {args.model} | hooked_layers={num_layers} "
            f"| capture_mode=decode"
        )

        for row in pending_rows:
            prompt = build_private_draft_prompt(row["query"])
            try:
                result = extract_generation_hidden_states_vllm(
                    llm,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    no_think=args.no_think,
                    model_name=chat_template_model_name(args.model),
                    system_prompt=system_prompt,
                    logprobs=0,
                )
                if result is None:
                    raise RuntimeError("empty generation or missing decode hidden states")

                answer, hidden_states, token_ids, _logprobs = result
                actmap = build_actmappp(hidden_states)
                if args.normalize_actmap:
                    actmap = normalize_actmap(actmap)

                dataset.append(
                    {
                        "id": int(row["id"]),
                        "query": row["query"],
                        "label": row["label"],
                        "route_id": LABEL_TO_ID[row["label"]],
                        "model": args.model,
                        "answer": answer,
                        "generated_token_count": len(token_ids),
                        "capture_mode": "decode",
                        "system_prompt": system_prompt,
                        "actmap": actmap.detach().cpu().to(dtype=actmap_dtype),
                    }
                )
                completed += 1
            except Exception as exc:
                failures += 1
                print(f"[warn] row id={row['id']} failed: {exc}")
                continue

            if completed % max(args.save_every, 1) == 0:
                save_dataset(args.output, dataset)
                print(
                    f"saved_rows={len(dataset)} completed={completed} "
                    f"failures={failures}"
                )
    finally:
        if llm is not None:
            release_model(llm, tokenizer)
        save_dataset(args.output, dataset)

    print(
        f"finished model={args.model} completed={completed} "
        f"failures={failures} total_rows={len(dataset)}"
    )


def main() -> None:
    args = parse_args()
    extract_rows(args)


if __name__ == "__main__":
    main()
