from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any


LABEL_ORDER = ("ANSWER", "VERIFY", "ESCALATE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the benchmark ActMap row used by the static browser demo."
    )
    parser.add_argument("--row-id", type=int, default=2055)
    parser.add_argument("--data", type=Path, default=Path("data/actmap_voice_qwen3_full_48tok.pt"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/actmap_vit2d_final_2967/best_model.pt")
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("runs/router_benchmark/test_qwen3_lm_vs_vit.json")
    )
    parser.add_argument("--output", type=Path, default=Path("submission/actmap_demo_data.js"))
    return parser.parse_args()


def find_row(rows: list[dict[str, Any]], row_id: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("id", -1)) == row_id:
            return row
    raise ValueError(f"row id={row_id} not found")


def find_text_route(benchmark_path: Path, row_id: int) -> str:
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    for item in payload["text_only_lm_router"]["raw_outputs"]:
        if int(item["id"]) == row_id:
            return str(item["prediction"])
    raise ValueError(f"text route for id={row_id} not found in {benchmark_path}")


def main() -> None:
    args = parse_args()

    import torch

    root = Path.cwd()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.vit import build_model, checkpoint_model_config, checkpoint_state_dict

    rows: list[dict[str, Any]] = torch.load(args.data, map_location="cpu", weights_only=False)
    row = find_row(rows, args.row_id)
    actmap = row["actmap"].float().cpu()
    if tuple(actmap.shape) != (12, 32, 128):
        raise ValueError(f"unexpected actmap shape: {tuple(actmap.shape)}")

    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint_model_config(checkpoint_payload, args.checkpoint)
    model = build_model(**model_config)
    model.load_state_dict(checkpoint_state_dict(checkpoint_payload))
    model.eval()
    with torch.no_grad():
        logits = model(actmap.unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(probs.argmax().item())

    flat = actmap.flatten()
    clip = torch.quantile(flat, torch.tensor([0.02, 0.98]))
    lo = float(clip[0].item())
    hi = float(clip[1].item())
    normalized = torch.clamp((actmap - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    quantized = (normalized * 255.0).round().to(torch.uint8).contiguous()
    encoded = base64.b64encode(quantized.numpy().tobytes()).decode("ascii")

    payload = {
        "id": int(row["id"]),
        "query": row["query"],
        "label": row["label"],
        "textLmRoute": find_text_route(args.benchmark, args.row_id),
        "actmapRoute": LABEL_ORDER[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probabilities": {label: float(probs[index].item()) for index, label in enumerate(LABEL_ORDER)},
        "logits": {label: float(logits[index].item()) for index, label in enumerate(LABEL_ORDER)},
        "actmapShape": list(actmap.shape),
        "model": row["model"],
        "generatedTokenCount": int(row["generated_token_count"]),
        "captureMode": row["capture_mode"],
        "source": {
            "data": str(args.data),
            "checkpoint": str(args.checkpoint),
            "benchmark": str(args.benchmark),
            "split": "test",
        },
        "clip": {
            "p02": lo,
            "p98": hi,
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
            "mean": float(flat.mean().item()),
            "std": float(flat.std().item()),
        },
        "heatmap": {
            "encoding": "base64",
            "dtype": "uint8",
            "layout": "channel-major",
            "shape": list(actmap.shape),
            "data": encoded,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.ACTMAP_DEMO_DATA = "
        + json.dumps(payload, indent=2, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} for id={payload['id']} "
        f"text_lm={payload['textLmRoute']} "
        f"actmap={payload['actmapRoute']} "
        f"confidence={payload['confidence']:.4f}"
    )


if __name__ == "__main__":
    main()
