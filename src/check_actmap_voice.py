from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_KEYS = [
    "id",
    "query",
    "label",
    "route_id",
    "model",
    "answer",
    "generated_token_count",
    "capture_mode",
    "system_prompt",
    "actmap",
]
EXPECTED_SHAPE = (12, 32, 128)
LABEL_TO_ID = {
    "ANSWER": 0,
    "VERIFY": 1,
    "ESCALATE": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check ActMap Voice .pt datasets.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--samples-per-label", type=int, default=2)
    return parser.parse_args()


def summarize(path: Path, *, samples_per_label: int) -> None:
    import torch

    rows: list[dict[str, Any]] = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n=== {path} ===")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    key_errors = [
        index for index, row in enumerate(rows) if list(row.keys()) != EXPECTED_KEYS
    ]
    labels = Counter(str(row.get("label")) for row in rows)
    models = Counter(str(row.get("model")) for row in rows)
    capture_modes = Counter(str(row.get("capture_mode")) for row in rows)
    route_errors = [
        index
        for index, row in enumerate(rows)
        if LABEL_TO_ID.get(row.get("label")) != row.get("route_id")
    ]
    duplicate_keys = [
        key
        for key, count in Counter((row.get("id"), row.get("model")) for row in rows).items()
        if count > 1
    ]

    print(f"schema_ok: {not key_errors}")
    print(f"schema_error_count: {len(key_errors)} {key_errors[:10]}")
    print(f"labels: {dict(labels)}")
    print(f"models: {dict(models)}")
    print(f"capture_modes: {dict(capture_modes)}")
    print(f"route_error_count: {len(route_errors)} {route_errors[:10]}")
    print(f"duplicate_id_model_count: {len(duplicate_keys)} {duplicate_keys[:5]}")

    actmaps = [row["actmap"] for row in rows if "actmap" in row]
    shapes = sorted({tuple(actmap.shape) for actmap in actmaps})
    dtypes = sorted({str(actmap.dtype) for actmap in actmaps})
    print(f"shapes: {shapes}")
    print(f"dtypes: {dtypes}")

    bad_shape = []
    nonfinite = []
    zeroish = []
    for index, row in enumerate(rows):
        actmap = row.get("actmap")
        if actmap is None:
            bad_shape.append(index)
            continue
        actmap_f = actmap.float()
        if tuple(actmap.shape) != EXPECTED_SHAPE:
            bad_shape.append(index)
        if not torch.isfinite(actmap_f).all().item():
            nonfinite.append(index)
        if actmap_f.std().item() < 1e-6 or actmap_f.abs().mean().item() < 1e-6:
            zeroish.append(index)

    print(f"bad_shape_count: {len(bad_shape)} {bad_shape[:10]}")
    print(f"nonfinite_count: {len(nonfinite)} {nonfinite[:10]}")
    print(f"zeroish_count: {len(zeroish)} {zeroish[:10]}")

    stacked = torch.stack([actmap.float() for actmap in actmaps])
    print(
        "global mean/std/min/max:",
        stacked.mean().item(),
        stacked.std().item(),
        stacked.min().item(),
        stacked.max().item(),
    )
    channel_mean_abs = stacked.abs().mean(dim=(0, 2, 3))
    channel_std = stacked.std(dim=(0, 2, 3))
    print("channel mean_abs:", [round(value, 6) for value in channel_mean_abs.tolist()])
    print("channel std:", [round(value, 6) for value in channel_std.tolist()])

    token_counts = [
        int(row["generated_token_count"])
        for row in rows
        if isinstance(row.get("generated_token_count"), int)
    ]
    if token_counts:
        token_counts_sorted = sorted(token_counts)
        print(
            "generated_token_count min/median/max:",
            token_counts_sorted[0],
            token_counts_sorted[len(token_counts_sorted) // 2],
            token_counts_sorted[-1],
        )

    samples_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get("label"))
        if len(samples_by_label[label]) < samples_per_label:
            samples_by_label[label].append(row)

    print("samples:")
    for label in ["ANSWER", "VERIFY", "ESCALATE"]:
        for row in samples_by_label.get(label, []):
            print(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "query": row["query"][:100],
                    "answer": row["answer"][:100],
                    "tokens": row["generated_token_count"],
                }
            )


def main() -> None:
    args = parse_args()
    for path in args.paths:
        summarize(path, samples_per_label=args.samples_per_label)


if __name__ == "__main__":
    main()
