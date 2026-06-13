from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("ANSWER", "VERIFY", "ESCALATE")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in LABEL_TO_ID:
                raise ValueError(f"{path}:{line_number}: unknown label {row.get('label')!r}")
            rows.append(row)
    return rows


def load_split(path: Path) -> dict[str, list[int]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "train": [int(idx) for idx in payload["train"]],
        "val": [int(idx) for idx in payload["val"]],
        "test": [int(idx) for idx in payload["test"]],
    }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def features(text: str, *, max_ngram: int, binary: bool) -> Counter[str]:
    tokens = tokenize(text)
    feats: Counter[str] = Counter()
    for n in range(1, max_ngram + 1):
        if len(tokens) < n:
            continue
        for start in range(0, len(tokens) - n + 1):
            key = " ".join(tokens[start : start + n])
            feats[key] += 1
    if binary:
        return Counter({key: 1 for key in feats})
    return feats


class NgramNaiveBayes:
    def __init__(self, *, max_ngram: int, alpha: float, binary: bool):
        self.max_ngram = max_ngram
        self.alpha = alpha
        self.binary = binary
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.default_log_likelihood: dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, rows: list[dict[str, Any]]) -> None:
        class_doc_counts = Counter(row["label"] for row in rows)
        class_feature_counts: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        class_total_features = Counter()

        for row in rows:
            label = row["label"]
            row_features = features(row["query"], max_ngram=self.max_ngram, binary=self.binary)
            class_feature_counts[label].update(row_features)
            class_total_features[label] += sum(row_features.values())
            self.vocab.update(row_features)

        total_docs = sum(class_doc_counts.values())
        vocab_size = max(len(self.vocab), 1)

        self.log_prior = {
            label: math.log((class_doc_counts[label] + self.alpha) / (total_docs + self.alpha * len(LABELS)))
            for label in LABELS
        }
        self.log_likelihood = {}
        self.default_log_likelihood = {}
        for label in LABELS:
            denom = class_total_features[label] + self.alpha * vocab_size
            self.default_log_likelihood[label] = math.log(self.alpha / denom)
            self.log_likelihood[label] = {
                token: math.log((count + self.alpha) / denom)
                for token, count in class_feature_counts[label].items()
            }

    def scores(self, text: str) -> dict[str, float]:
        row_features = features(text, max_ngram=self.max_ngram, binary=self.binary)
        scores = dict(self.log_prior)
        for label in LABELS:
            likelihoods = self.log_likelihood[label]
            default = self.default_log_likelihood[label]
            for token, count in row_features.items():
                scores[label] += count * likelihoods.get(token, default)
        return scores

    def predict(self, text: str) -> str:
        scores = self.scores(text)
        return max(LABELS, key=lambda label: scores[label])


def confusion_matrix(labels: list[str], preds: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for label, pred in zip(labels, preds):
        matrix[LABEL_TO_ID[label]][LABEL_TO_ID[pred]] += 1
    return matrix


def metrics(labels: list[str], preds: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(labels, preds)
    correct = sum(matrix[idx][idx] for idx in range(len(LABELS)))
    total = max(len(labels), 1)
    per_class = {}
    f1s: list[float] = []
    for label in LABELS:
        idx = LABEL_TO_ID[label]
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(LABELS))) - tp
        fn = sum(matrix[idx][col] for col in range(len(LABELS))) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1s) / len(f1s),
        "confusion": matrix,
        "per_class": per_class,
    }


def evaluate(model: NgramNaiveBayes, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    labels = [row["label"] for row in rows]
    preds = [model.predict(row["query"]) for row in rows]
    mistakes = [
        {
            "id": str(row["id"]),
            "query": row["query"],
            "label": label,
            "pred": pred,
        }
        for row, label, pred in zip(rows, labels, preds)
        if label != pred
    ]
    return metrics(labels, preds), mistakes


def rows_for_indices(rows: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    max_idx = max(indices) if indices else -1
    if max_idx >= len(rows):
        raise ValueError(f"Split references row {max_idx}, but dataset only has {len(rows)} rows")
    return [rows[index] for index in indices]


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {label: counts[label] for label in LABELS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    rows = load_jsonl(args.data_jsonl)
    split = load_split(args.split_file)

    train_rows = rows_for_indices(rows, split["train"])
    val_rows = rows_for_indices(rows, split["val"])
    test_rows = rows_for_indices(rows, split["test"])

    candidates = []
    for max_ngram in args.ngram_max_values:
        for alpha in args.alpha_values:
            for binary in args.binary_values:
                model = NgramNaiveBayes(max_ngram=max_ngram, alpha=alpha, binary=binary)
                model.fit(train_rows)
                val_metrics, _ = evaluate(model, val_rows)
                candidates.append(
                    {
                        "max_ngram": max_ngram,
                        "alpha": alpha,
                        "binary": binary,
                        "val_accuracy": val_metrics["accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                    }
                )

    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    best = candidates[0]
    model = NgramNaiveBayes(
        max_ngram=int(best["max_ngram"]),
        alpha=float(best["alpha"]),
        binary=bool(best["binary"]),
    )
    model.fit(train_rows)
    test_metrics, mistakes = evaluate(model, test_rows)

    result: dict[str, Any] = {
        "method": "text_ngram_naive_bayes",
        "data_jsonl": str(args.data_jsonl),
        "split_file": str(args.split_file),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "label_counts": {
            "train": label_counts(train_rows),
            "val": label_counts(val_rows),
            "test": label_counts(test_rows),
        },
        "best_hyperparams": {
            "max_ngram": best["max_ngram"],
            "alpha": best["alpha"],
            "binary": best["binary"],
        },
        "val_candidates": candidates,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion": test_metrics["confusion"],
        "test_per_class": test_metrics["per_class"],
        "mistake_count": len(mistakes),
        "sample_mistakes": mistakes[: args.max_mistakes],
        "elapsed_seconds": time.perf_counter() - started,
    }

    if args.actmap_results and args.actmap_results.exists():
        with args.actmap_results.open(encoding="utf-8") as handle:
            actmap = json.load(handle)
        result["actmap_comparison"] = {
            "actmap_results": str(args.actmap_results),
            "actmap_accuracy": actmap.get("test_accuracy"),
            "actmap_macro_f1": actmap.get("test_macro_f1"),
            "text_accuracy": result["test_accuracy"],
            "text_macro_f1": result["test_macro_f1"],
            "accuracy_delta_actmap_minus_text": (
                actmap.get("test_accuracy") - result["test_accuracy"]
                if isinstance(actmap.get("test_accuracy"), (int, float))
                else None
            ),
            "macro_f1_delta_actmap_minus_text": (
                actmap.get("test_macro_f1") - result["test_macro_f1"]
                if isinstance(actmap.get("test_macro_f1"), (int, float))
                else None
            ),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "text_baseline_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    summary_path = args.output / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Text Baseline Vs ActMap\n\n")
        handle.write(f"Split: `{args.split_file}`\n\n")
        handle.write("| Method | Accuracy | Macro F1 |\n")
        handle.write("| --- | ---: | ---: |\n")
        if "actmap_comparison" in result:
            comparison = result["actmap_comparison"]
            handle.write(
                f"| ActMap ViT2D | {comparison['actmap_accuracy']:.4f} | "
                f"{comparison['actmap_macro_f1']:.4f} |\n"
            )
        handle.write(
            f"| Text n-gram NB | {result['test_accuracy']:.4f} | "
            f"{result['test_macro_f1']:.4f} |\n"
        )
        handle.write("\n")
        handle.write(f"Best text hyperparameters: `{result['best_hyperparams']}`\n\n")
        handle.write(f"Text confusion rows=true cols=pred `{list(LABELS)}`:\n\n")
        handle.write(f"```json\n{json.dumps(result['test_confusion'], indent=2)}\n```\n")
        if result["sample_mistakes"]:
            handle.write("\nSample text-baseline mistakes:\n\n")
            for mistake in result["sample_mistakes"]:
                handle.write(
                    f"- id={mistake['id']} true={mistake['label']} pred={mistake['pred']}: "
                    f"{mistake['query']}\n"
                )

    print(f"Text baseline test accuracy={result['test_accuracy']:.4f}")
    print(f"Text baseline test macro_f1={result['test_macro_f1']:.4f}")
    if "actmap_comparison" in result:
        comparison = result["actmap_comparison"]
        print(
            "ActMap delta accuracy="
            f"{comparison['accuracy_delta_actmap_minus_text']:.4f} "
            "macro_f1="
            f"{comparison['macro_f1_delta_actmap_minus_text']:.4f}"
        )
    print(f"Saved results -> {results_path}")
    print(f"Saved summary -> {summary_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency-free text-only router baseline.")
    parser.add_argument("--data-jsonl", type=Path, default=Path("actmap_dataset.jsonl"))
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("runs/actmap_vit2d_incremental_1000/split_indices.json"),
    )
    parser.add_argument(
        "--actmap-results",
        type=Path,
        default=Path("runs/actmap_vit2d_incremental_1000/test_results.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("runs/text_baseline_actmap1000"))
    parser.add_argument("--ngram-max-values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--binary-values",
        type=lambda value: value.lower() in {"1", "true", "yes"},
        nargs="+",
        default=[False, True],
    )
    parser.add_argument("--max-mistakes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
