"""Evaluate saved case-name checkpoints with a source-grounded span-length cap.

No training occurs. Source run configurations and split hashes must match the
current dataset. Dev is scored before test; the output records why the cap was
chosen and keeps the original checkpoint metrics for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

from experiments.case_name.train import (
    ARCHITECTURES,
    SpanHead,
    _encode_example,
    _extract_features,
    _install_lora,
    _predict_all,
    _read_records,
    _restore_lora,
    _write_json,
    _write_jsonl,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_config(run: Path, data_dir: Path) -> dict[str, Any]:
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for split in ("train", "dev", "test"):
        actual = _digest(data_dir / f"{split}.jsonl")
        if config["dataset"][split]["sha256"] != actual:
            raise ValueError(f"{run}: saved {split} hash differs from current data")
    saved_manifest = config.get("dataset_manifest_sha256")
    if saved_manifest is not None and saved_manifest != _digest(data_dir / "manifest.json"):
        raise ValueError(f"{run}: saved manifest hash differs from current data")
    return config


def _architectures(run: Path) -> list[str]:
    return [name for name in ARCHITECTURES if (run / name / "head.pt").is_file()]


def _evaluate_run(
    run: Path,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    cap: int,
    device: torch.device,
) -> dict[str, Any]:
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise ValueError(f"{model_id}: fast tokenizer with EOS required")
    hyper = config["hyperparameters"]
    max_tokens = hyper["max_tokens"]
    max_span_tokens = hyper["max_span_tokens"]
    feature_batch_size = hyper["feature_batch_size"]
    examples = {
        split: [
            _encode_example(tokenizer, row, max_tokens, max_span_tokens)
            for row in _read_records(data_dir / f"{split}.jsonl", None)
        ]
        for split in ("dev", "test")
    }
    model = AutoModel.from_pretrained(model_id, local_files_only=True, dtype=torch.float32).to(device)
    model.requires_grad_(False)
    architectures = _architectures(run)
    if not architectures:
        raise ValueError(f"{run}: no saved case-name heads")
    frozen_features = (
        {
            split: _extract_features(model, rows, device, tokenizer.eos_token_id, feature_batch_size)
            for split, rows in examples.items()
        }
        if any(name in architectures for name in ("token", "latent"))
        else None
    )
    results: dict[str, Any] = {}
    for architecture in architectures:
        if architecture == "latent-summary":
            vector = torch.load(
                run / architecture / "summary_embedding.pt", map_location="cpu", weights_only=True
            ).to(device)
            features = {
                split: _extract_features(
                    model, rows, device, tokenizer.eos_token_id, feature_batch_size, vector
                )
                for split, rows in examples.items()
            }
        elif architecture == "latent-lora":
            _install_lora(
                model,
                hyper["lora_rank"],
                hyper["lora_alpha"],
                hyper["lora_dropout"],
            )
            adapter = torch.load(run / architecture / "lora_qv.pt", map_location="cpu", weights_only=True)
            _restore_lora(model, adapter)
            features = {
                split: _extract_features(model, rows, device, tokenizer.eos_token_id, feature_batch_size)
                for split, rows in examples.items()
            }
        else:
            assert frozen_features is not None
            features = frozen_features
        head = SpanHead(model.config.hidden_size, latent=architecture != "token").to(device)
        head.load_state_dict(
            torch.load(run / architecture / "head.pt", map_location="cpu", weights_only=True)
        )
        original_metrics = json.loads((run / architecture / "metrics.json").read_text(encoding="utf-8"))
        split_results: dict[str, Any] = {}
        for split in ("dev", "test"):
            uncapped, original_predictions = _predict_all(head, features[split], device, max_span_tokens)
            if uncapped["exact"] != original_metrics[split]["exact"]:
                raise ValueError(f"{run}/{architecture}/{split}: checkpoint does not reproduce saved result")
            capped, capped_predictions = _predict_all(
                head, features[split], device, max_span_tokens, max_span_chars=cap
            )
            changes = sum(
                a["predicted_span"] != b["predicted_span"]
                for a, b in zip(original_predictions, capped_predictions, strict=True)
            )
            _write_jsonl(output_dir / f"{architecture}-{split}-cap{cap}.jsonl", capped_predictions)
            split_results[split] = {
                "source_uncapped": uncapped,
                "capped": capped,
                "prediction_changes": changes,
            }
        results[architecture] = {
            "source_run": str(run.resolve()),
            "checkpoint": str((run / architecture / "head.pt").resolve()),
            "splits": split_results,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-span-chars", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.max_span_chars < 1:
        parser.error("--max-span-chars must be positive")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    runs = [path.resolve() for path in args.source_run]
    configs = [_run_config(run, args.data_dir) for run in runs]
    if len({config["model_id"] for config in configs}) != 1:
        raise ValueError("All source runs must use the same model")
    train = _read_records(args.data_dir / "train.jsonl", None)
    train_max_length = max(
        item["case_name"]["end"] - item["case_name"]["start"]
        for item in train
        if item["case_name"] is not None
    )
    if args.max_span_chars < train_max_length + 2:
        raise ValueError("Cap must exceed the largest training name plus token-boundary margin")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    results: dict[str, Any] = {}
    for run, config in zip(runs, configs, strict=True):
        for name in _architectures(run):
            if name in seen:
                raise ValueError(f"Architecture {name} occurs in multiple source runs")
            seen.add(name)
        results.update(
            _evaluate_run(run, config, args.data_dir, args.output_dir, args.max_span_chars, device)
        )
    report = {
        "source_runs": [str(run) for run in runs],
        "model_id": configs[0]["model_id"],
        "base_model_revision": getattr(
            AutoConfig.from_pretrained(configs[0]["model_id"], local_files_only=True),
            "_commit_hash",
            None,
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "device": str(device),
        "max_span_chars": args.max_span_chars,
        "train_max_gold_chars": train_max_length,
        "selection_rationale": (
            "The cap exceeds every training case name. "
            + ("A 100-character cap matches the existing rule bound. " if args.max_span_chars == 100 else "")
            + "Development exact span scores were checked before test was evaluated. "
            "Saved checkpoints were reused without retraining."
        ),
        "split_sha256": {
            split: _digest(args.data_dir / f"{split}.jsonl") for split in ("train", "dev", "test")
        },
        "manifest_sha256": _digest(args.data_dir / "manifest.json"),
        "architectures": results,
    }
    _write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                architecture: {
                    split: values["splits"][split]["capped"]["exact_span"] for split in ("dev", "test")
                }
                for architecture, values in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
