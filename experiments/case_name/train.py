"""Train small supervised case-name span extractors on locator-anchored windows.

The input is the source context with only the target locator bracketed, then a
fixed instruction and a terminal EOS token. Gold names, citation ids, and
provenance are never inserted into model input. A causal model's terminal state
can see the complete marked window and instruction; earlier token states cannot.

This is a case-name experiment, not a demonstration that the model can obey new
instructions at inference time. All three variants use the same tokenizer,
backbone, candidate positions, training spans, and constrained span decoder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCTION = (
    "\nInstruction: Mark the exact written case name associated with the marked "
    "citation locator. If no case name is written, mark nothing.\n"
)
OPEN_LOCATOR = "\n<target_locator>\n"
CLOSE_LOCATOR = "\n</target_locator>\n"
ARCHITECTURES = ("token", "latent", "latent-summary", "latent-lora")


@dataclass(slots=True)
class Example:
    record: dict[str, Any]
    input_ids: list[int]
    candidate_positions: list[int]
    candidate_offsets: list[tuple[int, int]]
    gold_start_token: int | None
    gold_end_token: int | None


@dataclass(slots=True)
class Feature:
    example: Example
    tokens: Tensor  # Only the candidate source tokens, on CPU.
    terminal: Tensor  # Terminal EOS representation, on CPU.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run build_dataset.py first")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No examples in {path}")
    return records


def _encode_segment(
    tokenizer: Any,
    text: str,
    base_offset: int | None,
    input_ids: list[int],
    source_offsets: list[tuple[int, int] | None],
) -> None:
    if not text:
        return
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids.extend(encoding["input_ids"])
    source_offsets.extend(
        (base_offset + start, base_offset + end) if base_offset is not None else None
        for start, end in encoding["offset_mapping"]
    )


def _encode_example(tokenizer: Any, record: dict[str, Any], max_tokens: int, max_span: int) -> Example:
    context = record["context_text"]
    locator = record["locator"]
    left, right = locator["start"], locator["end"]
    if not isinstance(context, str) or not 0 <= left < right <= len(context):
        raise ValueError(f"{record.get('id')}: invalid context or locator")
    if context[left:right] != locator["quote"]:
        raise ValueError(f"{record.get('id')}: locator quote does not match context")
    name = record["case_name"]
    if name is not None and (
        not 0 <= name["start"] < name["end"] <= left or context[name["start"] : name["end"]] != name["quote"]
    ):
        raise ValueError(f"{record.get('id')}: case name must be quoted before the target locator")

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The selected model needs BOS and EOS token ids")
    ids = [tokenizer.bos_token_id]
    offsets: list[tuple[int, int] | None] = [None]
    _encode_segment(tokenizer, context[:left], 0, ids, offsets)
    _encode_segment(tokenizer, OPEN_LOCATOR, None, ids, offsets)
    _encode_segment(tokenizer, context[left:right], left, ids, offsets)
    _encode_segment(tokenizer, CLOSE_LOCATOR, None, ids, offsets)
    _encode_segment(tokenizer, context[right:], right, ids, offsets)
    _encode_segment(tokenizer, INSTRUCTION, None, ids, offsets)
    ids.append(tokenizer.eos_token_id)
    offsets.append(None)
    if len(ids) > max_tokens:
        raise ValueError(f"{record.get('id')}: {len(ids)} tokens exceed --max-tokens={max_tokens}")

    # The task asks for the name before this locator. The suffix is visible to
    # the terminal representation but is not eligible as an answer.
    positions = [
        index
        for index, offset in enumerate(offsets)
        if offset is not None and 0 <= offset[0] < offset[1] <= left
    ]
    candidate_offsets = [offsets[index] for index in positions]
    if not positions:
        raise ValueError(f"{record.get('id')}: no candidate source tokens before locator")
    gold_start = gold_end = None
    if name is not None:
        start, end = name["start"], name["end"]
        gold_start = next((i for i, (a, b) in enumerate(candidate_offsets) if a <= start < b), None)
        gold_end = next((i for i, (a, b) in enumerate(candidate_offsets) if a < end <= b), None)
        if gold_start is None or gold_end is None or gold_end - gold_start + 1 > max_span:
            raise ValueError(f"{record.get('id')}: gold name cannot be represented by candidate tokens")
        mapped_positions = positions[gold_start : gold_end + 1]
        if mapped_positions != list(range(mapped_positions[0], mapped_positions[-1] + 1)):
            raise ValueError(f"{record.get('id')}: gold name is not contiguous in source tokens")
        envelope = (candidate_offsets[gold_start][0], candidate_offsets[gold_end][1])
        if _trim_span(context, *envelope) != (start, end):
            raise ValueError(
                f"{record.get('id')}: token envelope cannot reproduce exact gold span; "
                f"envelope={envelope}, gold={(start, end)}"
            )
    return Example(record, ids, positions, candidate_offsets, gold_start, gold_end)


def _trim_span(context: str, start: int, end: int) -> tuple[int, int] | None:
    """Undo leading-space and terminal-comma absorption by the GPT2 tokenizer."""
    while start < end and context[start].isspace():
        start += 1
    while end > start and context[end - 1] in " ,\t\r\n":
        end -= 1
    return (start, end) if start < end else None


def _batch_hidden(
    backbone: nn.Module,
    examples: list[Example],
    device: torch.device,
    pad_id: int,
    summary_embedding: Tensor | None = None,
) -> Tensor:
    longest = max(len(item.input_ids) for item in examples)
    ids = torch.full((len(examples), longest), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, example in enumerate(examples):
        length = len(example.input_ids)
        ids[i, :length] = torch.tensor(example.input_ids, dtype=torch.long, device=device)
        mask[i, :length] = 1
    if summary_embedding is None:
        return backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    # A virtual terminal token: only its input embedding is trainable. Its
    # hidden state still reads the whole marked context through the frozen LM.
    embeds = backbone.get_input_embeddings()(ids)
    terminal_mask = torch.zeros((*ids.shape, 1), device=device, dtype=embeds.dtype)
    terminal_mask[torch.arange(len(examples), device=device), [len(x.input_ids) - 1 for x in examples]] = 1
    embeds = embeds * (1 - terminal_mask) + summary_embedding.reshape(1, 1, -1) * terminal_mask
    return backbone(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state


def _features_from_hidden(examples: list[Example], hidden: Tensor) -> list[Feature]:
    return [
        Feature(
            example=example,
            tokens=hidden[i, example.candidate_positions].detach().float().cpu(),
            terminal=hidden[i, len(example.input_ids) - 1].detach().float().cpu(),
        )
        for i, example in enumerate(examples)
    ]


def _extract_features(
    backbone: nn.Module,
    examples: list[Example],
    device: torch.device,
    pad_id: int,
    batch_size: int,
    summary_embedding: Tensor | None = None,
) -> list[Feature]:
    backbone.eval()
    features: list[Feature] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            hidden = _batch_hidden(backbone, batch, device, pad_id, summary_embedding)
            features.extend(_features_from_hidden(batch, hidden))
    return features


class SpanHead(nn.Module):
    """Two boundary scores and one null option, with optional latent bilinear queries."""

    def __init__(self, hidden_size: int, latent: bool, query_rank: int = 64) -> None:
        super().__init__()
        self.latent = latent
        if latent:
            self.query_start = nn.Linear(hidden_size, query_rank, bias=False)
            self.query_end = nn.Linear(hidden_size, query_rank, bias=False)
            self.key_start = nn.Linear(hidden_size, query_rank, bias=False)
            self.key_end = nn.Linear(hidden_size, query_rank, bias=False)
            self.scale = 1 / math.sqrt(query_rank)
        else:
            self.start = nn.Linear(hidden_size, 1)
            self.end = nn.Linear(hidden_size, 1)
        self.null_start = nn.Parameter(torch.tensor(-2.0))
        self.null_end = nn.Parameter(torch.tensor(-2.0))

    def forward(self, tokens: Tensor, terminal: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        if self.latent:
            if terminal is None:
                raise ValueError("Latent head requires a terminal representation")
            terminal = F.layer_norm(terminal, (terminal.shape[-1],))
            starts = (self.key_start(tokens) * self.query_start(terminal)).sum(-1) * self.scale
            ends = (self.key_end(tokens) * self.query_end(terminal)).sum(-1) * self.scale
        else:
            starts = self.start(tokens).squeeze(-1)
            ends = self.end(tokens).squeeze(-1)
        return starts, ends, self.null_start, self.null_end


class LoRALinear(nn.Module):
    """A trainable low-rank update around a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def _install_lora(backbone: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    if not hasattr(backbone, "layers"):
        raise ValueError("LoRA setup expects a Llama-style backbone with layers")
    count = 0
    for layer in backbone.layers:
        attention = layer.self_attn
        for name in ("q_proj", "v_proj"):
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected linear {name}, got {type(base).__name__}")
            update = LoRALinear(base, rank, alpha, dropout).to(
                device=base.weight.device, dtype=base.weight.dtype
            )
            setattr(attention, name, update)
            count += 1
    return count


def _loss(head: SpanHead, tokens: Tensor, terminal: Tensor, example: Example) -> Tensor:
    starts, ends, null_start, null_end = head(tokens, terminal)
    n = starts.shape[0]
    target_start = n if example.gold_start_token is None else example.gold_start_token
    target_end = n if example.gold_end_token is None else example.gold_end_token
    start_logits = torch.cat((starts, null_start[None]))[None, :]
    end_logits = torch.cat((ends, null_end[None]))[None, :]
    start_target = torch.tensor([target_start], dtype=torch.long, device=starts.device)
    end_target = torch.tensor([target_end], dtype=torch.long, device=ends.device)
    return (F.cross_entropy(start_logits, start_target) + F.cross_entropy(end_logits, end_target)) / 2


def _decode(
    head: SpanHead,
    feature: Feature,
    device: torch.device,
    max_span: int,
    terminal: Tensor,
    max_span_chars: int | None = None,
) -> tuple[int, int] | None:
    tokens = feature.tokens.to(device)
    starts, ends, null_start, null_end = head(tokens, terminal)
    n = len(feature.example.candidate_positions)
    positions = torch.arange(n, device=device)
    valid = (positions[:, None] <= positions[None, :]) & (positions[None, :] - positions[:, None] < max_span)
    if max_span_chars is not None:
        offsets = feature.example.candidate_offsets
        starts_char = torch.tensor([start for start, _ in offsets], device=device)
        ends_char = torch.tensor([end for _, end in offsets], device=device)
        valid &= ends_char[None, :] - starts_char[:, None] <= max_span_chars
    scores = (starts[:, None] + ends[None, :]).masked_fill(~valid, -torch.inf)
    winner = int(scores.flatten().argmax().item())
    if float((null_start + null_end).item()) >= float(scores.flatten()[winner].item()):
        return None
    first, last = divmod(winner, n)
    offsets = feature.example.candidate_offsets
    start, end = offsets[first][0], offsets[last][1]
    context = feature.example.record["context_text"]
    return _trim_span(context, start, end)


def _score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "records": len(predictions),
        "gold_names": 0,
        "predicted_names": 0,
        "exact": 0,
        "overlap": 0,
        "correct_absence": 0,
    }
    char_iou_sum = 0.0
    for item in predictions:
        gold, predicted = item["gold_span"], item["predicted_span"]
        counts["gold_names"] += int(gold is not None)
        counts["predicted_names"] += int(predicted is not None)
        if gold is None:
            counts["correct_absence"] += int(predicted is None)
        elif predicted is not None:
            counts["exact"] += int((gold["start"], gold["end"]) == (predicted["start"], predicted["end"]))
            intersection = max(0, min(gold["end"], predicted["end"]) - max(gold["start"], predicted["start"]))
            counts["overlap"] += int(intersection > 0)
            if intersection:
                union = max(gold["end"], predicted["end"]) - min(gold["start"], predicted["start"])
                char_iou_sum += intersection / union

    def _prf(correct: int) -> dict[str, float | None]:
        precision = correct / counts["predicted_names"] if counts["predicted_names"] else None
        recall = correct / counts["gold_names"] if counts["gold_names"] else None
        f1 = (
            2 * correct / (counts["predicted_names"] + counts["gold_names"])
            if (counts["predicted_names"] + counts["gold_names"])
            else None
        )
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        **counts,
        "exact_span": _prf(counts["exact"]),
        "overlapping_span": _prf(counts["overlap"]),
        "mean_char_iou_per_gold": char_iou_sum / counts["gold_names"] if counts["gold_names"] else None,
        "correct_absence_rate": counts["correct_absence"] / (counts["records"] - counts["gold_names"])
        if counts["records"] > counts["gold_names"]
        else None,
    }


def _predict_all(
    head: SpanHead,
    features: list[Feature],
    device: torch.device,
    max_span: int,
    terminal_mode: str = "actual",
    max_span_chars: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head.eval()
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for i, feature in enumerate(features):
            if terminal_mode == "zero":
                terminal = torch.zeros_like(feature.terminal, device=device)
            elif terminal_mode == "rotated":
                terminal = features[(i + 1) % len(features)].terminal.to(device)
            elif terminal_mode == "cross_group":
                # Adjacent examples often cite the same case in one filing.
                # Use a different lawsuit group whenever the split has one.
                current_group = feature.example.record.get("case_group")
                alternate = next(
                    (
                        features[(i + step) % len(features)]
                        for step in range(1, len(features))
                        if features[(i + step) % len(features)].example.record.get("case_group")
                        != current_group
                    ),
                    features[(i + 1) % len(features)],
                )
                terminal = alternate.terminal.to(device)
            elif terminal_mode == "actual":
                terminal = feature.terminal.to(device)
            else:
                raise ValueError(f"Unknown terminal mode: {terminal_mode}")
            span = _decode(head, feature, device, max_span, terminal, max_span_chars)
            record = feature.example.record
            gold = record["case_name"]
            predictions.append(
                {
                    "id": record["id"],
                    "document": record.get("document"),
                    "case_group": record.get("case_group"),
                    "citation_kind": record.get("citation_kind"),
                    "gold_span": {"start": gold["start"], "end": gold["end"]} if gold else None,
                    "gold_text": gold["quote"] if gold else None,
                    "predicted_span": {"start": span[0], "end": span[1]} if span else None,
                    "predicted_text": record["context_text"][span[0] : span[1]] if span else None,
                }
            )
    return _score_predictions(predictions), predictions


def _snapshot_lora(backbone: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in backbone.named_parameters()
        if ".lora_a." in name or ".lora_b." in name
    }


def _restore_lora(backbone: nn.Module, state: dict[str, Tensor]) -> None:
    named = dict(backbone.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value.to(named[name].device))


def _train_frozen(
    head: SpanHead,
    train: list[Feature],
    dev: list[Feature],
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_span: int,
    max_span_chars: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    best_key = (-1.0, -1.0)
    best_state: dict[str, Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = [train[i] for i in order[start : start + batch_size]]
            loss = torch.stack(
                [
                    _loss(head, item.tokens.to(device), item.terminal.to(device), item.example)
                    for item in batch
                ]
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics, _ = _predict_all(head, dev, device, max_span, max_span_chars=max_span_chars)
        key = (
            dev_metrics["exact_span"]["f1"] or 0.0,
            dev_metrics["overlapping_span"]["f1"] or 0.0,
        )
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev": dev_metrics})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} dev_exact_f1={key[0]:.4f}", flush=True)
        if key > best_key:
            best_key = key
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()})
    if best_state is not None:
        head.load_state_dict(best_state)
    return history


def _train_lora(
    backbone: nn.Module,
    head: SpanHead,
    train: list[Example],
    dev: list[Example],
    device: torch.device,
    pad_id: int,
    *,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_span: int,
    max_span_chars: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    parameters.extend(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    rng = random.Random(seed)
    best_key = (-1.0, -1.0)
    best_head: dict[str, Tensor] | None = None
    best_adapter: dict[str, Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        backbone.train()
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            hidden = _batch_hidden(backbone, batch, device, pad_id)
            loss = torch.stack(
                [
                    _loss(
                        head,
                        hidden[i, example.candidate_positions],
                        hidden[i, len(example.input_ids) - 1],
                        example,
                    )
                    for i, example in enumerate(batch)
                ]
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_features = _extract_features(backbone, dev, device, pad_id, eval_batch_size)
        dev_metrics, _ = _predict_all(head, dev_features, device, max_span, max_span_chars=max_span_chars)
        key = (
            dev_metrics["exact_span"]["f1"] or 0.0,
            dev_metrics["overlapping_span"]["f1"] or 0.0,
        )
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev": dev_metrics})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} dev_exact_f1={key[0]:.4f}", flush=True)
        if key > best_key:
            best_key = key
            best_head = copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()})
            best_adapter = _snapshot_lora(backbone)
    if best_head is not None and best_adapter is not None:
        head.load_state_dict(best_head)
        _restore_lora(backbone, best_adapter)
    return history


def _train_summary(
    backbone: nn.Module,
    head: SpanHead,
    summary_embedding: nn.Parameter,
    train: list[Example],
    dev: list[Example],
    device: torch.device,
    pad_id: int,
    *,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_span: int,
    max_span_chars: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "weight_decay": weight_decay},
            {"params": [summary_embedding], "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )
    rng = random.Random(seed)
    best_key = (-1.0, -1.0)
    best_head: dict[str, Tensor] | None = None
    best_summary: Tensor | None = None
    history = []
    for epoch in range(1, epochs + 1):
        backbone.eval()
        head.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [train[i] for i in order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            hidden = _batch_hidden(backbone, batch, device, pad_id, summary_embedding)
            loss = torch.stack(
                [
                    _loss(
                        head,
                        hidden[i, example.candidate_positions],
                        hidden[i, len(example.input_ids) - 1],
                        example,
                    )
                    for i, example in enumerate(batch)
                ]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*head.parameters(), summary_embedding], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_features = _extract_features(backbone, dev, device, pad_id, eval_batch_size, summary_embedding)
        dev_metrics, _ = _predict_all(head, dev_features, device, max_span, max_span_chars=max_span_chars)
        key = (
            dev_metrics["exact_span"]["f1"] or 0.0,
            dev_metrics["overlapping_span"]["f1"] or 0.0,
        )
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev": dev_metrics})
        print(f"epoch {epoch}: loss={history[-1]['train_loss']:.4f} dev_exact_f1={key[0]:.4f}", flush=True)
        if key > best_key:
            best_key = key
            best_head = copy.deepcopy({k: v.detach().cpu() for k, v in head.state_dict().items()})
            best_summary = summary_embedding.detach().cpu().clone()
    if best_head is not None and best_summary is not None:
        head.load_state_dict(best_head)
        with torch.no_grad():
            summary_embedding.copy_(best_summary.to(device))
    return history


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _save_result(
    architecture: str,
    output_dir: Path,
    head: SpanHead,
    backbone: nn.Module,
    dev: list[Feature],
    test: list[Feature],
    device: torch.device,
    max_span: int,
    history: list[dict[str, Any]],
    summary_embedding: Tensor | None = None,
    max_span_chars: int | None = None,
) -> dict[str, Any]:
    run_dir = output_dir / architecture
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    for split, features in (("dev", dev), ("test", test)):
        result, predictions = _predict_all(head, features, device, max_span, max_span_chars=max_span_chars)
        metrics[split] = result
        _write_jsonl(run_dir / f"predictions_{split}.jsonl", predictions)
        if head.latent:
            ablations: dict[str, Any] = {}
            for mode in ("zero", "rotated", "cross_group"):
                altered_metrics, altered_predictions = _predict_all(
                    head, features, device, max_span, mode, max_span_chars
                )
                altered_metrics["prediction_changes"] = sum(
                    original["predicted_span"] != altered["predicted_span"]
                    for original, altered in zip(predictions, altered_predictions, strict=True)
                )
                ablations[mode] = altered_metrics
                if mode == "cross_group":
                    _write_jsonl(
                        run_dir / f"predictions_{split}_ablation_cross_group.jsonl", altered_predictions
                    )
            metrics[split]["terminal_ablation"] = ablations
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "history.json", history)
    torch.save(head.state_dict(), run_dir / "head.pt")
    if architecture == "latent-lora":
        torch.save(_snapshot_lora(backbone), run_dir / "lora_qv.pt")
    if architecture == "latent-summary":
        if summary_embedding is None:
            raise ValueError("Missing learned summary embedding")
        torch.save(summary_embedding.detach().cpu(), run_dir / "summary_embedding.pt")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--architecture", choices=(*ARCHITECTURES, "core", "all"), default="all")
    parser.add_argument("--allow-download", action="store_true", help="Allow uncached model downloads")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-span-tokens", type=int, default=64)
    parser.add_argument(
        "--max-span-chars",
        type=int,
        default=None,
        help="Optional maximum raw character envelope for a decoded name",
    )
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=16)
    parser.add_argument("--lora-batch-size", type=int, default=4)
    parser.add_argument("--summary-batch-size", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--summary-epochs", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lora-lr", type=float, default=3e-4)
    parser.add_argument("--summary-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit-train", type=int, default=None, help="Smoke-test limit; recorded in config")
    parser.add_argument("--limit-dev", type=int, default=None, help="Smoke-test limit; recorded in config")
    parser.add_argument("--limit-test", type=int, default=None, help="Smoke-test limit; recorded in config")
    args = parser.parse_args()
    for key in (
        "max_tokens",
        "max_span_tokens",
        "feature_batch_size",
        "head_batch_size",
        "lora_batch_size",
        "summary_batch_size",
    ):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_span_chars is not None and args.max_span_chars < 1:
        parser.error("--max-span-chars must be positive")
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    local_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=local_only, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Character-offset alignment requires a fast tokenizer")
    if tokenizer.eos_token_id is None:
        raise ValueError("Model tokenizer has no EOS token")
    limits = {"train": args.limit_train, "dev": args.limit_dev, "test": args.limit_test}
    paths = {split: args.data_dir / f"{split}.jsonl" for split in limits}
    examples = {
        split: [
            _encode_example(tokenizer, record, args.max_tokens, args.max_span_tokens)
            for record in _read_records(paths[split], limits[split])
        ]
        for split in limits
    }
    model = AutoModel.from_pretrained(args.model_id, local_files_only=local_only, dtype=torch.float32)
    model.requires_grad_(False)
    model.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "config.json",
        {
            "model_id": args.model_id,
            "base_model_revision": getattr(model.config, "_commit_hash", None),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "device": str(device),
            "seed": args.seed,
            "architecture": args.architecture,
            "instruction": INSTRUCTION,
            "open_locator": OPEN_LOCATOR,
            "close_locator": CLOSE_LOCATOR,
            "dataset": {
                split: {"path": str(path.resolve()), "sha256": _sha256(path), "used": len(examples[split])}
                for split, path in paths.items()
            },
            "dataset_manifest_sha256": _sha256(args.data_dir / "manifest.json")
            if (args.data_dir / "manifest.json").is_file()
            else None,
            "hyperparameters": {
                key: value
                for key, value in vars(args).items()
                if key not in ("data_dir", "output_dir", "model_id", "architecture", "device")
            },
            "note": "A single fixed instruction tests case-name extraction; it cannot test instruction generalization.",
        },
    )
    if args.architecture == "core":
        selected = ("token", "latent", "latent-lora")
    elif args.architecture == "all":
        selected = ARCHITECTURES
    else:
        selected = (args.architecture,)
    frozen_features: dict[str, list[Feature]] | None = None
    if any(name in selected for name in ("token", "latent")):
        print("Extracting frozen backbone features", flush=True)
        frozen_features = {
            split: _extract_features(model, rows, device, tokenizer.eos_token_id, args.feature_batch_size)
            for split, rows in examples.items()
        }

    all_metrics: dict[str, Any] = {}
    for architecture in selected:
        print(f"Training {architecture}", flush=True)
        latent = architecture != "token"
        head = SpanHead(model.config.hidden_size, latent=latent).to(device)
        summary_embedding: nn.Parameter | None = None
        if architecture == "latent-lora":
            installed = _install_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
            print(f"Installed LoRA on {installed} q_proj/v_proj modules", flush=True)
            history = _train_lora(
                model,
                head,
                examples["train"],
                examples["dev"],
                device,
                tokenizer.eos_token_id,
                epochs=args.lora_epochs,
                batch_size=args.lora_batch_size,
                eval_batch_size=args.feature_batch_size,
                learning_rate=args.lora_lr,
                weight_decay=args.weight_decay,
                max_span=args.max_span_tokens,
                max_span_chars=args.max_span_chars,
                seed=args.seed,
            )
            result_features = {
                split: _extract_features(
                    model, examples[split], device, tokenizer.eos_token_id, args.feature_batch_size
                )
                for split in ("dev", "test")
            }
        elif architecture == "latent-summary":
            # Initialize a new trainable terminal vector from the model's EOS
            # embedding; all pretrained model parameters remain frozen.
            initial = model.get_input_embeddings().weight[tokenizer.eos_token_id].detach().clone()
            summary_embedding = nn.Parameter(initial.to(device))
            history = _train_summary(
                model,
                head,
                summary_embedding,
                examples["train"],
                examples["dev"],
                device,
                tokenizer.eos_token_id,
                epochs=args.summary_epochs,
                batch_size=args.summary_batch_size,
                eval_batch_size=args.feature_batch_size,
                learning_rate=args.summary_lr,
                weight_decay=args.weight_decay,
                max_span=args.max_span_tokens,
                max_span_chars=args.max_span_chars,
                seed=args.seed,
            )
            result_features = {
                split: _extract_features(
                    model,
                    examples[split],
                    device,
                    tokenizer.eos_token_id,
                    args.feature_batch_size,
                    summary_embedding,
                )
                for split in ("dev", "test")
            }
        else:
            assert frozen_features is not None
            history = _train_frozen(
                head,
                frozen_features["train"],
                frozen_features["dev"],
                device,
                epochs=args.head_epochs,
                batch_size=args.head_batch_size,
                learning_rate=args.head_lr,
                weight_decay=args.weight_decay,
                max_span=args.max_span_tokens,
                max_span_chars=args.max_span_chars,
                seed=args.seed,
            )
            result_features = {split: frozen_features[split] for split in ("dev", "test")}
        all_metrics[architecture] = _save_result(
            architecture,
            args.output_dir,
            head,
            model,
            result_features["dev"],
            result_features["test"],
            device,
            args.max_span_tokens,
            history,
            summary_embedding,
            args.max_span_chars,
        )
    _write_json(args.output_dir / "metrics.json", all_metrics)
    print(json.dumps({name: result["test"]["exact_span"] for name, result in all_metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
