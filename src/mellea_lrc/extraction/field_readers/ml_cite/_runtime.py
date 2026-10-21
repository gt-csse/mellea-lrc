# Vendored from gt-csse/ML-cite at revision
# 881db5474fafe756474034ef6bf9c86a39e9630c (ml_cite.py).
# The inspected runtime is packaged locally; no remote Python code is executed.
"""Gold-independent input encoding and local inference for ML-cite checkpoints."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizerFast

FIELDS = ("case_name", "court", "date", "pin_cite")
OPEN_LOCATOR = "\n<target_locator>\n"
CLOSE_LOCATOR = "\n</target_locator>\n"
MAX_SPAN_TOKENS = 64  # Training representability policy; never an inference cap.
PRESENCE_THRESHOLD = 0.5  # Raw logit, not a sigmoid probability.


class RepresentationError(ValueError):
    """Valid source input that this token-span model cannot represent."""


@dataclass
class ModelInput:
    input_ids: list[int]
    positions: list[int]
    offsets: list[tuple[int, int]]


def trim_span(context: str, start: int, end: int) -> tuple[int, int]:
    while start < end and (context[start].isspace() or context[start] in "-."):
        start += 1
    while end > start and context[end - 1] in " ,\t\r\n":
        end -= 1
    return start, end


def encode_input(tokenizer, record: dict, max_tokens: int) -> ModelInput:
    """Read only context_text and locator; retain every source-token offset."""
    context, locator = record.get("context_text"), record.get("locator")
    if not isinstance(context, str):
        raise ValueError("context_text must be a string")
    if (
        not isinstance(locator, dict)
        or type(locator.get("start")) is not int
        or type(locator.get("end")) is not int
        or not 0 <= locator["start"] < locator["end"] <= len(context)
        or context[locator["start"] : locator["end"]] != locator.get("quote")
    ):
        raise ValueError("locator: invalid offsets or source quote")
    if not tokenizer.is_fast or any(
        x is None for x in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
    ):
        raise ValueError("a fast BERT tokenizer with native CLS/SEP/PAD is required")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    left, right = locator["start"], locator["end"]
    ids, offsets = [tokenizer.cls_token_id], [None]
    for text, base in [
        (context[:left], 0),
        (OPEN_LOCATOR, None),
        (context[left:right], left),
        (CLOSE_LOCATOR, None),
        (context[right:], right),
    ]:
        part = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids, token_offsets = part["input_ids"], part["offset_mapping"]
        if len(token_ids) != len(token_offsets):
            raise ValueError("tokenizer returned mismatched IDs and offsets")
        for token_id, offset in zip(token_ids, token_offsets):
            if (
                type(token_id) is not int
                or token_id < 0
                or len(offset) != 2
                or any(type(value) is not int for value in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text)
            ):
                raise ValueError("tokenizer returned invalid token IDs or offsets")
        ids.extend(token_ids)
        offsets.extend((base + a, base + b) if base is not None else None for a, b in token_offsets)
    ids.append(tokenizer.sep_token_id)
    offsets.append(None)
    if len(ids) > max_tokens:
        raise RepresentationError(
            f"input has {len(ids)} tokens; limit is {max_tokens}; no truncation applied"
        )
    positions = [i for i, offset in enumerate(offsets) if offset is not None and offset[0] < offset[1]]
    if not positions:
        raise ValueError("no source tokens")
    return ModelInput(ids, positions, [offsets[i] for i in positions])


class SpanHeads(nn.Module):
    """Four checkpoint-compatible bilinear heads, queried by native CLS."""

    def __init__(self, hidden_size: int, rank: int = 64):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.ModuleDict(
                {
                    f"{kind}_{boundary}": nn.Linear(hidden_size, rank, bias=False)
                    for kind in ("query", "key")
                    for boundary in ("start", "end", "presence")
                }
            )
            for _ in FIELDS
        )
        self.scale = 1 / math.sqrt(rank)

    def forward(self, tokens, query):
        tokens = F.layer_norm(tokens, (tokens.shape[-1],))
        query = F.layer_norm(query, (query.shape[-1],))
        return [
            tuple(
                (head[f"key_{boundary}"](tokens) * head[f"query_{boundary}"](query)).sum(-1) * self.scale
                for boundary in ("start", "end", "presence")
            )
            for head in self.heads
        ]


def batch_outputs(encoder, heads, batch, device, pad_id):
    ids = torch.full(
        (len(batch), max(len(e.input_ids) for e in batch)), pad_id, dtype=torch.long, device=device
    )
    mask = torch.zeros_like(ids)
    for i, example in enumerate(batch):
        ids[i, : len(example.input_ids)] = torch.tensor(example.input_ids, device=device)
        mask[i, : len(example.input_ids)] = 1
    hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return [heads(hidden[i, example.positions], hidden[i, 0]) for i, example in enumerate(batch)]


def _scores(values, label: str) -> list[float]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{label}: scores must be one-dimensional")
        values = values.detach().cpu().tolist()
    try:
        scores = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}: expected finite score values") from error
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{label}: expected nonempty finite scores")
    return scores


def best_ordered_pair(start_scores, end_scores) -> tuple[int, int]:
    """Linear maximum of start[i] + end[j], i <= j; earliest i, then j ties."""
    starts, ends = _scores(start_scores, "start"), _scores(end_scores, "end")
    if len(starts) != len(ends):
        raise ValueError("start and end score counts differ")
    prefix, pair, best = 0, (0, 0), starts[0] + ends[0]
    for end in range(len(ends)):
        if starts[end] > starts[prefix]:
            prefix = end
        score, candidate = starts[prefix] + ends[end], (prefix, end)
        if not math.isfinite(score):
            raise ValueError("ordered boundary score overflowed")
        if score > best or (score == best and candidate < pair):
            best, pair = score, candidate
    return pair


@dataclass(frozen=True)
class FieldPrediction:
    present: bool
    span: dict | None
    rescanned: bool = False


def _threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("presence threshold must be finite")
    return float(value)


def decode_outputs(
    outputs, example: ModelInput, context: str, presence_threshold: float = PRESENCE_THRESHOLD
) -> dict[str, FieldPrediction]:
    threshold = _threshold(presence_threshold)
    count = len(example.positions)
    if (
        not isinstance(context, str)
        or count == 0
        or len(example.offsets) != count
        or not isinstance(outputs, (list, tuple))
        or len(outputs) != len(FIELDS)
    ):
        raise ValueError("decoder inputs, offsets or field counts are invalid")
    for offset in example.offsets:
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
            or not 0 <= offset[0] < offset[1] <= len(context)
        ):
            raise ValueError("candidate source offsets are invalid")
    predictions = {}
    for field, values in zip(FIELDS, outputs):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"{field}: expected start, end and presence scores")
        for scores in values:
            if (
                not isinstance(scores, torch.Tensor)
                or scores.ndim != 1
                or scores.numel() != count
                or not scores.is_floating_point()
                or not torch.isfinite(scores).all().item()
            ):
                raise ValueError(f"{field}: expected finite one-dimensional candidate scores")
        start, end, presence = values
        logit = torch.logsumexp(presence, dim=-1).item()
        if not math.isfinite(logit):
            raise ValueError(f"{field}: presence logit is not finite")
        if logit < threshold:
            predictions[field] = FieldPrediction(False, None)
            continue
        first, last = start.argmax().item(), end.argmax().item()
        rescanned = first > last
        if rescanned:
            first, last = best_ordered_pair(start, end)
        left, right = trim_span(context, example.offsets[first][0], example.offsets[last][1])
        span = {"start": left, "end": right, "quote": context[left:right]} if left < right else None
        predictions[field] = FieldPrediction(True, span, rescanned)
    return predictions


@dataclass
class Checkpoint:
    encoder: object
    heads: SpanHeads
    tokenizer: object
    metadata: dict
    device: torch.device

    def encode(self, record: dict) -> ModelInput:
        return encode_input(self.tokenizer, record, self.encoder.config.max_position_embeddings)

    @torch.inference_mode()
    def predict(self, examples: list[ModelInput], contexts: list[str]) -> list[dict[str, FieldPrediction]]:
        if len(examples) != len(contexts):
            raise ValueError("example and context counts differ")
        if not examples:
            return []
        self.encoder.eval()
        self.heads.eval()
        outputs = batch_outputs(self.encoder, self.heads, examples, self.device, self.tokenizer.pad_token_id)
        threshold = _threshold(self.metadata["presence_logit_threshold"])
        return [
            decode_outputs(values, example, context, threshold)
            for values, example, context in zip(outputs, examples, contexts)
        ]


def load_checkpoint(run_dir: Path, device="auto") -> Checkpoint:
    """Load a complete local release variant without a base-model lookup."""
    run_dir = Path(run_dir).resolve()
    for name in ("training.json", "heads.safetensors"):
        if not (run_dir / name).is_file():
            raise ValueError(f"missing checkpoint file: {name}")
    for name in ("bert", "tokenizer"):
        if not (run_dir / name).is_dir():
            raise ValueError(f"missing checkpoint directory: {name}")
    metadata = json.loads((run_dir / "training.json").read_bytes())
    if not isinstance(metadata, dict) or metadata.get("fields") != list(FIELDS):
        raise ValueError("checkpoint field inventory differs from the four extraction fields")
    rank = metadata.get("query_rank")
    if type(rank) is not int or rank < 1:
        raise ValueError("checkpoint query_rank must be a positive integer")
    _threshold(metadata.get("presence_logit_threshold"))
    target = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    encoder = BertModel.from_pretrained(
        run_dir / "bert",
        add_pooling_layer=False,
        local_files_only=True,
        weights_only=True,
        use_safetensors=True,
    )
    if encoder.config.model_type != "bert":
        raise ValueError("checkpoint encoder must be BERT")
    tokenizer = BertTokenizerFast.from_pretrained(run_dir / "tokenizer", local_files_only=True)
    if not tokenizer.is_fast or any(
        value is None for value in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
    ):
        raise ValueError("checkpoint requires a fast BERT tokenizer with native CLS/SEP/PAD")
    heads = SpanHeads(encoder.config.hidden_size, rank)
    heads.load_state_dict(load_file(str(run_dir / "heads.safetensors"), device="cpu"), strict=True)
    encoder.to(target).eval()
    heads.to(target).eval()
    return Checkpoint(encoder, heads, tokenizer, metadata, target)


def load_model(
    repo_id: str | Path = "gt-csse/ML-cite",
    *,
    variant: str = "ML-cite-Bert-augmented",
    revision: str | None = None,
    device="auto",
    local_files_only: bool = False,
) -> Checkpoint:
    """Download one named variant, or load it from an existing local repo root."""
    if not isinstance(variant, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", variant) is None:
        raise ValueError("variant must be a single folder name without path traversal")
    candidate = Path(repo_id).expanduser()
    if candidate.is_dir():
        root = candidate.resolve()
    elif candidate.exists():
        raise ValueError("local repository root must be a directory")
    else:
        root = Path(
            snapshot_download(
                repo_id=str(repo_id),
                repo_type="model",
                revision=revision,
                allow_patterns=[f"{variant}/*"],
                local_files_only=local_files_only,
            )
        ).resolve()
    run_dir = (root / variant).resolve()
    if run_dir.parent != root:
        raise ValueError("variant must resolve to a single folder within the repository root")
    return load_checkpoint(run_dir, device=device)
