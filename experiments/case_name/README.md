# Case-name span experiment

This experiment asks whether a small decoder model can mark the source-text span of the case name associated with a known reporter or docket locator. It is an offline research path; it does not change the production extractor.

## Data and split

`build_dataset.py` reads the primary annotation files and their filing texts, checks each source hash and quoted span, and writes one local context window per full locator. The input window contains up to 384 characters before and 128 after the locator. Character offsets in the examples remain tied to the original filing text. The model receives the window and a marker identifying the target locator; it does not receive the annotated case-name quote, filename, lawsuit group, or normalization.

The current corpus has 647 full-locator examples from 26 filings: 645 quoted case names and two explicit `not_stated` names. Related filings are kept in the same split. With seed 42, the split contains 453 training examples from 12 lawsuit groups, 97 development examples from four groups, and 97 test examples from four groups. The two no-name examples belong to one lawsuit group, which is kept in training. Development and test therefore measure span selection and boundaries, not reliable abstention.

Generate the data from the linked primary corpus:

```sh
.venv/bin/python -m experiments.case_name.build_dataset --output-dir experiments/case_name/data
```

The generated data and model runs are ignored by Git because they contain source filing text or weights. `data/manifest.json` records the window settings, group split, counts, and SHA256 fingerprints of each split. The current combined content fingerprint is `407dcfb8374c92ca0938aca5edfc2a22d3acf61838f3ad3956c01feb80a456ac`.

## Rule comparison

The rule baseline applies the existing eyecite name anchor and regular-expression fallback to the same pre-locator text supplied to the models:

```sh
.venv/bin/python -m experiments.case_name.baseline \
  --data-dir experiments/case_name/data \
  --predictions-dir experiments/case_name/results
```

| Split | Exact spans | Gold names | Predicted names | Exact recall |
| --- | ---: | ---: | ---: | ---: |
| Development | 57 | 97 | 88 | 58.8% |
| Test | 85 | 97 | 95 | 87.6% |

This comparison uses the experiment's 384-character prefix. The saved production stage score uses a 160-character, adjacent-locator-bounded prefix, so its 548/609 span precision is a different evaluation.

## Model comparison

The recorded seed-42 comparisons use the locally cached [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) backbone. Recreate their architecture order with two runs:

```sh
.venv/bin/python -m experiments.case_name.train \
  --data-dir experiments/case_name/data \
  --output-dir experiments/case_name/runs/primary-smollm2-135m-seed42 \
  --architecture core

.venv/bin/python -m experiments.case_name.train \
  --data-dir experiments/case_name/data \
  --output-dir experiments/case_name/runs/primary-smollm2-135m-summary-seed42 \
  --architecture latent-summary
```

`--architecture all` runs all four variants in one process for a new comparison. The script requires `torch` and `transformers` in the Python environment and uses the local model cache by default. Pass `--allow-download` only when a different uncached model is intended. Runs save their configuration, split fingerprints, per-example predictions, metrics, head weights, and LoRA weights when applicable.

The comparison has a frozen token-only span head, a frozen terminal-state query head, a query head with a dedicated learned summary embedding and frozen decoder, and a query head with LoRA updates to the decoder's attention projections. The EOS position follows the window and a fixed case-name instruction. For the learned-summary variant, its input embedding is replaced with a trained 576-dimensional vector; this acts as a virtual `[SUMMARY]` token without changing the tokenizer vocabulary. Start and end scores are mapped back to source character offsets. A single fixed instruction can test whether the terminal representation helps this extraction task; it cannot show that the model generalizes to new instructions. That requires annotated spans for other fields and held-out instructions.

The first seed-42 run on Apple M4 uses the uncapped span decoder. Each held-out split has 97 quoted case names:

| Method | Development exact | Test exact | Test overlap |
| --- | ---: | ---: | ---: |
| Existing rule on experiment window | 57/97 | 85/97 | 92/97 |
| Frozen token head | 15/97 | 31/97 | 81/97 |
| Frozen terminal query head | 30/97 | 41/97 | 84/97 |
| Learned summary embedding, frozen decoder | 29/97 | 45/97 | 84/97 |
| Terminal query head with LoRA | 68/97 | 70/97 | 97/97 |

The LoRA model improves on the hard development bankruptcy lawsuit (50/76 exact versus 36/76 for the rule), while the rule remains stronger on the easier test lawsuits. Nine test examples are correct only for LoRA and 24 only for the rule. The LoRA model overlaps every test name but often chooses boundaries a little too wide or narrow.

The largest annotated training name is 74 characters. A 100-character maximum span, chosen after checking development scores, can be applied to the saved checkpoints without retraining:

```sh
.venv/bin/python -m experiments.case_name.evaluate_saved \
  --source-run experiments/case_name/runs/primary-smollm2-135m-seed42 \
  --source-run experiments/case_name/runs/primary-smollm2-135m-summary-seed42 \
  --data-dir experiments/case_name/data \
  --output-dir experiments/case_name/runs/analysis-cap100-seed42 \
  --max-span-chars 100
```

| Method with 100-character cap | Development exact | Test exact | Test overlap |
| --- | ---: | ---: | ---: |
| Frozen token head | 17/97 | 34/97 | 80/97 |
| Frozen terminal query head | 33/97 | 43/97 | 82/97 |
| Learned summary embedding | 31/97 | 47/97 | 81/97 |
| Terminal query head with LoRA | 69/97 | 73/97 | 96/97 |

The capped LoRA model still trails the rule's 85/97 exact test spans. `analysis-cap100-seed42/report.json` records the model revision, Python and library versions, data fingerprints, cap rationale, both sets of scores, and per-example changes.

Swapping the terminal representation with one from a different lawsuit changes only 0/97 LoRA test predictions and 3/97 learned-summary test predictions. These models use the terminal state, but this experiment does not show that its document-specific content drives the chosen span. A fixed case-name instruction also cannot test whether the model will follow a new instruction at inference time.

Report exact character-span matches as the primary metric, plus overlap and per-lawsuit breakdowns. Development scores choose checkpoints and decoding settings. Four held-out lawsuit groups are too few for a broad performance claim.

The rule baseline makes 31 overlapping but boundary-wrong predictions across development and test, including names cut at commas. The token alignment preflight ensures every current positive label can be recovered exactly from the tokenizer offsets after a fixed whitespace/comma trim. Some adjacent locators intentionally share a name span or use different subspans of a printed title; those annotation conventions should be reviewed when interpreting individual errors.
