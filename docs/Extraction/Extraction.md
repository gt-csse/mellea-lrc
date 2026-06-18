

> [!TIP]
> Cheaper than getting fined by a court
## Description

The extraction stage will be responsible for **extracting**, **cleaning**, **annotating**, and **aggregating** case law (opinions) citations from legal documents.

---
## Unstructured-to-Structured conversion

### Tools

- [`Docling`](https://www.docling.ai/)
- [`markitdown`](https://github.com/microsoft/markitdown)
### Inputs

- PDF (Most Common)
- DocX
- Images (Scans)
- Plain-text
---
## Pre-Processing (Augmentation)

> [!WARNING]
> **Goodhart's Law**: When a measurement becomes the target, it ceases to be a good measurement.
### Adaptors

We want to ensure that the pre-processing stage to stay agnostic to the choice of Unstructured-to-Structured conversion tool, e.g., [`Docling`](https://www.docling.ai/), [`markitdown`](https://github.com/microsoft/markitdown). We will use `Docling` for now. 

we will provide a suite of functions for fixing the formatting mistakes made by the Unstructured-to-Structured conversion tools. The 

---
## Benchmarks

## Models

- **Hybrid**
- **Mellea**
	- Extract + Label
	- Extract (use `eyecite` for labelling)
	- Label
- **Eyecite**
- [Pelaikan](https://pelaikan-app.web.app)

## Extraction 

> Extraction is the processing of identify that a string of characters is a citations. We need to evaluate our models on their ability to identify citations. There are two main metrics: Recall and Precision. Recall will tell us how many of the total case citations did the model identify. For instance, if there 100 cite citations, did the model find them all? Precision will tell us: how all of the citations that the model identified, which ones are case citations or just citations? The two metrics are complimentary. For examples, we can a Recall of 100; however, it could have that the model selected all possible sets of strings. The precision will then tell us that the model did a horrible job since most of the selected strings are not citations. The same goes the other way around. We can get 100 percent precision, but a Recall of 1 percent, meaning that out of 100 citations, the model only found one.


## Labelling Metrics

> A case citation is composed of a couple of parts. 

| Metrics                    | Mellea | Eyecite | Hybrid (Eyecite + Mellea) |     |
| -------------------------- | ------ | ------- | ------------------------- | --- |
| Precision (TP vs. TP + FP) |        |         |                           |     |
| Recall (TP vs. TP + TN)    |        |         |                           |     |
| Speed (Seconds)            |        |         |                           |     |
| Cost (Tokens)              |        |         |                           |     |
|                            |        |         |                           |     |
|                            |        |         |                           |     |

|          | True                           | False                      |
| -------- | ------------------------------ | -------------------------- |
| Positive | It's AI, labeled as AI         | It's not AI, labeled as AI |
| Negative | It's not AI, labeled as not AI | It's AI, labeled as not AI |
|          |                                |                            |
|          |                                |                            |
> Precision: If we look into the bucket of citations that were labeled as AI-hallucinated, how many are actually AI-hallucinated?
$$
\text{Precision}= \frac{\text{It's AI, Labeled AI (TP)}}{\text{It's AI, Labeled AI (TP) + It's Not AI, Labled AI (FP)}}
$$
> Recall: Out all of the AI-hallucinated cases, how many did the model identify?
$$
\text{Recall} = \frac{\text{It's AI, Labeled AI (TP)}}{\text{It's AI, Labeled AI (TP)}+ \text{It's AI, Labeled Not AI (FN)}}
$$
### Measurements

