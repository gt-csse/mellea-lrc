# Document Inference Development

## Purpose

While citation-level inference and assessment analyze individual references to legal authority, **Document Inference** develops a holistic understanding of the source document itself.

Providing downstream tasks (and our LLM) with contextual understanding about the entire case or brief allows for more accurate assessments of citations, enabling checks for legal properness (e.g., whether it is appropriate for a state trial court to cite a particular appellate decision) and establishing a broader context for the arguments being made.

## Methodological Grounding

A key requirement for Document Inference is that it must operate in a highly **methodological** and **grounded** way. 

We do not want the LLM to re-infer the entire document context for each individual citation it evaluates. Instead, we want to extract the essential document-level context once, and we must do so by grounding every inference in specific evidence from the text. 

For every piece of document-level context extracted, the system must show **exactly what phrase or sentence** was used to derive that conclusion.

## Extracted Context

At a minimum, the Document Inference module should extract the following contextual signals:

1. **Court Level/Jurisdiction:** Which court is this document filed in or issued by? (e.g., state trial court, federal appellate court, supreme court).
2. **Case Topic/Nature:** What is the case fundamentally about? (e.g., contract dispute, criminal appeal).
3. **Party Role (if applicable):** If the document is a brief or motion, did the plaintiff or the defendant draft it?

By extracting these attributes comprehensively upfront and providing them alongside citations, downstream evaluation pipelines can make complex legal properness judgements efficiently and accurately.

## The Translation Layer Heuristic

To successfully determine the **Court Level/Jurisdiction** of a document, we can leverage the aggregate network of citations it contains. This requires normalizing extracted citations from arbitrary text into a standardized court taxonomy (CourtListener Court IDs). Because no authoritative 1:1 crosswalk exists between the extraction layer's MLZ jurisdictions and CourtListener's archival IDs, we are building a dedicated **Translation Layer**.

### Methodology: Heuristics over Automation
The translation layer will be driven strictly by deterministic heuristics, utilizing manual mappings and triangulation, rather than automated fuzzy matching:
1. **Direct Mapping Rules**: We will manually author and maintain mappings for the most unambiguous jurisdictions (e.g., mapping `us;supreme.court` directly to `scotus`). This ensures absolute accuracy for the most common citations.
2. **Triangulation**: When a reporter implies multiple MLZ jurisdictions (e.g., a reporter covering all Federal Circuits), we will triangulate using the explicitly extracted court string (if available) to narrow down to a precise CourtListener ID.
3. **Fuzzy Automation is Excluded**: We will not attempt automated fuzzy matching between Jurism's database and CourtListener's taxonomy, as the risk of cascading errors in downstream legal tasks is too high.

### Document-Level Profiling & Tracing
By synthesizing the translated Court IDs from all citations within a document, we can generate a reliable document-level jurisdictional profile (e.g., if a document predominantly cites `ca9` and `scotus`, it is highly likely a federal case). 

Crucially, the translation layer and its resulting inferences will be integrated directly into the application's **tracing UI**. This provides full transparency into how the system triangulates the jurisdiction and maps it to a canonical CourtListener ID, allowing users to verify the exact logic used for each inference.
