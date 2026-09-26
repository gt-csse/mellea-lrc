# Incremental reporter field judgments

<!-- Generated from evaluations/results/current-primary/summary.json by evaluations.evaluate_run. -->

Each row scores judgments written by that stage. Precision divides correct decided judgments by decided judgments; recall divides correctly judged annotated roots by annotated roots eligible for that stage. An unresolved judgment counts as a recall miss.

| Stage | Set | Field | Precision | Recall |
| --- | --- | --- | ---: | ---: |
| reporter_root_lookup | primary | case name | 88.7% | 82.8% |
| reporter_root_lookup | primary | court | 98.3% | 90.0% |
| reporter_root_lookup | primary | date | 98.8% | 98.4% |
| reporter_root_lookup_ambiguous | primary | case name | 100.0% | 17.3% |
| reporter_root_lookup_ambiguous | primary | court | 100.0% | 16.0% |
| reporter_root_lookup_ambiguous | primary | date | 100.0% | 17.6% |
| reporter_root_lookup_ambiguous_llm | primary | case name | 97.6% | 93.0% |
| reporter_root_lookup_ambiguous_llm | primary | court | 100.0% | 92.9% |
| reporter_root_lookup_ambiguous_llm | primary | date | 100.0% | 97.6% |
| reporter_root_lookup_unique_llm | primary | case name | 97.3% | 96.0% |
| reporter_root_lookup_unique_llm | primary | court | 95.4% | 83.8% |
| reporter_root_lookup_unique_llm | primary | date | 100.0% | 92.0% |
