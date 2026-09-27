# Validate-roots evaluation: primary

## docket_root_lookup_review

| Field | Precision |
| --- | ---: |
| case_name | 31/35 (88.6%) |
| court | 34/35 (97.1%) |
| date | 3/35 (8.6%) |

## reporter_root_lookup

| Field | Precision |
| --- | ---: |
| case_name | 211/248 (85.1%) |
| court | 244/250 (97.6%) |
| date | 251/254 (98.8%) |

## reporter_root_lookup_ambiguous

| Field | Precision |
| --- | ---: |
| case_name | 9/9 (100.0%) |
| court | 8/8 (100.0%) |
| date | 9/9 (100.0%) |

## reporter_root_lookup_unique_llm

| Field | Precision |
| --- | ---: |
| case_name | 71/75 (94.7%) |
| court | 65/75 (86.7%) |
| date | 71/75 (94.7%) |

## reporter_root_lookup_ambiguous_llm

| Field | Precision |
| --- | ---: |
| case_name | 41/42 (97.6%) |
| court | 38/42 (90.5%) |
| date | 42/42 (100.0%) |

## Root field judgments

| Field | Precision | Recall |
| --- | ---: | ---: |
| case_name | 333/342 (97.4%) | 333/440 (75.7%) |
| court | 326/342 (95.3%) | 326/440 (74.1%) |
| date | 305/342 (89.2%) | 305/440 (69.3%) |
