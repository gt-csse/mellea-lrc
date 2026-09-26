# Grow-roots evaluation: primary

Docket site hunting: not run

## Stage precision

| Stage | Decision | Precision |
| --- | --- | ---: |
| full_reporter_locators | span | 589/589 (100.0%) |
| full_reporter_locators | normalization | 392/392 (100.0%) |
| docket_locators | span | 19/20 (95.0%) |
| docket_locators | normalization | 13/14 (92.9%) |
| docket_entries | span | 1/1 (100.0%) |
| docket_entries | normalization | 1/1 (100.0%) |
| colocations | groups | 23/23 (100.0%) |
| case_names | span | 513/576 (89.1%) |
| case_names | normalization | 513/571 (89.8%) |
| courts | span | 424/426 (99.5%) |
| courts | normalization | 500/501 (99.8%) |
| dates | span | 550/552 (99.6%) |
| dates | normalization | 550/552 (99.6%) |
| pin_cites | span | 391/392 (99.7%) |
| pin_cites | normalization | 391/392 (99.7%) |
| roots | root_assignment | 607/609 (99.7%) |

## Root fields

| Field | Span precision | Span recall | Normalization precision | Normalization recall |
| --- | ---: | ---: | ---: | ---: |
| full_reporter_locator | 392/392 (100.0%) | 392/393 (99.7%) | 392/392 (100.0%) | 392/392 (100.0%) |
| docket_locator | 13/15 (86.7%) | 13/47 (27.7%) | 13/15 (86.7%) | 13/47 (27.7%) |
| docket_entry | 1/1 (100.0%) | 1/6 (16.7%) | 1/1 (100.0%) | 1/6 (16.7%) |
| case_name | 352/395 (89.1%) | 352/436 (80.7%) | 352/395 (89.1%) | 352/434 (81.1%) |
| court | 304/307 (99.0%) | 304/341 (89.1%) | 360/362 (99.4%) | 360/420 (85.7%) |
| date | 390/393 (99.2%) | 390/426 (91.5%) | 390/393 (99.2%) | 390/426 (91.5%) |
| pin_cite | 264/264 (100.0%) | 264/264 (100.0%) | 264/264 (100.0%) | 264/264 (100.0%) |
