# Ours vs both production builds — human COCO labels, violation = positive


## head_cap — 202 persons, 93 violations (46% base rate)

| model | precision | recall | F1 | **MCC** | bal-acc |
|---|---|---|---|---|---|
| *always-flag baseline* | 46.0% | 100.0% | 0.631 | 0.000 | 50.0% |
| apd_mbg Dec-2025 | 65.8% | 26.9% | 0.382 | 0.191 | 57.5% |
| apd_mbg Mar-2026 | 56.6% | 32.3% | 0.411 | 0.126 | 55.6% |
| effnet-b0 +SDK data | 68.3% | 76.3% | 0.721 | 0.459 | 73.0% |
| effnet-b0 (regularised) | 61.0% | 68.8% | 0.646 | 0.311 | 65.6% |
| resnet34 (regularised) | 76.9% | 64.5% | 0.702 | **0.491** | 74.0% |
| resnet34 +SDK data | 68.7% | 73.1% | 0.708 | 0.445 | 72.3% |

## mask — 44 persons, 38 violations (86% base rate)

| model | precision | recall | F1 | **MCC** | bal-acc |
|---|---|---|---|---|---|
| *always-flag baseline* | 86.4% | 100.0% | 0.927 | 0.000 | 50.0% |
| apd_mbg Dec-2025 | 93.3% | 36.8% | 0.528 | 0.146 | 60.1% |
| apd_mbg Mar-2026 | 75.0% | 23.7% | 0.360 | -0.203 | 36.8% |
| effnet-b0 +SDK data | 91.9% | 89.5% | 0.907 | 0.370 | 69.7% |
| effnet-b0 (regularised) | 96.4% | 71.1% | 0.818 | **0.388** | 77.2% |
| resnet34 (regularised) | 88.9% | 84.2% | 0.865 | 0.156 | 58.8% |
| resnet34 +SDK data | 92.3% | 63.2% | 0.750 | 0.208 | 64.9% |

## gloves — 122 persons, 92 violations (75% base rate)

| model | precision | recall | F1 | **MCC** | bal-acc |
|---|---|---|---|---|---|
| *always-flag baseline* | 75.4% | 100.0% | 0.860 | 0.000 | 50.0% |
| apd_mbg Dec-2025 | 81.8% | 9.8% | 0.175 | 0.047 | 51.6% |
| apd_mbg Mar-2026 | 87.5% | 7.6% | 0.140 | 0.074 | 52.1% |
| effnet-b0 +SDK data | 77.7% | 79.3% | 0.785 | 0.096 | 54.7% |
| effnet-b0 (regularised) | 78.1% | 89.1% | 0.832 | **0.155** | 56.2% |
| resnet34 (regularised) | 77.6% | 90.2% | 0.834 | 0.134 | 55.1% |
| resnet34 +SDK data | 75.0% | 71.7% | 0.733 | -0.015 | 49.2% |
