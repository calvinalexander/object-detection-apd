# Single shared model vs production — human COCO labels, violation = positive

| attribute | model | precision | recall | F1 | **MCC** | bal-acc |
|---|---|---|---|---|---|---|
| **head_cap** (93/202) | *always-flag* | 46.0% | 100.0% | 0.631 | 0.000 | 50.0% |
| | apd_mbg Dec-2025 | 65.8% | 26.9% | 0.382 | 0.191 | 57.5% |
| | apd_mbg Mar-2026 | 56.6% | 32.3% | 0.411 | 0.126 | 55.6% |
| | **ours single model** | **77.5%** | **59.1%** | **0.671** | **0.464** | **72.2%** |
| **mask** (38/44) | *always-flag* | 86.4% | 100.0% | 0.927 | 0.000 | 50.0% |
| | apd_mbg Dec-2025 | 93.3% | 36.8% | 0.528 | 0.146 | 60.1% |
| | apd_mbg Mar-2026 | 75.0% | 23.7% | 0.360 | -0.203 | 36.8% |
| | **ours single model** | **90.6%** | **76.3%** | **0.829** | **0.203** | **63.2%** |
| **gloves** (92/122) | *always-flag* | 75.4% | 100.0% | 0.860 | 0.000 | 50.0% |
| | apd_mbg Dec-2025 | 81.8% | 9.8% | 0.175 | 0.047 | 51.6% |
| | apd_mbg Mar-2026 | 87.5% | 7.6% | 0.140 | 0.074 | 52.1% |
| | **ours single model** | **79.1%** | **78.3%** | **0.787** | **0.148** | **57.5%** |

## Deployment footprint

| | production apd_v3 | ours single model |
|---|---|---|
| networks to compile | 1 | 2 (pose + classifier) |
| AIPU resources | 0.25 | 0.50 |
| params | ~9.4M (yolo11s) | 2.9M pose + 21M trunk |
| attributes per forward pass | 7 classes, one pass | 3 attrs, one pass per ROI |
