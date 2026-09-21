## resnet34 vs production — the only model significant on all three

| attribute | | production | resnet34 | change |
|---|---|---|---|---|
| **head_cap** | precision | 65.8% | 76.9% | +11.1% |
| (93/202 violating) | recall | 26.9% | 64.5% | **+37.6%** |
| | F1 | 0.382 | **0.702** | +0.320 (p=2.9e-04) |
| **mask** | precision | 93.3% | 88.9% | -4.4% |
| (38/44 violating) | recall | 36.8% | 84.2% | **+47.4%** |
| | F1 | 0.528 | **0.865** | +0.337 (p=2.6e-03) |
| **gloves** | precision | 81.8% | 77.6% | -4.2% |
| (92/122 violating) | recall | 9.8% | 90.2% | **+80.4%** |
| | F1 | 0.175 | **0.834** | +0.659 (p=1.3e-07) |

## Full bake-off — operational, human labels (** = beats production, p<0.05)


**head_cap** — 202 persons, 93 violations

| model | precision | recall | F1 | p vs production |
|---|---|---|---|---|
| apd_mbg (production) | 65.8% | 26.9% | 0.382 | baseline |
| effnet-b1 (published) | 69.5% | 61.3% | **0.651** | 2.2e-02 ✓ |
| effnet-b0 (regularised) | 61.0% | 68.8% | 0.646 | 3.0e-01 (ns) |
| resnet34 (regularised) | 76.9% | 64.5% | **0.702** | 2.9e-04 ✓ |
| yolo26n-cls | 74.0% | 61.3% | **0.671** | 1.0e-03 ✓ |
| yolo11s-cls | 60.2% | 63.4% | 0.618 | 4.2e-01 (ns) |

**mask** — 44 persons, 38 violations

| model | precision | recall | F1 | p vs production |
|---|---|---|---|---|
| apd_mbg (production) | 93.3% | 36.8% | 0.528 | baseline |
| effnet-b1 (published) | 92.0% | 60.5% | 0.730 | 1.3e-01 (ns) |
| effnet-b0 (regularised) | 96.4% | 71.1% | **0.818** | 1.1e-02 ✓ |
| resnet34 (regularised) | 88.9% | 84.2% | **0.865** | 2.6e-03 ✓ |
| yolo26n-cls | 89.3% | 65.8% | 0.758 | 7.8e-02 (ns) |
| yolo11s-cls | 96.2% | 65.8% | **0.781** | 2.7e-02 ✓ |

**gloves** — 122 persons, 92 violations

| model | precision | recall | F1 | p vs production |
|---|---|---|---|---|
| apd_mbg (production) | 81.8% | 9.8% | 0.175 | baseline |
| effnet-b1 (published) | 75.0% | 84.8% | **0.796** | 4.3e-06 ✓ |
| effnet-b0 (regularised) | 78.1% | 89.1% | **0.832** | 9.4e-08 ✓ |
| resnet34 (regularised) | 77.6% | 90.2% | **0.834** | 1.3e-07 ✓ |
| yolo26n-cls | 75.0% | 84.8% | **0.796** | 3.3e-06 ✓ |
| yolo11s-cls | 74.5% | 79.3% | **0.768** | 1.6e-05 ✓ |
