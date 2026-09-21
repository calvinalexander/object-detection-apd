## Operational comparison — human COCO labels, violation = positive

Abstention counted as no alert raised. Trained on all site + val + unannotated
SPPG data; the 53 annotated frames excluded from training.

| attribute | model | abstains | precision | recall | F1 |
|---|---|---|---|---|---|
| **head_cap** (93/202 violating) | ours effnet-b1 (no gate) | 0% | 69.5% | 61.3% | **0.651** |
|  | ours effnet-b1 (ROI gate) | 8% | 70.7% | 57.0% | 0.631 |
|  | apd_mbg (production) | 59% | 65.8% | 26.9% | 0.382 |
| **mask** (38/44 violating) | ours effnet-b1 (no gate) | 0% | 92.0% | 60.5% | **0.730** |
|  | ours effnet-b1 (ROI gate) | 2% | 92.0% | 60.5% | **0.730** |
|  | apd_mbg (production) | 64% | 93.3% | 36.8% | 0.528 |
| **gloves** (92/122 violating) | ours effnet-b1 (no gate) | 0% | 75.0% | 84.8% | **0.796** |
|  | ours effnet-b1 (ROI gate) | 28% | 76.6% | 64.1% | 0.698 |
|  | apd_mbg (production) | 84% | 81.8% | 9.8% | 0.175 |
