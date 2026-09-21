## Classifier ranking — production annotation (violation = positive class)

Threshold 0.5, ROI gate on. n = head_cap 247 (138 violations), mask 202 (121 violations), gloves 156 (83 violations).

| model | head_cap P | head_cap R | head_cap F1 | mask P | mask R | mask F1 | gloves P | gloves R | gloves F1 |
|---|---|---|---|---|---|---|---|---|---|
| ci3 (yolo26n-cls) | 98.5% | 92.8% | 0.955 | 94.8% | 90.9% | 0.928 | **95.4%** | **100.0%** | **0.976** |
| inpaint (yolo11n-cls) | 98.5% | 97.1% | 0.978 | 93.9% | 88.4% | 0.911 | 94.3% | 100.0% | 0.971 |
| inpaint (yolo11s-cls) | **100.0%** | **98.6%** | **0.993** | 94.0% | 77.7% | 0.851 | 91.1% | 98.8% | 0.948 |
| inpaint (yolo11m-cls) | 98.6% | 99.3% | 0.989 | 91.2% | 94.2% | 0.927 | 92.0% | 97.6% | 0.947 |
| inpaint (effnet-b1) | 90.7% | 98.6% | 0.944 | **95.7%** | **92.6%** | **0.941** | 92.0% | 97.6% | 0.947 |
| voter4 (original) | 97.2% | 99.3% | 0.982 | 91.1% | 93.4% | 0.922 | 91.2% | 100.0% | 0.954 |

### Summary

| attribute | best model | precision | recall | F1 | AUC | false alarms |
|---|---|---|---|---|---|---|
| head_cap | inpaint (yolo11s-cls) | 100.0% | 98.6% | **0.993** | 0.999 | 0 of 138 |
| mask | inpaint (effnet-b1) | 95.7% | 92.6% | **0.941** | 0.974 | 5 of 121 |
| gloves | ci3 (yolo26n-cls) | 95.4% | 100.0% | **0.976** | 0.998 | 4 of 83 |
