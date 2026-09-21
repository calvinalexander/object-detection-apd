## MCC on human labels — after adding the SDK training data

| model | head_cap | mask | gloves | mean |
|---|---|---|---|---|
| effnet-b0 +SDK data 🆕 | 0.459 | 0.370 | 0.096 | 0.309 |
| effnet-b0 (regularised) | 0.311 | **0.388** | **0.155** | 0.285 |
| resnet34 (regularised) | **0.491** | 0.156 | 0.134 | 0.261 |
| resnet34 +SDK data 🆕 | 0.445 | 0.208 | -0.015 | 0.213 |
| effnet-b1 (regularised) | 0.310 | 0.265 | 0.038 | 0.204 |
| yolo11s-cls | 0.276 | 0.343 | -0.043 | 0.192 |
| effnet-b1 (published) | 0.389 | 0.188 | -0.023 | 0.185 |
| yolo26n-cls | 0.441 | 0.113 | -0.023 | 0.177 |
| yolo26s-cls | 0.362 | -0.073 | 0.028 | 0.106 |
| *always-flag baseline* | 0.000 | 0.000 | 0.000 | 0.000 |

### effect of adding SDK data (same architecture, same val split)

| model | head_cap | mask | gloves | mean |
|---|---|---|---|---|
| effnet-b0 | +0.148 | -0.018 | -0.059 | +0.024 |
| resnet34 | -0.046 | +0.052 | -0.149 | -0.048 |
