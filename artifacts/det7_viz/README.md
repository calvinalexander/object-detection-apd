# det7 dataset visualisation

Green is a satisfied attribute, red is a violation, white is the person box.

| class | train | val |
|---|---|---|
| person | 2257 | 401 |
| gloves | 1227 | 174 |
| head_cap | 1526 | 288 |
| mask | 548 | 81 |
| no_gloves | 1296 | 199 |
| no_head_cap | 973 | 130 |
| no_mask | 798 | 132 |

Images: 2257 train, 401 val  (one per person, cropped from the frame)

| folder | shows |
|---|---|
| 01_train / 02_val | what the model sees, all boxes drawn |
| 03_by_class | one folder per class, that class highlighted and the rest dimmed |
| 04_overlay_cleaned | source snapshot beside the dataset copy, caption removed |
| 06_by_source/native | crops from frames that arrived already annotated |
| 06_by_source/site | crops from production snapshots, consensus labels |
