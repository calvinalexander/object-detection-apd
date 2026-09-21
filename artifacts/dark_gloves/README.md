# Dark glove examples in the training data (det7)

Every `gloves` box in data/clean/det7, classified by median brightness (HSV V).
"Dark" means V < 70, i.e. black or very dark navy gloves.

| | boxes | median glove width | median person height |
|---|---|---|---|
| all gloves boxes | 1394 | 29 px | 255 px |
| dark ones | **217** (16%) | 22 px | 288 px |

On the device camera the alerted people are ~118 px tall, so its gloves are far
smaller than these examples.

Dark gloves by source (7 distinct sources):

| source | dark boxes | all gloves boxes there |
|---|---|---|
| pre-annotated | 193 | 1257 |
| site 72.02.07.2007.01 | 7 | 10 |
| site 73.71.09.1003.08 | 5 | 8 |
| site 33.21.11.2012.04 | 5 | 11 |
| site 73.09.04.1003.02 | 4 | 5 |
| site 71.71.09.1003.02 | 2 | 3 |
| site 34.03.08.2002.01 | 1 | 2 |

Sheets: each tile is the glove box (left) and its surroundings (right), with
source, box width and brightness. boxes.csv lists every gloves box with its
measurements, so the cut can be re-drawn at a different brightness.
