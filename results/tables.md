## Table 1. Evaluation set (BDD100K val; pedestrians >= 25 px)

| Condition | Images | Images with >=1 pedestrian | Pedestrian boxes | Boxes per image (with ped.) | Mean image luma |
|---|---|---|---|---|---|
| daytime | 5258 | 2004 | 7999 | 3.99 | 103.9 |
| night | 3929 | 813 | 2547 | 3.13 | 31.1 |

## Table 2. AP50 and AP (%), value [95% bootstrap CI]; delta in percentage points

| Model | AP50 day | AP50 night | dAP50 (pp) | rel. drop | AP day | AP night | dAP (pp) |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 48.5 [46.9, 49.9] | 33.3 [30.6, 36.1] | 15.2 [12.1, 18.2] | 31% | 23.0 [22.1, 23.9] | 15.6 [14.1, 17.0] | 7.4 [5.6, 9.2] |
| YOLO11n | 48.4 [46.7, 49.8] | 35.2 [32.3, 37.8] | 13.2 [10.1, 16.4] | 27% | 22.8 [21.9, 23.7] | 15.9 [14.5, 17.4] | 6.9 [5.1, 8.7] |
| SSDLite320-MobileNetV3 | 8.0 [7.2, 8.9] | 5.2 [4.2, 6.3] | 2.8 [1.5, 4.2] | 35% | 3.0 [2.6, 3.3] | 2.0 [1.5, 2.4] | 1.0 [0.4, 1.6] |

## Table 3. Operating point (score >= 0.25, IoU 0.5), %

| Model | Precision day | Precision night | Recall day | Recall night | Miss rate day | Miss rate night | dMiss (pp) |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 79.9 [78.4, 81.4] | 72.0 [68.7, 75.3] | 35.9 [34.4, 37.2] | 26.0 [23.7, 28.1] | 64.1 [62.8, 65.6] | 74.0 [71.9, 76.3] | 9.9 [7.4, 12.6] |
| YOLO11n | 78.6 [77.0, 80.1] | 77.0 [73.5, 80.1] | 36.4 [34.9, 37.8] | 24.5 [22.2, 26.7] | 63.6 [62.2, 65.1] | 75.5 [73.3, 77.8] | 11.8 [9.4, 14.6] |
| SSDLite320-MobileNetV3 | 50.7 [47.1, 54.3] | 40.4 [34.3, 46.9] | 6.0 [5.3, 6.6] | 4.3 [3.5, 5.3] | 94.0 [93.4, 94.7] | 95.7 [94.7, 96.5] | 1.7 [0.5, 2.8] |

## Table 4. Model size and single-image latency (batch 1, night images)

| Model | Params (M) | CUDA mean (ms) | CUDA p95 (ms) | CUDA FPS | CPU mean (ms) | CPU p95 (ms) | CPU FPS |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 3.16 | 8.8 | 9.4 | 113.5 | 60.1 | 63.4 | 16.6 |
| YOLO11n | 2.62 | 11.2 | 12.5 | 89.2 | 59.0 | 62.8 | 17.0 |
| SSDLite320-MobileNetV3 | 3.44 | 56.6 | 66.0 | 17.7 | 120.4 | 133.5 | 8.3 |

## Table 5. Recall (%) at the operating point by mean luma of the pedestrian box (n = boxes)

| Model | Condition | Y 0-29 | Y 30-59 | Y 60-99 | Y 100-149 | Y 150-255 |
|---|---|---|---|---|---|---|
| YOLOv8n | daytime | 29.8 (n=198) | 34.2 (n=2555) | 36.2 (n=3692) | 39.0 (n=1420) | 32.8 (n=134) |
| YOLOv8n | night | 32.7 (n=630) | 24.4 (n=1240) | 22.7 (n=578) | 22.0 (n=91) | 12.5 (n=8) |
| YOLO11n | daytime | 29.3 (n=198) | 33.7 (n=2555) | 37.1 (n=3692) | 40.6 (n=1420) | 34.3 (n=134) |
| YOLO11n | night | 31.7 (n=630) | 23.2 (n=1240) | 20.8 (n=578) | 17.6 (n=91) | 12.5 (n=8) |
| SSDLite320-MobileNetV3 | daytime | 2.0 (n=198) | 5.1 (n=2555) | 6.9 (n=3692) | 5.8 (n=1420) | 3.0 (n=134) |
| SSDLite320-MobileNetV3 | night | 5.7 (n=630) | 4.2 (n=1240) | 3.5 (n=578) | 1.1 (n=91) | 0.0 (n=8) |

## Table 6. Sensitivity: minimum pedestrian height

| Model | dAP50 (pp), >= 25 px | dAP50 (pp), >= 50 px | Rank day | Rank night | Rank night (>= 50 px) |
|---|---|---|---|---|---|
| YOLOv8n | 15.2 [12.1, 18.2] | 18.7 [15.3, 22.1] | 1 | 2 | 2 |
| YOLO11n | 13.2 [10.1, 16.4] | 16.5 [13.1, 20.0] | 2 | 1 | 1 |
| SSDLite320-MobileNetV3 | 2.8 [1.5, 4.2] | 5.7 [3.6, 7.6] | 3 | 3 | 3 |

Pairwise AP50 differences (pp) [95% CI], resolved = CI excludes 0:

- daytime: YOLOv8n - YOLO11n: 0.1 [-0.6, 0.7] unresolved
- daytime: YOLOv8n - SSDLite320-MobileNetV3: 40.4 [39.0, 41.7] resolved
- daytime: YOLO11n - SSDLite320-MobileNetV3: 40.3 [38.9, 41.6] resolved
- night: YOLOv8n - YOLO11n: -1.9 [-3.2, -0.5] resolved
- night: YOLOv8n - SSDLite320-MobileNetV3: 28.1 [25.6, 30.6] resolved
- night: YOLO11n - SSDLite320-MobileNetV3: 30.0 [27.4, 32.4] resolved

## Table A2. Recall (%) at the operating point by pedestrian height

| Model | Condition | 25-49 px | 50-99 px | >= 100 px |
|---|---|---|---|---|
| YOLOv8n | daytime | 6.8 (n=3265) | 45.6 (n=3345) | 80.8 (n=1389) |
| YOLOv8n | night | 1.6 (n=931) | 28.8 (n=1095) | 63.5 (n=521) |
| YOLO11n | daytime | 7.8 (n=3265) | 46.0 (n=3345) | 80.4 (n=1389) |
| YOLO11n | night | 1.7 (n=931) | 26.2 (n=1095) | 61.8 (n=521) |
| SSDLite320-MobileNetV3 | daytime | 0.2 (n=3265) | 1.4 (n=3345) | 30.5 (n=1389) |
| SSDLite320-MobileNetV3 | night | 0.0 (n=931) | 0.7 (n=1095) | 19.4 (n=521) |
