# Lightweight Pedestrian Detectors at Night: A Controlled Replication on BDD100K

**Authors:** Tarielov Baiaman (Student ID 80205), Mahmoud Eldessouky (Student ID 80590)
**Course:** Architecture of Computers (CII3SE003CI), Student Research Challenge, Winter Semester 2026/2027, Group 4, Group 1

This repository holds the paper, the code, and every result for a study of how much the pedestrian
detection accuracy of three lightweight, COCO-pretrained detectors (YOLOv8n, YOLO11n,
SSDLite320-MobileNetV3-Large) drops from daytime to night-time images of the BDD100K validation split.

**Paper (PDF):** [`paper/Research_Paper_Night_Pedestrian_Detection.pdf`](paper/Research_Paper_Night_Pedestrian_Detection.pdf)

## Key results

Evaluation set: all 5,258 daytime and 3,929 night images of the BDD100K validation split, with pedestrians ≥ 25 px.
95% bootstrap intervals are in brackets.

| Model | AP50 day (%) | AP50 night (%) | Drop (pp) | GPU latency (ms, Tesla T4) | CPU latency (ms, 2 threads) |
|---|---|---|---|---|---|
| YOLOv8n | 48.5 [46.9, 49.9] | 33.3 [30.6, 36.1] | 15.2 [12.1, 18.2] | 8.8 | 60.1 |
| YOLO11n | 48.4 [46.7, 49.8] | 35.2 [32.3, 37.8] | 13.2 [10.1, 16.4] | 11.2 | 59.0 |
| SSDLite320-MobileNetV3-Large | 8.0 [7.2, 8.9] | 5.2 [4.2, 6.3] | 2.8 [1.5, 4.2] | 56.6 | 120.4 |

- Every model loses accuracy at night (H1 supported for all three).
- YOLOv8n and YOLO11n are tied by day. At night, YOLO11n leads on AP50 and YOLOv8n on recall.
- When daytime pedestrians are matched to the night ones on brightness and height, night recall is still 9–23 pp lower. Darkness of the pedestrian alone does not explain the drop.
- All numbers come from offline evaluation on a cloud machine (Kaggle). They are relative model comparisons, not in-vehicle performance.

## Repository structure

```
aoc-projectfinal/
├── paper/
│   ├── Research_Paper_Night_Pedestrian_Detection.pdf    submitted paper
│   └── Research_Paper_Night_Pedestrian_Detection.docx   editable source
├── code/
│   ├── bdd_night_experiment.py   the whole experiment in one file: finds the data, runs the 3 models,
│   │                             measures latency, computes all metrics, writes results.zip
│   ├── analyze.py                metric code (COCO-style matching, AP, bootstrap, tables, figures);
│   │                             recomputes everything from results/ without images or GPU
│   ├── paper_analysis.py         extra analyses in the paper: paired model differences (Sec. 4.1),
│   │                             height-adjusted recall by pedestrian brightness (Table 5, Fig. 2)
│   ├── test_ap_vs_pycocotools.py checks that our AP equals the official pycocotools AP
│   └── requirements.txt
├── results/                      unchanged output of the Kaggle run
│   ├── tables.md                 Tables 1-4, 6, A2 (Table 5 in tables.md is the unadjusted version)
│   ├── summary.json              every number with 95% bootstrap intervals
│   ├── run_meta.json             hardware, software versions, settings, latency
│   ├── raw_detections.npz        all raw detections and ground truth (re-analysis without images)
│   ├── fig1_ap50_vs_latency.png, fig2_recall_vs_luminance.png
│   └── failures/                 night images where every model misses a pedestrian >= 50 px
└── figures/                      figures exactly as they appear in the paper (from paper_analysis.py)
```

## How to reproduce

**1. Run the experiment on Kaggle (free GPU, about 30–60 min):**
1. Kaggle → Code → New Notebook → **Add Input**: dataset [`solesensei/solesensei_bdd100k`](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k).
2. Settings: Accelerator = **GPU T4**, Internet = **On**.
3. Cell 1: `!pip install -q ultralytics`
4. Cell 2: paste `code/bdd_night_experiment.py` and run it.
5. Download `/kaggle/working/results.zip`. Its contents are the files in `results/`.

To run it locally instead:

```bash
pip install -r code/requirements.txt
python code/bdd_night_experiment.py --root path/to/unzipped/bdd100k   # add --n 50 --B 50 for a quick test
```

**2. Recompute the analyses from the stored detections (CPU only, a few minutes):**

```bash
cd code
python analyze.py --results ../results                  # tables.md, summary.json, figures
python paper_analysis.py ../results ../figures          # Table 5, Figure 1, Figure 2, Figure D1
python test_ap_vs_pycocotools.py                        # must print OK
```

## Experimental setup (as run)

| | |
|---|---|
| Hardware | Kaggle notebook: NVIDIA Tesla T4 (16 GB), Intel Xeon @ 2.00 GHz (2 cores, 2 PyTorch threads), 31 GB RAM, Linux |
| Software | Python 3.13, PyTorch 2.11.0 (CUDA 13.0), torchvision 0.26.0, Ultralytics 8.4.218 |
| Models | `yolov8n.pt`, `yolo11n.pt` (Ultralytics); torchvision default COCO weights for SSDLite320-MobileNetV3-Large; no fine-tuning |
| Ground truth | BDD100K pedestrians ≥ 25 px (sensitivity: ≥ 50 px); riders and smaller pedestrians are ignore regions |
| Metrics | COCO-style AP50 and AP; precision, recall and miss rate at score ≥ 0.25 and IoU 0.5; 1,000 image-level bootstrap resamples |
| Latency | batch size 1 on night images; GPU: 20 warm-up + 100 timed runs; CPU: 10 warm-up + 50 timed runs |

## Data and licences

- **BDD100K** (Yu et al., CVPR 2020) was obtained from the public Kaggle mirror linked above. It is used under the BDD100K licence, which permits educational, research and not-for-profit use. The six images in `results/failures/` are BDD100K images with our detection boxes drawn on them: copyright © 2018 The Regents of the University of California, all rights reserved.
- Model weights are © Ultralytics (AGPL-3.0) and © PyTorch / torchvision (BSD-3-Clause).

## Declaration of Generative AI use

AI was used for writing code, searching for information, drafting content, and correcting grammatical mistakes.
