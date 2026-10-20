# Vision-Based Analysis of Auricular Acupoints for Hypertension and Diabetes

A computer vision pipeline that locates eight auricular (ear) acupoints in ear photographs, measures colour and texture at each point, and tests whether those measurements differ between healthy people, people with hypertension and people with diabetes.

> **Research use only.** This project does not diagnose or screen for hypertension or diabetes. It investigates whether selected auricular points show visual differences between known clinical groups. The results are exploratory (pilot scale).

## Pipeline

1. **Image acquisition:** standardised front-view ear photographs (ring light, macro lens, fixed distance and framing), both ears per participant.
2. **Annotation:** eight points labelled manually in LabelMe (one JSON file per image).
3. **Detection:** a YOLO11-pose model fine-tuned to locate the eight points automatically.
4. **Feature extraction:** a 15x15 pixel patch around each point is converted into colour (RGB, CIELAB, HSV) and texture features with OpenCV.
5. **Statistical comparison:** features are averaged per participant and compared across the three groups with non-parametric tests and false-discovery-rate control.

## Acupoints studied

Ear Apex, Shen Men, Superior Triangular Fossa, Sympathetic, Kidney, Heart, Pancreas and Endocrine, selected from published clinical literature (WFAS international standard of auricular points).

## Repository structure

```
.
├── README.md
├── .gitignore
└── scripts/
    ├── prepare_yolo_data.py   # LabelMe JSON -> YOLO format, automatic train/val split
    ├── data.yaml              # YOLO dataset config (8 keypoints)
    ├── train_yolo.py          # fine-tune YOLO11-pose
    ├── predict.py             # run the trained model on one image
    ├── extract_features.py    # colour/texture features at the labelled points
    └── compare_groups.py      # group comparison and statistics
```

Folders created or expected when you run the code (not included in the repository):

```
data/
├── images/           # ear photographs
└── labelme_json/     # LabelMe point annotations
results/              # generated: features.csv, stats tables, plots, overlays
```

## Dataset and privacy

The dataset (ear photographs and annotations) is **not included** in this repository, to protect participant privacy. Data were collected through a collaborating clinical practitioner with informed consent, and images were named with anonymous codes. To run the scripts on your own data, provide `data/images` and `data/labelme_json`.

File names encode the group, participant and ear side:

| Pattern | Meaning |
|---|---|
| `HE_01_L.jpg` | Healthy, participant 01, left ear |
| `HY_01_R.jpg` | Hypertension, participant 01, right ear |
| `DB_01_L.jpg` | Diabetes, participant 01, left ear |
| `01_L.jpg` | General group (diagnosis unknown), used only to train the detector |

| Group | Participants | Images | Use |
|---|---|---|---|
| General | 25 | 50 | Detector training only |
| Healthy | 24 | 48 | Detector training and statistics |
| Hypertension | 25 | 50 | Detector training and statistics |
| Diabetes | 23 | 45 | Detector training and statistics |
| Total | 97 | 193 | |

## Setup

Python 3 is required.

```bash
pip install ultralytics opencv-python numpy pandas scipy statsmodels matplotlib
```

## Usage

Run from the project root, in this order.

```bash
# 1. Convert annotations to YOLO format and split into train/validation
python scripts/prepare_yolo_data.py

# 2. Fine-tune the YOLO11-pose detector
python scripts/train_yolo.py

# 3. (Optional) Detect the points on one image; check the paths at the top of the script
python scripts/predict.py

# 4. Extract colour and texture features -> results/features.csv
python scripts/extract_features.py

# 5. Compare the groups -> results/stats_omnibus.csv, stats_posthoc.csv, top_features.png
python scripts/compare_groups.py
```

### What the feature step does

For each image in `data/labelme_json`, `extract_features.py` reads the group from the file-name prefix (HE, HY, DB; general images are skipped), cuts a 15x15 pixel patch around each labelled point, and records:

| Column(s) | Meaning |
|---|---|
| `R`, `G`, `B` | Mean red, green, blue of the patch (0 to 255) |
| `L` | Mean CIELAB lightness (OpenCV 0 to 255 scale) |
| `a` | CIELAB red-green axis (128 is neutral, higher is redder) |
| `b` | CIELAB blue-yellow axis (128 is neutral, higher is more yellow) |
| `saturation`, `brightness` | Mean HSV saturation and value |
| `texture` | Standard deviation of grayscale intensity in the patch |
| `patch_pixels` | Pixels in the patch (225 for a full patch; fewer if clipped by the image border) |
| `*_rel` | The feature minus the mean over the eight points of the same image (partly cancels exposure differences) |

The script also saves an overlay image for each photograph, with a box drawn on every measured patch, so the patches can be checked visually.

### What the statistics step does

1. Keeps seven points (Ear Apex is localised by the detector but left out of the group comparison).
2. Uses five features chosen in advance: `a`, `a_rel`, `L`, `saturation`, `texture`.
3. Averages the left and right ear per participant (72 participants: 24, 25, 23).
4. Runs a Kruskal-Wallis test across the three groups for each of the 7 x 5 = 35 point-feature combinations (one-way ANOVA is reported alongside).
5. Controls the false-discovery rate across the 35 tests (Benjamini-Hochberg, significant if q < 0.05).
6. For combinations with an omnibus p below 0.05, runs pairwise Mann-Whitney U tests with Holm correction.
7. Reports the effect size (epsilon-squared) and group medians, and plots the six strongest results.

## Results

### Detector

YOLO11-pose was fine-tuned on 193 annotated images on a CPU-only laptop. Training stopped early at epoch 118 of 150 (no improvement for 30 epochs).

| Stage | Training data | Pose mAP50-95 |
|---|---|---|
| Pipeline check | 4 photographs | Not usable (too little data) |
| Practice round | 50 photographs, 25 subjects | about 0.87 |
| Final model | 193 photographs, 97 participants | 0.94 |

Final validation metrics (epoch 118): precision 1.000, recall 1.000, mAP50 0.995, pose mAP50-95 0.941.

### Group comparison

Of the 35 tests, 5 remain significant after FDR correction (q < 0.05).

| Point | Feature | p (Kruskal-Wallis) | q (FDR) | Epsilon-squared | Median healthy | Median hypertension | Median diabetes |
|---|---|---|---|---|---|---|---|
| Endocrine | L | 7.7e-05 | 0.0027 | 0.267 | 99.0 | 62.5 | 64.9 |
| ShenMen | a_rel | 6.6e-04 | 0.0116 | 0.206 | 3.4 | 6.3 | 5.4 |
| Heart | L | 1.2e-03 | 0.0142 | 0.189 | 145.4 | 106.1 | 107.2 |
| Heart | saturation | 2.2e-03 | 0.0183 | 0.172 | 98.4 | 124.2 | 127.9 |
| ShenMen | saturation | 2.6e-03 | 0.0183 | 0.167 | 131.5 | 157.6 | 168.0 |

Main observations:

- Endocrine and Heart are darker (lower L) in both hypertension and diabetes than in healthy ears. The two patient groups do not differ from each other.
- Shen Men saturation is higher in both patient groups. Heart saturation differs significantly only between healthy and diabetes.
- Shen Men is relatively redder in hypertension (a_rel); this is the only result separating the two patient groups, and it is borderline (Holm p = 0.049).
- Points tied specifically to diabetes in the literature (Kidney, Pancreas, Sympathetic) showed no significant difference after correction.

## Limitations

- Small, single-site sample: 24, 25 and 23 participants per group.
- Exposure, age, sex, skin tone and medication were not controlled, and exposure parity between groups was not verified. The lightness and saturation results could partly reflect such differences.
- Individual values overlap heavily between groups, so these features cannot classify a single person.
- The statistical comparison used the manually annotated points, not detector output.
- The detector's validation set is small, with no external test set.
- 35 tests were run on one dataset, so the findings need replication.

## Future work

- A larger, age- and sex-matched cohort, with a colour reference card in every photograph.
- Including Ear Apex and testing other patch sizes.
- Running the statistics on detector-predicted points.
- Modelling left and right ears together and adding ROC analysis if a feature shows promise.
- Validating the detector on photographs from other cameras and sites.

## Tech stack

Python, Ultralytics YOLO11-pose, OpenCV, LabelMe, SciPy, pandas, NumPy, statsmodels, matplotlib.
