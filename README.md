# 👂 Vision-Based Analysis of Auricular Acupoints for Hypertension and Diabetes

<p align="center">
  <strong>A Computer Vision and Statistical Analysis Pipeline for Exploring Visual Differences in Auricular Acupoints</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLO11-Pose-purple" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Research-Exploratory-orange" />
</p>

A computer vision pipeline that automatically localises eight auricular (ear) acupoints, extracts colour and texture features from ear photographs, and statistically investigates whether these features differ between three known clinical groups: healthy individuals, individuals with hypertension, and individuals with diabetes.

The project combines **YOLO11-Pose, OpenCV, and statistical hypothesis testing** to explore potential visual patterns in selected auricular points.

> ⚠️ **Research use only:** This project is not a diagnostic or screening tool for hypertension or diabetes. It investigates visual differences between previously identified clinical groups. Findings are exploratory and based on a pilot-scale dataset.

---

## ✨ Key Features

- **📍 Auricular Acupoint Localisation** — Detects eight predefined ear acupoints using a fine-tuned YOLO11-Pose model.
- **🖼️ Image Annotation** — Uses LabelMe for manual keypoint annotation and dataset preparation.
- **🎨 Colour Feature Extraction** — Extracts RGB, CIELAB, and HSV colour information from localised regions.
- **🔍 Texture Analysis** — Measures grayscale intensity variation around each acupoint.
- **📊 Statistical Comparison** — Compares feature distributions across healthy, hypertension, and diabetes groups.
- **🧪 Multiple Testing Correction** — Applies Benjamini-Hochberg FDR and Holm correction to reduce false-positive findings.
- **📈 Visualisation** — Generates annotated image overlays, statistical tables, and feature comparison plots.

## 🔬 Project Pipeline

```text
Ear Photographs
      │
      ▼
Manual Annotation (LabelMe)
      │
      ▼
Dataset Preparation
(LabelMe JSON → YOLO Format)
      │
      ▼
YOLO11-Pose Training
      │
      ▼
Acupoint Localisation
      │
      ▼
Colour & Texture Extraction
      │
      ▼
Participant-Level Aggregation
      │
      ▼
Statistical Group Comparison
      │
      ▼
Results, Tables & Visualisations
```

### Pipeline Overview

| Stage | Description | Tools |
|---|---|---|
| Image Acquisition | Standardised front-view ear photographs | Macro lens, ring light |
| Annotation | Manual labelling of eight acupoints | LabelMe |
| Detection | Automated keypoint localisation | YOLO11-Pose |
| Feature Extraction | Colour and texture measurements from 15 × 15 pixel patches | OpenCV, NumPy |
| Statistical Analysis | Group comparisons and multiple-testing correction | SciPy, statsmodels |
| Visualisation | Detection overlays, statistical plots and result tables | Matplotlib |

## 📍 Auricular Acupoints Studied

Eight auricular points were selected based on published clinical literature and the WFAS international standard of auricular points.

1. Ear Apex
2. Shen Men
3. Superior Triangular Fossa
4. Sympathetic
5. Kidney
6. Heart
7. Pancreas
8. Endocrine

## 🗂️ Repository Structure

```text
.
├── README.md
├── .gitignore
│
└── scripts/
    ├── prepare_yolo_data.py   # LabelMe JSON → YOLO format and dataset split
    ├── data.yaml              # YOLO dataset configuration (8 keypoints)
    ├── train_yolo.py          # Fine-tune YOLO11-Pose
    ├── predict.py             # Detect acupoints in an image
    ├── extract_features.py    # Extract colour and texture features
    └── compare_groups.py      # Statistical group comparison
```

### 📁 Data Directories

The following folders are created or expected during execution and are not included in the repository:

```text
data/
├── images/           # Ear photographs
└── labelme_json/     # LabelMe point annotations

results/              # Generated outputs
├── features.csv
├── stats_omnibus.csv
├── stats_posthoc.csv
├── plots/
└── overlays/
```

## 📊 Dataset

The dataset consists of **193 ear photographs from 97 participants**, collected through a collaborating clinical practitioner with informed consent.

| Group | Participants | Images | Purpose |
|---|---:|---:|---|
| General | 25 | 50 | Detector training only |
| Healthy | 24 | 48 | Detector training and statistical analysis |
| Hypertension | 25 | 50 | Detector training and statistical analysis |
| Diabetes | 23 | 45 | Detector training and statistical analysis |
| **Total** | **97** | **193** | |

### Dataset Naming Convention

Image filenames encode the participant group, participant number, and ear side.

| Filename | Meaning |
|---|---|
| `HE_01_L.jpg` | Healthy, participant 01, left ear |
| `HY_01_R.jpg` | Hypertension, participant 01, right ear |
| `DB_01_L.jpg` | Diabetes, participant 01, left ear |
| `01_L.jpg` | General group, diagnosis unknown |

🔒 **Privacy:** Participant photographs and annotations are not included in this repository. Images were stored using anonymous participant codes. To reproduce the pipeline, provide your own dataset in the expected directory structure.

## ⚙️ Installation & Setup

### Requirements

- Python 3
- pip
- Ear photographs and corresponding LabelMe annotations

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy pandas scipy statsmodels matplotlib
```

The scripts also use NLTK for sentence-independent processing only if required by the relevant implementation; check the script imports before adding any additional dependencies.

## 🚀 Usage

Run all commands from the project root directory, in the following order.

**1. Prepare the YOLO Dataset**

Convert LabelMe annotations into YOLO-compatible format and automatically split the dataset into training and validation sets.

```bash
python scripts/prepare_yolo_data.py
```

**2. Train the YOLO11-Pose Model**

Fine-tune the pose estimation model to detect the eight auricular keypoints.

```bash
python scripts/train_yolo.py
```

**3. Predict Acupoints (Optional)**

Run the trained model on an individual ear photograph. Check the image and model paths configured in the script.

```bash
python scripts/predict.py
```

**4. Extract Colour and Texture Features**

Extract local image features around manually annotated acupoints and generate `features.csv`.

```bash
python scripts/extract_features.py
```

**5. Perform Statistical Analysis**

Compare the extracted features across the three clinical groups and generate statistical reports and plots.

```bash
python scripts/compare_groups.py
```

## 🎨 Feature Extraction

For each photograph, `extract_features.py` reads the group information from the filename and extracts features from a **15 × 15 pixel patch** centred around each manually annotated acupoint.

General-group images are excluded from the statistical feature analysis.

### Extracted Features

| Feature | Description |
|---|---|
| `R`, `G`, `B` | Mean red, green, and blue channel intensities |
| `L` | Mean CIELAB lightness |
| `a` | CIELAB red-green colour axis |
| `b` | CIELAB blue-yellow colour axis |
| `saturation` | Mean HSV saturation |
| `brightness` | Mean HSV value |
| `texture` | Standard deviation of grayscale intensity |
| `patch_pixels` | Number of pixels in the extracted patch |
| `*_rel` | Feature value relative to the mean across all eight points in the same image |

Relative features are calculated to partially account for image-level exposure differences.

The script also generates **visual overlay images** with boxes around each measured patch, allowing the extracted regions to be inspected.

## 📈 Statistical Analysis

The statistical pipeline investigates whether the extracted features differ across the three participant groups.

### Analysis Workflow

1. Excludes Ear Apex from statistical comparison while retaining it for detector localisation.
2. Selects seven acupoints and five predefined features:
   - `a`
   - `a_rel`
   - `L`
   - `saturation`
   - `texture`
3. Averages left and right ear measurements for each participant.
4. Performs Kruskal-Wallis tests across the three groups for all **35 point-feature combinations**.
5. Reports one-way ANOVA results alongside the non-parametric tests.
6. Applies Benjamini-Hochberg FDR correction across the 35 omnibus tests.
7. Performs pairwise Mann-Whitney U tests for combinations with omnibus p-values below 0.05, followed by Holm correction.
8. Reports epsilon-squared effect sizes, group medians, and visualisations of the six strongest results.

## 🏆 Results & Findings

### 1. YOLO11-Pose Detector Performance

The YOLO11-Pose model was fine-tuned using 193 annotated ear photographs on a CPU-only laptop.

Training stopped early at epoch 118 out of 150 after 30 epochs without improvement.

| Training Stage | Dataset | Pose mAP50-95 |
|---|---|---:|
| Pipeline Check | 4 photographs | Not usable |
| Practice Round | 50 photographs, 25 subjects | ~0.87 |
| Final Model | 193 photographs, 97 participants | **0.94** |

**Final validation metrics (epoch 118):**

| Metric | Score |
|---|---:|
| Precision | 1.000 |
| Recall | 1.000 |
| mAP50 | 0.995 |
| Pose mAP50-95 | **0.941** |

### 2. Statistical Group Comparison

Out of 35 point-feature combinations, **five remained statistically significant after FDR correction (q < 0.05).**

| Acupoint | Feature | p-value | FDR q-value | Epsilon-squared |
|---|---|---:|---:|---:|
| Endocrine | L | 7.7e-05 | 0.0027 | 0.267 |
| Shen Men | a_rel | 6.6e-04 | 0.0116 | 0.206 |
| Heart | L | 1.2e-03 | 0.0142 | 0.189 |
| Heart | Saturation | 2.2e-03 | 0.0183 | 0.172 |
| Shen Men | Saturation | 2.6e-03 | 0.0183 | 0.167 |

### Group Medians

| Acupoint – Feature | Healthy | Hypertension | Diabetes |
|---|---:|---:|---:|
| Endocrine – L | 99.0 | 62.5 | 64.9 |
| Shen Men – a_rel | 3.4 | 6.3 | 5.4 |
| Heart – L | 145.4 | 106.1 | 107.2 |
| Heart – Saturation | 98.4 | 124.2 | 127.9 |
| Shen Men – Saturation | 131.5 | 157.6 | 168.0 |

### Key Observations

- **Endocrine and Heart:** Both points showed lower lightness values in the hypertension and diabetes groups compared with the healthy group. The two patient groups did not differ significantly from each other.
- **Shen Men:** Saturation was higher in both patient groups. Relative redness (`a_rel`) was the only result that distinguished hypertension from diabetes, with a borderline Holm-adjusted p-value of 0.049.
- **Heart:** Saturation showed a significant pairwise difference only between healthy and diabetes participants.
- **Kidney, Pancreas and Sympathetic:** These points did not show significant differences after multiple-testing correction.

> 📌 These findings represent exploratory group-level associations. They do not establish causation or provide reliable individual-level classification.

## ⚠️ Limitations

- **Small sample size:** The study includes 24 healthy, 25 hypertension, and 23 diabetes participants.
- **Single-site dataset:** Images were collected through one collaborating clinical source.
- **Uncontrolled variables:** Age, sex, skin tone, medication, and exposure conditions were not controlled.
- **Image exposure:** Exposure parity between groups was not verified, potentially influencing lightness and saturation measurements.
- **No individual classification:** Feature distributions overlap considerably across groups.
- **Manual-point statistics:** Statistical analysis used manually annotated points rather than detector-predicted locations.
- **Limited detector validation:** The validation set is small, with no independent external test set.
- **Multiple comparisons:** Findings are based on 35 tests performed on a single pilot dataset and require independent replication.

## 🔮 Future Work

- Collect a larger, age- and sex-matched participant cohort.
- Introduce colour reference cards during image acquisition to improve colour consistency.
- Include Ear Apex in the statistical comparison and investigate alternative patch sizes.
- Perform statistical analysis using detector-predicted keypoints.
- Model left and right ear measurements jointly.
- Explore ROC analysis if individual features demonstrate sufficient promise.
- Validate the detector across different cameras, imaging conditions, and clinical sites.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Keypoint Detection | Ultralytics YOLO11-Pose |
| Annotation | LabelMe |
| Data Processing | NumPy, pandas |
| Statistical Analysis | SciPy, statsmodels |
| Visualisation | Matplotlib |

## 🎯 Project Highlights

- Developed an end-to-end computer vision pipeline for auricular acupoint localisation and feature analysis.
- Fine-tuned a YOLO11-Pose model achieving **0.941 pose mAP50-95** on the validation set.
- Extracted multi-colour-space and texture features from eight predefined auricular points.
- Conducted participant-level statistical comparisons across three clinical groups.
- Applied FDR and Holm corrections to account for multiple statistical comparisons.
- Investigated exploratory visual associations while explicitly accounting for limitations in clinical interpretation.

---

<p align="center">
  <strong>👩‍💻 Developed as an academic research project</strong>
  <br />
  <sub>Computer Vision • Deep Learning • Statistical Analysis • Healthcare Research</sub>
</p>
