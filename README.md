# 🚦 Bengaluru Traffic Demand Prediction
### Predicting urban mobility intensity using spatio-temporal feature engineering and LightGBM

🚀 **Hackathon Score:** 84 / 100 &nbsp;|&nbsp; 🎯 **Metric:** `max(0, 100 × R²)` &nbsp;|&nbsp; 📉 **OOF R²:** 0.9967

---

## 📌 The Bengaluru Traffic Challenge

Bengaluru's traffic is legendary — driven by rapid urbanization, massive tech parks (like Silk Board, Electronic City, and Manyata), and unpredictable weather. Standard routing apps tell you *where* congestion is right now, but urban planners and fleet operators need to predict **where demand will spike next**.

This project builds a highly optimized Machine Learning pipeline to forecast traffic demand intensity at any specific geographic chunk and timestamp across the city. By unlocking **predictive insights**, this model enables:

- **Dynamic Fleet Relocation** — Helping ride-hailing services pre-stage vehicles before peak hours
- **Smart Signal Timing** — Allowing municipal systems to proactively adjust green-light windows
- **Bottleneck Mitigation** — Identifying high-demand propagation across micro-neighborhoods

---

## 🗂️ Repository Structure

```
bengaluru-traffic-prediction/
│
├── traffic_demand_prediction.ipynb   # End-to-end ML pipeline (EDA to Inference)
├── submission.csv                    # Final test predictions (41,778 × 2)
├── approach_and_features.txt         # Deep-dive documentation on lag mechanics
└── README.md                         # Project overview
```

---

## 📊 Dataset Features

| Column | Description |
|---|---|
| `Index` | Unique row ID |
| `geohash` | Geographic location encoded as a geohash string |
| `day` | Day number when the record was captured |
| `timestamp` | Time of record in `H:MM` format |
| `RoadType` | Residential / Street / Highway |
| `NumberofLanes` | Number of lanes at the location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at the location |
| `Weather` | Sunny / Rainy / Foggy / Snowy |
| `demand` | ⭐ Target — traffic demand at that timestamp *(train only)* |

---

## ⚙️ Feature Engineering

### 🌍 1. Spatial Partitioning (Decoding the City Grid)

Bengaluru doesn't move uniformly; traffic in Indiranagar behaves differently than on the Outer Ring Road.

- **Geohash → Coordinates:** Decoded the string locations into precise latitude and longitude using the base-32 bit-interleaving algorithm
- **Hierarchical Geo-Prefixes:** Grouped locations into `geo_prefix3` (city-scale), `geo_prefix4` (neighbourhood-scale, e.g., Koramangala), and `geo_prefix5` (block-scale, e.g., a specific tech park gate) — captures how congestion spills from local streets to major arterials

### 🕐 2. Temporal Coherence & Cyclical Commutes

- **Cyclic Encoding:** Traffic at 23:45 and 00:00 is highly continuous. Raw time numbers confuse models; `sin`/`cos` transformations map time onto a continuous 24-hour wheel, eliminating the artificial midnight gap
- **Time-of-Day Buckets:** `is_morning`, `is_afternoon`, `is_evening`, `is_night`
- Raw `time_minutes`, `hour`, `minute` extracted from timestamp string

### ⏪ 3. Lag Features *(most impactful group)*

Traffic is highly recursive — what happened 15 minutes ago dictates what happens next.

| Feature | Description |
|---|---|
| `lag_1` | Demand 15 minutes ago |
| `lag_2` | Demand 30 minutes ago |
| `lag_4` | Demand 1 hour ago |
| `lag_9` | Demand ~2.25 hours ago |
| `lag_96` | Demand at the **same time yesterday** (baseline commute rhythm) |
| `lag_95` / `lag_97` | ±15 min around same time yesterday |
| `lag_ratio_96_1` | `lag_1 / lag_96` — today vs yesterday ratio |
| `lag_diff_1_96` | `lag_1 - lag_96` — absolute change vs yesterday |

> **Key Insight:** Test data is Day 49 (timestamps 135–825). Train contains Day 49 timestamps 0–120. So `lag_1` for a test row at `t=135` resolves to `t=120` in training — valid and high-coverage. Earlier iterations missed this, causing 98% NaN lags and a significantly lower score. Once fixed, model accuracy stabilized dramatically.

### 📈 4. Rolling Features

- `roll_mean_2` — mean demand over last 30 minutes
- `roll_mean_4` — mean demand over last 1 hour

### 📍 5. Geohash Aggregate Statistics

| Feature | Description |
|---|---|
| `gh_mean` / `gh_std` / `gh_median` | Overall demand stats per geohash |
| `gh_max` / `gh_min` / `gh_count` | Range and data richness |
| `gh_hour_mean` / `gh_hour_std` | Demand at geohash × hour of day |
| `gh_min_mean` | Demand at geohash × minute |
| `gh_time_mean` | ⭐ Demand at geohash × **exact timestamp** — most granular feature |
| `geo_prefix4_mean/std` | Neighbourhood-level demand stats |
| `geo_prefix5_mean/std` | Block-level demand stats |

### 🌦️ 6. Road & Weather Features

- `NumberofLanes`, `Temperature` — missing values filled via geohash × hour median, then global median
- Label-encoded: `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`

---

## 🤖 Model

```
Algorithm:        LightGBM (LGBMRegressor)
Objective:        Regression (RMSE)
n_estimators:     Up to 5000 (early stopping: 150 rounds)
learning_rate:    0.02
num_leaves:       511
subsample:        0.75
colsample_bytree: 0.75
reg_alpha:        0.05   (L1 regularisation)
reg_lambda:       0.10   (L2 regularisation)
Validation:       5-Fold KFold (shuffle=True, seed=42)
Final output:     Average of 5 fold models, clipped to [0, 1]
```

---

## 📉 Results

| Metric | Value |
|---|---|
| OOF R² | 0.9967 |
| CV Score (0–100) | **99.67** |
| Hackathon Score | **84 / 100** |

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/lowkeyprisha/bengaluru-traffic-prediction.git
cd bengaluru-traffic-prediction
```

**2. Install dependencies**
```bash
pip install lightgbm scikit-learn pandas numpy matplotlib seaborn
```

**3. Add the dataset files**

Place `train.csv`, `test.csv`, and `sample_submission.csv` in the root folder.

**4. Run the notebook**
```bash
jupyter notebook traffic_demand_prediction.ipynb
# OR open in VS Code and click Run All
```

Running all cells will generate `submission.csv` in the same folder.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-brightgreen?style=flat)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-CV%20%26%20metrics-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## 👩‍💻 Author

**Prisha** — [github.com/lowkeyprisha](https://github.com/lowkeyprisha)
