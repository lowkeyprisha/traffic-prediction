# 🚦 Traffic Demand Prediction

> Predicting urban traffic demand using spatio-temporal feature engineering + LightGBM  
> **Hackathon Score: 97+ / 100** &nbsp;|&nbsp; Metric: `max(0, 100 × R²)`

---

## 📌 Problem Statement

Cities worldwide face increasing traffic congestion. This project builds an ML model to predict **traffic demand** at a given geographic location and timestamp — enabling data-driven urban mobility planning.

**Dataset:** 77,299 training rows × 11 columns &nbsp;|&nbsp; 41,778 test rows  
**Target:** `demand` — a normalized float in [0, 1] representing traffic intensity

---

## 🗂️ Repository Structure

```
traffic-demand-prediction/
│
├── traffic_demand_prediction.ipynb   # Full solution notebook
├── submission.csv                    # Final predictions (41778 × 2)
├── approach_and_features.txt         # Detailed feature & approach doc
└── README.md
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

### 🕐 Time Features
- Raw `time_minutes`, `hour`, `minute` extracted from timestamp
- **Cyclic encoding** via `sin`/`cos` of hour and minute — prevents the model from treating 23:45 → 0:00 as a large gap
- Time-of-day buckets: `is_morning`, `is_afternoon`, `is_evening`, `is_night`

### 🌍 Spatial Features
- **Geohash decoded** to actual `lat`/`lon` coordinates using the base-32 bit-interleaving algorithm
- Hierarchical geo prefixes: `geo_prefix3`, `geo_prefix4`, `geo_prefix5` (city → neighbourhood → block)

### ⏪ Lag Features *(most impactful group)*
| Feature | Description |
|---|---|
| `lag_1` | Demand 15 minutes ago |
| `lag_2` | Demand 30 minutes ago |
| `lag_4` | Demand 1 hour ago |
| `lag_9` | Demand ~2.25 hours ago |
| `lag_96` | Demand at the **same time yesterday** |
| `lag_95` / `lag_97` | ±15 min around same time yesterday |
| `lag_ratio_96_1` | `lag_1 / lag_96` — today vs yesterday ratio |
| `lag_diff_1_96` | `lag_1 - lag_96` — absolute change vs yesterday |

> **Key insight:** Test data is Day 49 (timestamps 135–825). Train contains Day 49 timestamps 0–120. So `lag_1` for a test row at `t=135` resolves to `t=120` in training — valid and high-coverage. Earlier versions missed this, causing 98% NaN lags and a much lower score.

### 📈 Rolling Features
- `roll_mean_2` — mean demand over last 30 minutes
- `roll_mean_4` — mean demand over last 1 hour

### 📍 Geohash Aggregate Statistics
| Feature | Description |
|---|---|
| `gh_mean` / `gh_std` / `gh_median` | Overall demand stats per geohash |
| `gh_max` / `gh_min` / `gh_count` | Range and data richness |
| `gh_hour_mean` / `gh_hour_std` | Demand at geohash × hour of day |
| `gh_min_mean` | Demand at geohash × minute |
| `gh_time_mean` | ⭐ Demand at geohash × **exact timestamp** — most granular |
| `geo_prefix4_mean/std` | Neighbourhood-level demand stats |
| `geo_prefix5_mean/std` | Block-level demand stats |

### 🌦️ Road & Weather Features
- `NumberofLanes`, `Temperature` (missing values filled via geohash × hour median)
- Label-encoded: `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`

---

## 🤖 Model

```
Algorithm:       LightGBM (LGBMRegressor)
Objective:       Regression (RMSE)
n_estimators:    Up to 5000 (early stopping: 150 rounds)
learning_rate:   0.02
num_leaves:      511
subsample:       0.75
colsample_bytree: 0.75
reg_alpha:       0.05  (L1)
reg_lambda:      0.10  (L2)
Validation:      5-Fold KFold (shuffle=True, seed=42)
Final output:    Average of 5 models, clipped to [0, 1]
```

---

## 📉 Results

| Metric | Value |
|---|---|
| OOF R² | 0.9967 |
| CV Score (0–100) | **99.67** |
| Hackathon Score | **97+** |

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/traffic-demand-prediction.git
cd traffic-demand-prediction
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

This will generate `submission.csv` in the same folder.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-brightgreen?style=flat)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-CV%20%26%20metrics-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## 👩‍💻 Author

**Prisha** — connect on [GitHub](https://github.com/YOUR_USERNAME)
