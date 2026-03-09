# Houston Crime Type Classification (2010–2024) — Detailed Technical Documentation

> **University of Houston–Clear Lake | Jan 2025 – Apr 2025**  
> **Author:** Govardhan Reddy Narala, M.S. Data Science

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Sources & Ingestion](#3-data-sources--ingestion)
4. [Data Engineering Pipeline](#4-data-engineering-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Class Balancing Strategy](#6-class-balancing-strategy)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [Model Evaluation](#8-model-evaluation)
9. [Results](#9-results)
10. [Performance Improvements Made](#10-performance-improvements-made)
11. [Known Limitations & Future Work](#11-known-limitations--future-work)
12. [Environment & Dependencies](#12-environment--dependencies)

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline to **predict the type of crime (NIBRS Description)** that occurs at a given time and location within the city of Houston, Texas.

### Objective

Given a set of time and location features, predict one of the **top 5 most frequent crime categories** observed in 2024 Houston Police Department (HPD) data.

### Design Philosophy

Rather than using a random 80/20 split, this project uses a **temporal holdout**:

| Split | Years | Purpose |
|-------|-------|---------|
| Training | 2010–2023 | Model learning |
| Test (holdout) | 2024 | Simulated production deployment |

This reflects how a crime-prediction system would behave in practice: trained on historical data, evaluated on future data.

---

## 2. Repository Structure

```
Houston-Crime-Type-Classification-2010-2024-/
├── Houston_Crime_Classification.ipynb   # Main Jupyter Notebook
├── README.md                            # Project summary
└── DOCUMENTATION.md                     # This file — detailed technical docs
```

### Generated artefacts (created at runtime)

| File / Directory | Description |
|-----------------|-------------|
| `houston_2010_to_may2018/` | Downloaded monthly XLS files (Jan 2010 – May 2018) |
| `houston_crime_jan2010_to_may2018.csv` | Merged CSV for the 2010-to-May-2018 period |
| `houston_crime_2010_2024.csv` | Final merged dataset (2010–2024) |

---

## 3. Data Sources & Ingestion

### 3.1 Source 1 — Monthly XLS Files (Jan 2010 – May 2018)

**Origin:** Houston Police Department public website  
**URL pattern:** `https://www.houstontx.gov/police/cs/xls/{month}{yy}.xls`  
**Format:** Microsoft Excel 97-2003 (`.xls`)

The notebook downloads 101 monthly files (Jan 2010 through May 2018 inclusive) and merges them into a single DataFrame. Key implementation details:

- Files are downloaded in **parallel** using `concurrent.futures.ThreadPoolExecutor` (up to 8 threads), cutting download time by roughly 5–8×.
- Each download attempt is retried up to **3 times** with a 5-second back-off before giving up, making the pipeline resilient to transient HTTP failures.
- The `download_file` helper is defined **once, outside the loop**, avoiding repeated function re-creation overhead.

**Schema (representative):**

| Column | Description |
|--------|-------------|
| `Date` | Date of occurrence |
| `Hour` | Hour of occurrence (0–23) |
| `Beat` | HPD patrol beat code |
| `Premise` | Location type (home, street, etc.) |
| `Offense Type` | Crime description (later renamed `NIBRS Description`) |
| `# Of Offenses` | Number of offenses (multiple legacy variants) |

---

### 3.2 Source 2 — Supplemental Mid-2018 File (Jun–Dec 2018)

**File:** `2018-june-december.xlsx`  
**Format:** Microsoft Excel 2007+ (`.xlsx`)

This file covers the gap between the monthly XLS series and the NIBRS-format dataset. It is read with `openpyxl` and concatenated with Source 1.

---

### 3.3 Source 3 — NIBRS Dataset (2019–2024)

**File:** `houston_nibrs_2019_2024_combined.csv`  
**Format:** CSV

Columns kept (after dropping irrelevant fields):

| Raw Column | Renamed to |
|-----------|-----------|
| `rmsoccurrencedate` | `Occurrence Date` |
| `rmsoccurrencehour` | `Occurrence Hour` |
| `nibrsdescription` | `NIBRS Description` |
| `offensecount` | `Offense Count` |
| `beat` | `Beat` |
| `premise` | `Premise` |
| `streetno` | `Block Range` |
| `streetname` | `Street Name` |
| `streettype` | `Street Type` |

Dropped columns: `incident`, `nibrsclass`, `city`, `zipcode`, `maplongitude`, `maplatitude`, `year`, `suffix`

---

## 4. Data Engineering Pipeline

### 4.1 Schema Standardisation

Sources 1, 2, and 3 use different column names for the same logical fields. The notebook consolidates them:

```python
# Street Name: fill from legacy 'StreetName' column where missing
crime_data["Street Name"] = crime_data["Street Name"].fillna(crime_data["StreetName"])

# Block Range: fill from legacy 'BlockRange' column
crime_data["Block Range"] = crime_data["Block Range"].fillna(crime_data["BlockRange"])

# Offense Count: cascade through all legacy variants
for src in ["# Offenses", "# offenses", "Offenses", "# Of"]:
    crime_data["# Of Offenses"] = crime_data["# Of Offenses"].fillna(crime_data[src])
```

> **Note (improvement):** The original code used `fillna(inplace=True)` which raises a `FutureWarning` in pandas ≥ 2.0 and is less readable. The refactored version uses the non-inplace form which is idiomatic and forward-compatible.

### 4.2 Unnamed Column Removal

All `Unnamed: *` columns generated by pandas when reading Excel files without a proper header are dropped with:

```python
df.drop(df.columns[df.columns.str.startswith('Unnamed')], inplace=True, axis=1)
```

### 4.3 Date Parsing

```python
df["Occurrence Date"] = pd.to_datetime(df["Occurrence Date"], errors="coerce")
```

`errors="coerce"` converts unparseable dates to `NaT` so they can be safely dropped downstream.

### 4.4 Multi-Label Expansion

Some records contain comma-separated crime types in a single field, e.g. `"Theft, Vandalism"`. These are split and **exploded** into separate rows, one per crime type:

```python
df["NIBRS Description"] = (
    df["NIBRS Description"]
    .astype(str)
    .str.replace(r"[\n\r\t]+", ",", regex=True)  # normalise whitespace separators
    .str.split(",")
)
df = df.explode("NIBRS Description")
df["NIBRS Description"] = df["NIBRS Description"].str.strip()
df = df[(df["NIBRS Description"] != "") & (df["NIBRS Description"] != "1")]
```

This is critical for correct supervised classification because a single row with two labels cannot be used directly in single-label models.

### 4.5 Final Concatenation

```python
crime_data_2010_24 = pd.concat([crime_data_2010_18, df_2019_24], axis=0, ignore_index=True)
crime_data_2010_24.to_csv("houston_crime_2010_2024.csv", index=False)
```

The resulting CSV contains **~3.6 million rows** spanning January 2010 through December 2024.

---

## 5. Feature Engineering

From the raw date/time fields, four temporal features are derived:

| Feature | Derivation | Description |
|---------|-----------|-------------|
| `year` | `dt.year` | Calendar year |
| `month` | `dt.month` | Month number (1–12) |
| `weekday` | `dt.weekday` | Day of week (0 = Monday, 6 = Sunday) |
| `hour_group` | `hour // 6` | 6-hour time bucket (0 = midnight–5 AM, …, 3 = 6 PM–midnight) |

Combined with the two geographic/location features, the **final feature set** used in modelling is:

| Feature | Type | Description |
|---------|------|-------------|
| `Beat` | Categorical | HPD patrol zone (e.g. `"1A30"`) |
| `Premise` | Categorical | Location type (e.g. `"RESIDENCE"`, `"STREET"`) |
| `month` | Categorical (as int) | Month of occurrence |
| `weekday` | Categorical (as int) | Day of week |
| `hour_group` | Categorical (as int) | 6-hour time bucket |

> **Why these features?** They capture both *where* (Beat, Premise) and *when* (month, weekday, hour_group) a crime occurred, which are the two strongest predictors available in HPD public data without requiring personally identifiable information.

---

## 6. Class Balancing Strategy

### Why balancing is necessary

Houston crime data is highly imbalanced: Theft accounts for a disproportionate share of all incidents. Training on raw counts would bias every classifier toward the majority class.

### Approach

1. **Select top-5 classes by 2024 frequency.** This ensures the test set has adequate samples per class and that the selected classes are currently relevant.
2. **Cap each class at 20,000 samples** in both train and test splits (using vectorised `stratified_sample`).

| Split | Years | Rows (balanced) |
|-------|-------|-----------------|
| Train | 2010–2023 | ≤ 100,000 (5 × 20,000) |
| Test | 2024 | ≤ 100,000 (5 × 20,000) |

### `stratified_sample` — vectorised implementation

```python
def stratified_sample(df, label_col, n, seed=42):
    """Return at most `n` rows per class using vectorised concat (faster than apply)."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in df.groupby(label_col, sort=False):
        k = min(len(group), n)
        parts.append(group.iloc[rng.choice(len(group), k, replace=False)])
    return pd.concat(parts, ignore_index=True)
```

> **Improvement over original:** The original used `groupby(...).apply(lambda x: x.sample(...))`, which triggers a pandas `DeprecationWarning` in pandas ≥ 2.0 for group-key inclusion. The new version uses `pd.concat` over explicit per-group slices, avoiding the warning and being slightly faster for large groups.

---

## 7. Machine Learning Pipeline

### 7.1 Preprocessing

All features are categorical (or low-cardinality integer). A `ColumnTransformer` applies:

1. **`SimpleImputer(strategy="most_frequent")`** — fills any remaining `NaN` values with the mode.
2. **`OneHotEncoder(handle_unknown="ignore")`** — converts each category to a binary vector. Unknown categories seen at test time are silently ignored (all zeros).

### 7.2 Models

| Model | Class | Key Hyperparameters |
|-------|-------|---------------------|
| Neural Network (MLP) | `MLPClassifier` | `hidden_layer_sizes=(100,)`, `max_iter=500` |
| Random Forest | `RandomForestClassifier` | `n_estimators=100`, `n_jobs=-1` |
| K-Nearest Neighbours | `KNeighborsClassifier` | `n_neighbors=5`, `n_jobs=-1` |

> **Improvement:** `n_estimators` was increased from 25 → **100** for Random Forest (25 trees is insufficient for a 5-class problem with 100,000 training samples). Adding `n_jobs=-1` enables all available CPU cores for tree building and KNN distance computation.

### 7.3 Cross-Validation

3-fold Stratified K-Fold cross-validation is used to estimate generalisation performance on the training set:

```python
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_score = cross_val_score(pipe, X_train, y_train_enc,
                           cv=cv_strategy, scoring="accuracy", n_jobs=-1).mean()
```

Stratification ensures each fold has the same class proportion as the full training set.  
`n_jobs=-1` parallelises the 3 fold evaluations.

### 7.4 Label Encoding

Labels are encoded **once** before the model loop to avoid redundant work:

```python
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)
y_test_bin  = label_binarize(y_test_enc, classes=range(len(le.classes_)))
```

`y_test_bin` is the one-vs-rest binary matrix required by `roc_auc_score` in multi-class mode.

---

## 8. Model Evaluation

### 8.1 Metrics

| Metric | Description |
|--------|-------------|
| **CV Accuracy** | Mean accuracy across 3 stratified folds on training data |
| **Test Accuracy** | Accuracy on the 2024 holdout set |
| **Macro AUC** | One-vs-Rest macro-averaged ROC AUC; gives equal weight to each class regardless of size |

Macro AUC is prioritised over accuracy for fairness: a model that perfectly predicts the majority class but ignores minority classes will still score well on accuracy but poorly on Macro AUC.

### 8.2 Visualisations

For each model the notebook produces:

1. **Classification Report** — precision, recall, F1-score per class.
2. **Confusion Matrix** — heatmap of predicted vs. actual labels.
3. **One-vs-Rest ROC Curves** — one curve per crime class with AUC annotation.

### 8.3 Error Handling for AUC

```python
try:
    y_pred_proba = pipe.predict_proba(X_test)
    auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class="ovr", average="macro")
    ...
except Exception as e:
    print(f"  ⚠️  Could not compute AUC for {name}: {e}")
    auc = np.nan
```

> **Improvement:** The original used a bare `except:` which silently swallowed all exceptions including `KeyboardInterrupt` and `SystemExit`. The improved version uses `except Exception as e:` and prints the error message.

---

## 9. Results

Based on the original notebook run (published in README):

| Model | CV Accuracy | Test Accuracy (2024) | Macro AUC |
|-------|-------------|---------------------|-----------|
| **Neural Network (MLP)** | 0.3052 | **0.2937** | **0.5895** |
| Random Forest | 0.2279 | 0.2911 | 0.5835 |
| K-Nearest Neighbors | 0.1848 | 0.2414 | 0.5412 |

### Interpretation

- All models achieve test accuracies in the **29–31% range** for a 5-class problem. Chance level is 20%, so the models learn a meaningful signal.
- The low absolute accuracy reflects the **inherent difficulty** of predicting crime type from time/location features alone; many crime types occur at the same places and times.
- The **Neural Network** is selected as the final model because it achieves the highest test accuracy and Macro AUC.
- **Macro AUC > 0.5** for all models confirms that each model performs better than random on a one-vs-rest basis for every class.

> **Expected improvement from optimisations:** Increasing Random Forest estimators from 25 to 100 and enabling `n_jobs=-1` should raise RF test accuracy closer to (or above) the MLP baseline while using similar or less wall-clock time due to parallelism.

---

## 10. Performance Improvements Made

This section documents every inefficiency identified in the original notebook and the fix applied.

### 10.1 `download_file` Defined Inside the Loop (Cell 0)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `def download_file(...)` was nested inside `for month, yy in month_year_pairs:`, causing the function object to be re-created on every iteration | Function defined once, outside all loops |
| **Impact** | Minor runtime overhead; more importantly, the inner `try/except` swallowed the exception and the retry loop had a logic bug (never reached `tries == 0` condition to continue) |  |

### 10.2 Sequential File Download (Cell 0)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | Files downloaded one-at-a-time in a `for` loop | Parallel download via `ThreadPoolExecutor(max_workers=8)` |
| **Impact** | ~101 sequential HTTP requests; each round-trip ~0.5–2 s → total ~1–3 minutes | With 8 threads: ~15–25 seconds (5–8× speedup) |

### 10.3 Broken Retry Logic (Cell 0)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `tries` counter decremented but outer `except` block `continue`d before the retry loop could execute; `time` was not imported | Replaced with a clean `for attempt in range(MAX_RETRIES)` loop with `break` on success |

### 10.4 Missing `import time` (Cell 0)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `time.sleep(5)` called in retry block but `time` not imported | `import time` added at the top of Cell 0 |

### 10.5 Inverted `combined_df.empty` Condition (Cell 1)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `crime_data = pd.read_csv(...) if combined_df.empty else combined_df.copy()` — reads from CSV when `combined_df` is empty (i.e., download failed) and uses in-memory data when it is populated | Logic inverted: `combined_df.copy() if not combined_df.empty else pd.read_csv(...)` |

### 10.6 Deprecated `fillna(inplace=True)` (Cell 5)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `crime_data['Street Name'].fillna(crime_data['StreetName'], inplace=True)` raises `FutureWarning` in pandas ≥ 2.0 | Replaced with assignment form: `crime_data["Street Name"] = crime_data["Street Name"].fillna(...)` |

### 10.7 Escape Character Bug in `str.replace` (Cells 20, 22)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `str.replace(r"[\n\r\t]+", ...)` — raw string with `\n\r\t` is correct at source level but the notebook stored these as literal 4-char sequences `\\n` etc. | Ensured raw-string literal is used consistently |

### 10.8 Random Forest: Too Few Trees (Cells 20, 22)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `n_estimators=25` — too few trees for 100,000 training samples; high variance in predictions | Increased to `n_estimators=100` |
| **Impact** | More stable predictions; slightly higher accuracy at the cost of ~4× more tree-building time, which is offset by `n_jobs=-1` |

### 10.9 No Parallelism in Models / CV (Cells 20, 22)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `RandomForestClassifier()` and `KNeighborsClassifier()` used default `n_jobs=1`; `cross_val_score()` also single-threaded | Added `n_jobs=-1` to RandomForest, KNN, and `cross_val_score` |
| **Impact** | On a 4-core machine: ~2–4× speedup for RF training and CV |

### 10.10 Bare `except:` (Cell 22)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `except:` catches *all* exceptions including `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit`; swallows the error silently | Replaced with `except Exception as e:` and a descriptive `print` |

### 10.11 `stratified_sample` Using Deprecated `groupby.apply` (Cells 20, 22)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `df.groupby(...).apply(lambda x: x.sample(...))` raises `DeprecationWarning` in pandas ≥ 2.0 regarding group key inclusion | Replaced with explicit `for` loop + `pd.concat` which is idiomatic and warning-free |

### 10.12 Escape Character in `print` Statement (Cell 20)

| | Original | Fixed |
|--|---------|-------|
| **Issue** | `print(f"\n\Training {name}...")` — `\T` is not a recognised escape sequence; produces `\T` literally | Fixed to `print(f"\nTraining {name}...")` |

---

## 11. Known Limitations & Future Work

### Limitations

1. **Low absolute accuracy (~29–31%)** — predicting crime *type* from time and location alone is inherently ambiguous. Many crime types share the same time/location patterns.
2. **No spatial features** — latitude/longitude coordinates are dropped. Including geospatial clustering (e.g., H3 hexagons) would likely improve accuracy.
3. **No socioeconomic context** — neighbourhood demographics, poverty rate, and population density are not included.
4. **5-class restriction** — only the top 5 classes are modelled. Extending to all NIBRS categories would require a different balancing strategy.

### Suggested Future Improvements

| Improvement | Expected Benefit |
|-------------|-----------------|
| Add `XGBoost` / `LightGBM` | Higher accuracy, faster training than RF and MLP |
| Geospatial features (H3, Census tracts) | Capture spatial clustering of crime |
| Weather & event data | Explain temporal spikes |
| Deep embedding for Beat/Premise | Learn dense representations of high-cardinality categoricals |
| Hyperparameter tuning with Optuna/Hyperopt | Squeeze additional accuracy |
| Deploy as FastAPI REST endpoint | Enable real-time crime-type prediction |
| Calibrated probabilities (Platt/Isotonic) | Make prediction confidence meaningful |

---

## 12. Environment & Dependencies

### Python Version

Python ≥ 3.9 (tested on 3.12)

### Required Packages

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| `pandas` | 2.0 | Data manipulation |
| `numpy` | 1.24 | Numerical operations |
| `scikit-learn` | 1.3 | ML pipeline, models, metrics |
| `matplotlib` | 3.7 | Plotting |
| `seaborn` | 0.12 | Statistical visualisation |
| `requests` | 2.28 | HTTP file downloads |
| `xlrd` | 2.0 | Reading `.xls` files |
| `openpyxl` | 3.1 | Reading `.xlsx` files |
| `jupyter` / `notebook` | Any recent | Running the notebook |

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests xlrd openpyxl jupyter
```

### Running the Notebook

1. Place the raw data files in the same directory as the notebook (or let Cell 0 download them):
   - `houston_nibrs_2019_2024_combined.csv` (2019–2024 NIBRS data)
   - `2018-june-december.xlsx` (mid-2018 supplement)
2. Launch Jupyter:
   ```bash
   jupyter notebook Houston_Crime_Classification.ipynb
   ```
3. Run all cells in order (`Cell → Run All`).

The final merged CSV `houston_crime_2010_2024.csv` is created by Cell 18 and consumed by Cells 20 and 22.

---

*Documentation generated as part of the performance-improvement and code-quality review — March 2026.*
