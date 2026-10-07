# ANSC 4040 Mini Project
## Machine Learning-Based Prediction of Missing Animal IDs in Dairy Cow Data

## Project Objective

The objective of this project is to develop and evaluate machine-learning methods for recovering missing `AnimalId` values in a large dairy-cow dataset using animal-level, lactation, reproduction, date, and milking-performance information.

The dataset contains repeated milking records for many animals. Because the same cow can appear many times over time, the final modelling workflow is designed as a **time-aware animal-identification problem** rather than a standard random train/test classification task.

Records with known `AnimalId` values are used to construct historical reference data, train candidate-ranking models, and evaluate performance. Records with missing `AnimalId` values are retained separately for the final inference stage.

---

# Project Status

The project has progressed beyond the initial model-comparison plan.

The completed workflow now includes:

1. Data profiling and cleaning.
2. Chronological train/validation/test splitting.
3. An Extra Trees and KNN benchmark.
4. Development of a structural candidate-retrieval system.
5. Candidate-level distance, trajectory, prototype, residual, recency, and ranking features.
6. A final **Hybrid HistGradientBoostingClassifier (HGB)** reranker.
7. Outer-validation evaluation.
8. Frozen final-test evaluation.
9. Saving the final model, feature order, and model metadata.

The remaining production step is to apply the frozen pipeline to the records with missing `AnimalId`.

---

## 1. Development Environment

The project is developed locally using:

- **IDE:** Visual Studio Code
- **Programming language:** Python
- **Notebook environment:** Jupyter Notebook in VS Code
- **Version control:** Git
- **Repository hosting:** GitHub
- **Main libraries:** pandas, NumPy, scikit-learn, Matplotlib, joblib

The raw dataset is stored locally and is not intended to be committed to GitHub.

---

## 2. Dataset Variables

The source data contain dairy-cow production and milking information.

| Variable | Description |
|---|---|
| `AnimalId` | Unique animal identifier and prediction target |
| `LactationNumber` | Lactation number of the animal |
| `DaysInMilk` | Number of days since the beginning of the current lactation |
| `ReproductionStatus` | Reproductive status, such as Pregnant, Bred, or Open |
| `EventDate` | Date associated with the milking record |
| `Avgmilkflow` | Average milk flow during the milking session |
| `Flow30_60Session` | Milk flow during the 30–60 second period |
| `YieldFirst2Min_Session` | Milk yield during the first two minutes |
| `YieldSession` | Total milk yield during the session |
| `DurationSession_sec` | Milking-session duration in seconds |
| `milking` | Milking session/category indicator used in the earlier benchmark workflow |

The final Hybrid HGB notebook uses the first ten variables above and does not use `milking` in the retained final feature pipeline.

---

## 3. Name Standardization and Schema Consistency

Consistent naming is important because the modelling notebooks reference columns and saved model features by exact name.

### 3.1 Column-Name Standardization

When the final Hybrid HGB notebook loads the modelling dataset, all column labels are first converted to strings and leading/trailing whitespace is removed:

```python
data.columns = data.columns.astype(str).str.strip()
```

A fixed list of required columns is then checked before modelling:

```python
MODEL_COLUMNS = [
    "AnimalId",
    "LactationNumber",
    "DaysInMilk",
    "ReproductionStatus",
    "EventDate",
    "Avgmilkflow",
    "Flow30_60Session",
    "YieldFirst2Min_Session",
    "YieldSession",
    "DurationSession_sec",
]
```

If a required name is missing, the notebook raises an error instead of continuing with an inconsistent schema.

The project therefore keeps the source-data variable names as the **canonical column names** rather than repeatedly renaming them in different notebooks.

### 3.2 Data-Type Standardization

The final modelling notebook also standardizes important data types:

- `AnimalId` → string
- `EventDate` → datetime
- `LactationNumber` → numeric integer
- `DaysInMilk` → numeric integer
- `DurationSession_sec` → numeric integer
- Milk-flow and milk-yield variables → numeric floating point
- `ReproductionStatus` → categorical

This prevents the same variable from being interpreted differently across training, validation, testing, and prediction data.

### 3.3 Derived-Feature Naming

Engineered features use descriptive names that identify their meaning. Examples include:

- `EventDay`
- `LactationStartDay`
- `YieldPerMinute`
- `First2MinFraction`
- `FlowShapeRatio`
- `StartDifference`
- `RecencyDays`
- `ClosestDIMDifference`
- `LocalMinManhattan`
- `LocalMinEuclidean`
- `LactationPrototypeManhattan`
- `CowPrototypeEuclidean`
- `Residual_Avgmilkflow`

Candidate-level identifier columns are also kept consistent:

- `QueryId`
- `CandidateClassId`
- `TrueClassId`
- `IsCorrectCandidate`

The final model stores the exact feature order in a JSON file so that inference uses the same names and ordering as training.

### 3.4 File-Name Standardization

New project files should use:

- lowercase descriptive names,
- `snake_case`,
- no unnecessary spaces,
- no temporary suffixes such as `(2)` or `(5)` in the repository version,
- numbered notebook prefixes when order matters.

Recommended repository notebook names are:

```text
01_import_data.ipynb
02_data_profile.ipynb
03_extra_trees_knn_benchmark.ipynb
04_hybrid_hgb_final.ipynb
```

Recommended dataset names are:

```text
dataset_for_model_building.csv
records_missing_animal_id.csv
```

The current notebooks still contain some legacy local file names, including the misspelling `buliding`, because those paths were used during model development. If these files are renamed, the corresponding path constants in the notebooks must be updated at the same time.

Saved model artifacts already follow a consistent `snake_case` convention:

```text
hybrid_hgb_top160_final.joblib
hybrid_hgb_top160_features.json
hybrid_hgb_top160_metadata.json
```

---

## 4. Data Cleaning

### 4.1 Remove Exact Duplicate Records

The initial data profile identified **60,653 exact duplicate rows**.

Duplicates were removed while keeping the first occurrence:

- Original duplicated rows removed: **60,653**
- Rows remaining: **8,434,768**
- Remaining exact duplicates: **0**

### 4.2 Missing and Zero `Avgmilkflow`

The profile identified:

- **172** missing `Avgmilkflow` values
- **204** zero `Avgmilkflow` values

Records with missing or zero `Avgmilkflow` were removed from the modelling data because this variable is a core milking-performance measurement.

### 4.3 Z-Score Outlier Filtering

The following continuous variables were screened using an absolute Z-score threshold of 3:

- `Avgmilkflow`
- `Flow30_60Session`
- `YieldFirst2Min_Session`
- `YieldSession`
- `DurationSession_sec`

This step removed **53,775 rows**, leaving **8,380,617 rows** before the plausibility-range filter.

### 4.4 Plausibility-Range Filtering

The following conservative ranges were applied:

| Variable | Accepted Range |
|---|---:|
| `LactationNumber` | 1–10 |
| `DaysInMilk` | 0–500 days |
| `Avgmilkflow` | 0.2–8.0 kg/min |
| `Flow30_60Session` | 0.1–10.0 kg/min |
| `YieldFirst2Min_Session` | 0.1–15.0 kg |
| `YieldSession` | 0.5–70.0 kg |
| `DurationSession_sec` | 60–900 seconds |

After threshold filtering, the current modelling dataset contains **6,645,172 rows**.

### 4.5 Missing-`AnimalId` Records

The cleaned model-building export is used for model development.

The current data-profile notebook also separately extracts rows with missing `AnimalId` from the original source dataset and saves them for the later prediction stage. This keeps the unknown-ID records available even though the cleaned model-building dataset is composed of labelled records.

---

## 5. Chronological Split Strategy

Because each cow contributes repeated records over time, a random split could place highly related observations from the same cow and nearby dates in both training and evaluation sets.

The modelling workflow therefore sorts records chronologically by `EventDate` and uses a **60/20/20 outer split**:

- **60% training**
- **20% validation**
- **20% test**

The benchmark notebook produced:

| Split | Rows |
|---|---:|
| Training | 3,985,088 |
| Validation | 1,327,822 |
| Test | 1,332,262 |

The training split contained **7,539 AnimalId classes**.

The final Hybrid HGB notebook also creates an internal chronological split inside the outer training period for historical reference data, reranker training queries, and development queries. This allows model development without using the outer validation or test labels.

---

## 6. Benchmark Models

### 6.1 Extra Trees

The Extra Trees benchmark was revised after the first version was found to be too constrained for a problem with more than 7,000 animal classes.

The optimized benchmark used class-aware sampling, a more realistic tree search, and the derived lactation-timing feature:

```text
LactationStartDate = EventDate - DaysInMilk
```

The best retained Extra Trees validation accuracy was approximately:

**29.23%**

### 6.2 KNN Benchmark

KNN was tested because repeated measurements from the same cow may occupy similar regions of feature space.

Before KNN fitting, numerical and encoded features were standardized using `StandardScaler` because distance-based methods are sensitive to feature scale.

The best KNN validation accuracy was approximately:

**2.86%**

Extra Trees therefore substantially outperformed KNN in the benchmark and motivated a more structured candidate-generation approach rather than a direct nearest-neighbor classifier.

---

## 7. Final Hybrid HGB Pipeline

The final retained system is not a direct 9,000-class classifier. Instead, it is a **candidate retrieval + candidate reranking pipeline**.

### 7.1 Structural Candidate Generation

For each query record, the pipeline builds a historical reference system and retrieves structurally plausible animal candidates.

The retained configuration is:

```text
Global structural retrieval: Top 320 candidates
HGB reranker input: Top 160 candidates
```

Candidate retrieval uses information including:

- lactation number,
- estimated lactation start timing,
- historical recency,
- prior observations of each cow,
- projected next-lactation profiles when appropriate.

### 7.2 Distance and Prototype Features

The final reference system creates standardized numeric features using `StandardScaler`.

The base numeric measurements are:

- `DaysInMilk`
- `Avgmilkflow`
- `Flow30_60Session`
- `YieldFirst2Min_Session`
- `YieldSession`
- `DurationSession_sec`

Additional engineered measurements include:

- `YieldPerMinute`
- `First2MinFraction`
- `FlowShapeRatio`

`ReproductionStatus` is one-hot encoded and combined with the standardized numeric measurements.

The pipeline then calculates candidate-level features based on:

- Manhattan distance,
- Euclidean distance,
- nearest historical measurements,
- lactation prototypes,
- whole-cow prototypes,
- Days-in-Milk proximity,
- recency,
- reproduction-status difference,
- residual differences,
- within-query candidate ranks.

### 7.3 Fresh and Long-Gap Snapshot Training

The final HGB is trained using two complementary historical snapshot families.

**Fresh multi-snapshot training** represents short- and medium-horizon prediction conditions.

**Long-gap multi-snapshot training** explicitly covers reference ages from **1 to 160 days**, which was added after temporal drift was observed during validation.

Combining these two training families improves robustness when the historical reference data become older.

### 7.4 Final Reranker

The final reranker is:

```text
HistGradientBoostingClassifier
```

with **52 candidate-level features**.

The retained HGB configuration uses:

```python
learning_rate = 0.05
max_iter = 400
max_leaf_nodes = 7
min_samples_leaf = 10
l2_regularization = 1.0
early_stopping = True
validation_fraction = 0.10
n_iter_no_change = 20
random_state = 42
```

The completed training run contained:

- **3,377 training queries**
- **539,751 candidate rows**
- **3,377 positive candidate rows**
- **52 final features**

---

## 8. Model Evaluation

### 8.1 Outer Validation

The final Hybrid HGB was evaluated on a fixed **2,000-query** outer-validation sample using only the earlier training period as the historical reference.

Results:

| Metric | Result |
|---|---:|
| Top-160 candidate coverage | **82.05%** |
| Overall accuracy | **41.65%** |
| Seen-`AnimalId` accuracy | **45.17%** |
| Conditional ranking accuracy | **50.76%** |
| Correct predictions | **833 / 2,000** |

The validation analysis showed that accuracy decreased as the reference data became older, which motivated the long-gap snapshot training family.

### 8.2 Frozen Final Test

After the modelling protocol was frozen, the final HGB was evaluated on a 2,000-query chronological test sample.

The historical reference for this test ended on **2021-05-20**.

Results:

| Metric | Result |
|---|---:|
| `AnimalId` seen in reference | **90.60%** |
| Top-160 candidate coverage | **78.30%** |
| Overall test accuracy | **43.55%** |
| Seen-`AnimalId` test accuracy | **48.07%** |
| Conditional ranking accuracy | **55.62%** |
| Correct predictions | **871 / 2,000** |

The final test was run using the frozen model without retraining or additional tuning.

---

## 9. Saved Model Artifacts

The final notebook saves:

```text
xgboost_cache/
├── hybrid_hgb_top160_final.joblib
├── hybrid_hgb_top160_features.json
└── hybrid_hgb_top160_metadata.json
```

Although the directory is named `xgboost_cache` for historical compatibility, the retained final model is a scikit-learn `HistGradientBoostingClassifier`, not XGBoost.

The files contain:

- the trained HGB model,
- the exact 52-feature order required for inference,
- configuration and validation metadata.

---

## 10. Final Prediction Workflow

The final inference workflow is:

1. Load the records with missing `AnimalId`.
2. Apply the same column-name and data-type standardization used during model development.
3. Load the frozen HGB model and saved feature order.
4. Build the historical reference system using only information available before each query.
5. Generate the Global Top-320 structural candidate set.
6. Compute candidate-level distance, prototype, residual, recency, and rank features.
7. Restrict the candidate set to the Top 160 candidates.
8. Reindex features to the saved 52-feature order.
9. Score each candidate with the frozen HGB classifier.
10. Select the highest-probability candidate for each query.
11. Save the predicted `AnimalId` together with enough source information to trace each prediction back to the original record.

No retraining or parameter tuning should occur during this final prediction stage.

---

## 11. Repository Workflow

Recommended notebook order:

```text
01_import_data.ipynb
02_data_profile.ipynb
03_extra_trees_knn_benchmark.ipynb
04_hybrid_hgb_final.ipynb
```

The workflow is:

```text
Raw data
   ↓
Data profiling
   ↓
Duplicate / missing / zero / outlier cleaning
   ↓
Clean labelled modelling dataset
   ↓
Chronological split
   ↓
Extra Trees + KNN benchmark
   ↓
Structural candidate retrieval
   ↓
Hybrid HGB reranker
   ↓
Outer validation
   ↓
Frozen final test
   ↓
Missing-AnimalId prediction
```

This structure keeps data preparation, benchmark modelling, final model development, evaluation, and inference separated and reproducible.
