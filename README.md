# ANSC 4040 Mini Project
## Machine Learning-Based Prediction of Missing Animal IDs in Dairy Cow Data

## Project Objective

The objective of this project is to develop and evaluate machine learning models that can predict missing `AnimalId` values in a dairy cow dataset using information from the remaining animal-level and milking-session variables.

The dataset contains repeated milking records and includes variables such as lactation number, days in milk, reproductive status, milk flow, milk yield, milking duration, event date, and milking session. Because `AnimalId` is missing for a substantial subset of records while many production measurements remain available, the project will investigate whether patterns in these features can be used to identify the most likely animal associated with a record.

This is treated as a **supervised multiclass classification problem**. Records with a known `AnimalId` will be used for model development and evaluation, while records with a missing `AnimalId` will be retained separately for the final prediction step.

---

# Project Plan

## 1. Updated Timeline

| Week | Plan |
| --- | --- |
| **Week 1** | Explore the dataset, identify duplicated records, quantify missing values, and inspect zero or implausible production values |
| **Week 2** | Clean the data, remove invalid observations, define plausible biological/production ranges, and prepare labelled and unlabelled datasets |
| **Week 3** | Encode categorical/date variables, create the train/test split, establish a baseline model, and train several classification models |
| **Week 4** | Compare model performance, select the most appropriate model, predict missing `AnimalId` values, organize the code, and finalize the README |

---

## 2. Development Environment

The project will be developed **locally** on my personal computer.

The primary development environment will be:

- **IDE:** Visual Studio Code
- **Programming Language:** Python
- **Notebook Environment:** Jupyter Notebook in VS Code
- **Version Control:** Git
- **Repository Hosting:** GitHub
- **Main Libraries:** pandas, NumPy, scikit-learn, Matplotlib

The raw dataset will remain stored locally and will not be uploaded to GitHub. It will be excluded from version control using `.gitignore`.

GitHub will be used to track changes in the code, data-cleaning workflow, model development, model evaluation, and project documentation.

---

## 3. Dataset Variables and Naming Convention

The dataset contains dairy cow production and milking information.

| Variable | Description |
|---|---|
| `AnimalId` | Unique identification value for each animal and the prediction target |
| `LactationNumber` | Lactation number of the animal |
| `DaysInMilk` | Number of days since the beginning of the current lactation |
| `ReproductionStatus` | Reproductive status of the animal, such as Pregnant, Bred, or Open |
| `EventDate` | Date associated with the milking record |
| `Avgmilkflow` | Average milk flow during the milking session |
| `Flow30_60Session` | Milk flow measured during the 30–60 second period of the milking session |
| `YieldFirst2Min_Session` | Milk yield during the first two minutes of the milking session |
| `YieldSession` | Total milk yield during the milking session |
| `DurationSession_sec` | Duration of the milking session in seconds |
| `milking` | Milking session/category indicator |

The dataset contains both numerical and categorical variables. Numerical production variables describe the milking performance of each record, while variables such as `ReproductionStatus` and `milking` provide additional categorical information that may help distinguish animals.

---

## 4. Data Cleaning Strategy

Data cleaning will be performed before model training so that obvious data-quality problems do not influence the model.

### 4.1 Remove Exact Duplicate Records

The initial data profile identified **60,653 duplicated rows**. These exact duplicates are removed while keeping the first occurrence. After this step, 8,434,768 rows remain and no exact duplicated rows remain.

Removing duplicated observations prevents identical records from being counted multiple times and reduces the possibility that duplicated data artificially affects model training and evaluation.

### 4.2 Separate Labelled and Unlabelled Records

Because the goal is to predict missing `AnimalId` values, records will be separated according to whether the target is available:

- **Labelled data:** rows with a known `AnimalId`; used for model training and evaluation.
- **Unlabelled data:** rows with a missing `AnimalId`; retained for the final prediction step.

Missing target values will therefore **not** be treated as ordinary rows to delete during the final workflow. Cleaning rules will be applied to predictor variables without removing a row simply because `AnimalId` is missing.

This separation is important because the final model must be applied to the records whose `AnimalId` is currently unknown.

### 4.3 Handle Missing and Zero Production Measurements

The current data profile identified 172 missing values in `Avgmilkflow` and 204 zero values before the cleaning steps were completed.

Since average milk flow is a required milking-performance measurement, records with a missing or zero `Avgmilkflow` are removed from the modelling dataset.

Other missing predictor values will be reviewed before modelling. For variables required by a model, the strategy will be either to remove records with unusable predictor information or to apply an appropriate preprocessing/imputation method inside the modelling pipeline.

### 4.4 Statistical Outlier Filtering

A Z-score filter is used for the main continuous milking-performance variables:

- `Avgmilkflow`
- `Flow30_60Session`
- `YieldFirst2Min_Session`
- `YieldSession`
- `DurationSession_sec`

For each variable, the Z-score is calculated using the mean and standard deviation of the cleaned dataset. A record is flagged as an outlier when at least one of these variables has an absolute Z-score greater than 3.

In the current data profile, this procedure removes **53,775 rows**, leaving 8,380,617 rows before the additional plausibility screening.

The Z-score step is intended to remove statistically extreme measurements, but it is not used alone because a statistically unusual value is not automatically biologically impossible.

### 4.5 Plausibility-Range Filtering

After Z-score filtering, predefined conservative ranges are applied to variables where obviously unrealistic values could reduce model quality.

| Variable | Accepted Range |
|---|---:|
| `LactationNumber` | 1–10 |
| `DaysInMilk` | 0–500 days |
| `Avgmilkflow` | 0.2–8.0 kg/min |
| `Flow30_60Session` | 0.1–10.0 kg/min |
| `YieldFirst2Min_Session` | 0.1–15.0 kg |
| `YieldSession` | 0.5–70.0 kg |
| `DurationSession_sec` | 60–900 seconds |

The current notebook removes records outside these ranges after the Z-score step. This produces 6,645,172 records in the current threshold-filtered dataset.

For the final prediction workflow, these predictor-based rules will be applied consistently to both labelled and unlabelled data. The target variable `AnimalId` itself will not be included as a criterion for deleting records that are intended for prediction.

### 4.6 Feature Preparation

Before model training, predictors will be converted into a machine-learning-ready format.

Planned preprocessing includes:

- Convert `EventDate` to a proper datetime variable.
- Extract potentially useful date components such as year, month, day, or day of year when appropriate.
- Encode categorical variables such as `ReproductionStatus` and `milking`.
- Keep the numerical milking variables as numeric predictors.
- Ensure the same preprocessing steps are applied to the training set, test set, and final missing-`AnimalId` dataset.
- Use a scikit-learn preprocessing pipeline where possible to reduce data leakage and keep preprocessing reproducible.

---

## 5. Train/Test Split Strategy

The cleaned records with a known `AnimalId` will be divided into training and testing datasets.

The current notebook uses an **80/20 split**:

- **80% training data**
- **20% testing data**
- `random_state=42` for reproducibility

Because `AnimalId` is a multiclass target and the number of records may differ among animals, the final split will use **stratification by `AnimalId` when feasible**. This will help preserve the distribution of animal classes between the training and testing datasets.

The planned split is therefore:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Before using stratification, the number of observations per animal will be checked because classes with too few observations may require separate handling.

A second validation approach may also be considered using `EventDate`. For example, earlier records can be used for training and later records for testing. A time-based holdout would provide a more difficult evaluation of whether an animal-identification model continues to perform well on records collected later in time.

All preprocessing parameters that are learned from the data will be fitted using the training set only and then applied to the test set. The final unlabelled records will not be used to evaluate or tune the model.

---

## 6. Modelling Strategy

Since `AnimalId` is a categorical identifier with potentially many unique values, the problem will be approached as **multiclass classification** rather than regression.

Several modelling techniques will be compared instead of selecting one model in advance.

### 6.1 Baseline Classifier

A simple baseline classifier will first be created, for example using `DummyClassifier`.

The baseline provides a minimum reference point for model performance. More complex models should perform meaningfully better than this baseline before they are considered useful.

### 6.2 Logistic Regression

Multiclass Logistic Regression will be tested as an interpretable linear baseline after categorical variables are encoded.

Advantages:

- Provides a simple supervised-learning benchmark.
- Works well with standardized or appropriately encoded predictors.
- Helps determine whether the relationship between the predictors and animal identity can be captured with a relatively simple decision boundary.

Limitations:

- The dataset is very large and `AnimalId` may contain many classes.
- Training may become computationally expensive.
- Linear decision boundaries may not capture complex relationships among milking variables.

For these reasons, Logistic Regression may initially be tested on a manageable training subset.

### 6.3 Decision Tree

A Decision Tree classifier will be used as a nonlinear baseline.

Advantages:

- Can capture nonlinear relationships and interactions.
- Does not require feature scaling.
- Provides an interpretable structure for understanding how variables separate records.

The main limitation is that a single tree can overfit, so its test performance will be compared carefully with ensemble methods.

### 6.4 Random Forest

Random Forest is one of the main models planned for this project.

It combines many decision trees and can model nonlinear interactions among variables such as milk yield, milk flow, lactation stage, reproductive status, and milking duration.

Advantages:

- Handles nonlinear relationships.
- Can model interactions among predictors.
- Requires relatively little feature scaling.
- Provides feature-importance information.

Because the dataset contains millions of records, memory use and training time will be considered. Initial model development may use a representative subset of the training data before fitting a larger final model.

### 6.5 Extra Trees

`ExtraTreesClassifier` will also be considered as an ensemble-tree model.

Extra Trees uses additional randomization when constructing trees and can be computationally competitive with Random Forest. Comparing the two models will help determine whether the additional randomization improves generalization for this dataset.

### 6.6 Nearest-Neighbor Approach

A nearest-neighbor method may be explored on a reduced or carefully selected feature set.

The idea is that records from the same cow may have similar patterns in lactation stage, milk yield, milk flow, and milking duration. However, standard K-Nearest Neighbors can be computationally expensive with millions of rows, so it will be treated as an exploratory model rather than the expected final model.

### 6.7 Model Selection

Model selection will be based on performance on data that were not used for model fitting.

Because the target is multiclass, planned evaluation measures include:

- **Accuracy:** proportion of test records assigned the correct `AnimalId`.
- **Balanced accuracy:** useful if some animals have many more observations than others.
- **Macro F1-score:** gives each animal class equal importance when summarizing precision and recall.
- **Weighted F1-score:** summarizes F1 performance while accounting for the number of observations in each class.
- **Top-k accuracy, if appropriate:** useful for examining whether the true animal is among the model's highest-probability candidates.

A confusion matrix may also be examined for a subset of animals to identify which animals are most frequently confused with one another.

The final model will be selected using test/validation performance together with practical considerations such as computation time, memory requirements, and the ability to generate predictions for the missing-`AnimalId` records.

---

## 7. Final Prediction Workflow

The planned workflow is:

1. Load the raw dataset.
2. Remove exact duplicate rows.
3. Separate records with known and missing `AnimalId`.
4. Clean predictor variables and apply the same plausibility rules to both groups.
5. Prepare numerical, categorical, and date-based features.
6. Split the labelled records into training and testing datasets.
7. Fit preprocessing steps on the training data only.
8. Train the baseline and candidate classification models.
9. Compare model performance using the held-out data.
10. Select the most appropriate model.
11. Refit the selected modelling pipeline using the available labelled data if appropriate.
12. Predict `AnimalId` for the retained unlabelled records.
13. Save the predictions together with enough record information to trace each prediction back to the original data.

---

## 8. File Naming

Project files will use descriptive names without unnecessary spaces.

Jupyter Notebook files will follow a numbered workflow so that the analysis can be read in the correct order.

Planned notebook names:

```text
01_data_import.ipynb
02_exploratory_data_analysis.ipynb
03_data_preprocessing.ipynb
04_model_training.ipynb
05_model_evaluation.ipynb
06_missing_animalid_prediction.ipynb
```

This structure separates data inspection, preprocessing, model development, evaluation, and final prediction so that the project remains reproducible and easy to follow.
