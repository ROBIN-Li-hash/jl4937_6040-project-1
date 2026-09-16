# ANSC 4040 Mini Project
## Machine Learning-Based Missing Value Prediction in Dairy Cow Data

## Project Objective

The objective of this project is to use machine learning techniques to predict and impute missing values in a dairy cow dataset based on information available from the existing features.

The dataset contains animal-level and milking-related variables, including lactation number, days in milk, reproductive status, milk flow, milk yield, and milking duration.

Rather than relying only on simple imputation methods such as mean or median replacement, this project will investigate whether machine learning models can learn relationships among the existing variables and provide more accurate estimates for missing observations.

---

# Project Plan
## 1. Timeline

| Week | Plan |
| --- | --- |
| **Week 1** | Explore the dataset and identify the missing values |
| **Week 2** | Clean the data and prepare variables for machine learning |
| **Week 3** | Try several machine learning models and compare their performance |
| **Week 4** | Select a model, predict the missing values, organize the code, and complete the README |

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

GitHub will be used to track changes in the code, project development, model testing, and documentation.

---

## 3. Naming Convention

The dataset contains dairy cow production and milking information.

| Variable | Description |
|---|---|
| `AnimalNumber` | Unique identification number for each animal |
| `LactationNumber` | Lactation number of the animal |
| `DaysInMilk` | Number of days since the beginning of the current lactation |
| `ReproductionStatus` | Reproductive status of the animal, such as Pregnant, Bred, or Open |
| `Avgmilkflow` | Average milk flow during the milking session |
| `Flow30_60Session` | Milk flow measured during the 30–60 second period of the milking session |
| `YieldFirst2Min_Session` | Milk yield during the first two minutes of the milking session |
| `YieldSession` | Total milk yield during the milking session |
| `DurationSession_sec` | Duration of the milking session in seconds |
| `milking` | Milking session/category indicator |

The dataset contains both:

- **Numerical variables**, such as milk yield, milk flow, days in milk, and milking duration.
- **Categorical variables**, such as reproductive status.

These variables can potentially provide useful information for predicting missing observations.

---


### File Naming

Project files will use descriptive names without unnecessary spaces.

Jupyter Notebook files will follow a numbered workflow so that the analysis can be read in the correct order.

Planned notebook names:

```text
01_data_import.ipynb
02_exploratory_data_analysis.ipynb
03_data_preprocessing.ipynb
04_model_training.ipynb
05_model_evaluation.ipynb
06_missing_value_prediction.ipynb
