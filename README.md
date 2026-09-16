# jl4937_6040-project-1
jl4937_6040 project 1
# ANSC 4040 Machine Learning Project  
## Missing Data Prediction in Dairy Cow Milking Records

## 1. Project Overview

This project aims to apply machine learning techniques to predict and fill in missing values in a dairy cow dataset.

The dataset contains information related to individual animals, lactation status, reproductive status, and milking performance. The main objective is to use the existing observed variables as predictors and train machine learning models to estimate the values of fields that are missing or intentionally removed.

Rather than simply replacing missing values using basic statistical methods such as the mean or median, this project explores whether machine learning can capture relationships among biological and production-related variables and provide more accurate predictions.

---

## 2. Project Objective

The main goal of this project is to:

> **Predict and fill in missing fields in the dataset based on the existing available information using machine learning techniques.**

The general workflow includes:

1. Importing and inspecting the dataset.
2. Identifying missing values.
3. Exploring relationships among variables.
4. Preprocessing numerical and categorical variables.
5. Dividing available observations into training and testing datasets.
6. Training machine learning models using the existing fields as predictors.
7. Evaluating model performance.
8. Using the selected model to predict the missing values.
9. Filling the predicted values back into the dataset.

---

## 3. Dataset Description

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

## 4. Machine Learning Approach

The project treats missing-value estimation as a supervised machine learning problem.

For a variable containing missing observations:

- Rows where the target variable is available are used for model training and evaluation.
- Other available variables are used as predictor features (`X`).
- The variable containing missing values is treated as the prediction target (`y`).
- After evaluating the model, the trained model is applied to rows where the target value is missing.

Conceptually:

```text
Existing Variables
       ↓
Data Preprocessing
       ↓
Train / Test Split
       ↓
Machine Learning Model
       ↓
Model Evaluation
       ↓
Prediction of Missing Values
       ↓
Completed Dataset
