# Pima Indians Diabetes Prediction

A machine learning project for predicting diabetes in Pima Indian women using various classification algorithms with hyperparameter tuning and experiment tracking via Weights & Biases (W&B).

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Models & Results](#models--results)
- [Handling Imbalanced Dataset](#handling-imbalanced-dataset)
- [Experiment Tracking](#experiment-tracking)
- [Usage](#usage)
- [Future Work](#future-work)

## Overview

This project aims to predict the onset of diabetes in Pima Indian women based on diagnostic measurements. The dataset includes medical predictor variables and a binary target variable indicating diabetes presence.

**Key Features:**
- Comprehensive exploratory data analysis (EDA)
- Automated preprocessing pipelines with scikit-learn
- Multiple ML models with hyperparameter tuning via RandomizedSearchCV
- Experiment tracking and visualization with Weights & Biases
- Handling imbalanced datasets with SMOTE and other techniques
- Ensemble methods including Stacking Classifier

## Dataset

**Source:** Pima Indians Diabetes Database  
**Instances:** 768  
**Features:** 8 numeric predictive attributes + 1 target variable

### Features:
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (2 hours in oral glucose tolerance test)
3. **BloodPressure** - Diastolic blood pressure (mm Hg)
4. **SkinThickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction** - Diabetes pedigree function
8. **Age** - Age (years)

**Target Variable:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)

### Data Quality Issues:
- **Class Imbalance:** ~65% Non-diabetic, ~35% Diabetic
- **Invalid Zero Values:** Several features contain biologically implausible zero values that represent missing data:
  - Glucose: X instances (~0.65%)
  - BloodPressure: 35 instances (~4.55%)
  - SkinThickness: 277 instances (~29.55%)
  - Insulin: 374 instances (~48.69%)
  - BMI: 11 instances (~1.43%)

## Installation

### Prerequisites
```bash
Python 3.8+
```

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install imbalanced-learn wandb joblib
```

### Setup
```python
import wandb
wandb.login()  # Enter your W&B API key
```

## Project Structure

```
├── diabetes.csv                 # Dataset
├── pima_indians_diabetes.ipynb  # Main notebook
├── models/                      # Saved models (optional)
└── README.md                    # This file
```

##  Exploratory Data Analysis

### Key Findings:
- **No null values** in the dataset
- **No duplicate rows** detected
- **Class imbalance:** Need to address the 65-35 split
- **Invalid zeros:** Multiple features contain zero values that represent missing data
- **Feature distributions:** Most features show right-skewed distributions
- **Correlations:** Glucose shows the strongest correlation with diabetes outcome

### Visualizations:
- Distribution plots for all features
- Correlation heatmap
- Class balance visualization
- Invalid data analysis per feature

## Data Preprocessing

### Preprocessing Pipeline:

```python
ColumnTransformer with:
├── Median Imputation (for zeros) → SkinThickness, BMI, Insulin
├── Mean Imputation (for zeros) → Glucose, BloodPressure
└── Standard Scaling (optional) → All features (Optional)
```

### Data Split:
- **Training Set:** 70% (537 samples)
- **Validation Set:** 15% (116 samples)
- **Test Set:** 15% (115 samples)

### Key Preprocessing Decisions:
- **Imputation Strategy:** Used median for features with outliers, mean for more normally distributed features
- **Scaling:** Applied StandardScaler for distance-based algorithms (KNN)
- **Invalid Values:** Replaced zeros with imputed values instead of dropping rows

## Models & Results

All models were trained with:
- **Hyperparameter Tuning:** RandomizedSearchCV with 5-fold cross-validation
- **Evaluation Metrics:** Accuracy, ROC-AUC, PR-AUC, MCC, Precision, Recall
- **Class Weighting:** Applied where applicable (Decision Tree, Random Forest, KNN)

## Handling Imbalanced Dataset

### Techniques Explored:
>  **Work in Progress** - Additional imbalance handling techniques being evaluated:
> - SMOTE (Synthetic Minority Over-sampling Technique)
> - ADASYN (Adaptive Synthetic Sampling)
> - SMOTEN (SMOTE for Nominal features)
> - SVMSMOTE (SVM-based SMOTE)
> - SMOTEENN
> - Class Weight Adjustment

**Preliminary Findings:**

###  **Final Findings Summary**

After completing all experiments — from base models to imbalanced handling, tuning, and final calibration — the following insights were observed:

#### **1. Baseline Models**

* Initial models (Decision Tree, Random Forest, Logistic Regression, KNN, etc.) were trained on the original imbalanced dataset.
* Accuracy scores ranged between **0.62 – 0.75** on the validation set, but **ROC-AUC values** were inconsistent (0.69–0.84), showing class imbalance impact.
* Random Forest and KNN achieved the most stable validation results but tended to **overfit** on training data.

#### **2. Imbalanced Data Handling**

* Applied multiple resampling methods: **SMOTE**, **ADASYN**, and **SMOTEENN**.
* **SMOTEENN with KNN** produced the best trade-off between recall and precision for the minority (diabetic) class.
* After balancing, the model achieved roughly **68–70% validation accuracy** with better recall for diabetic cases, improving fairness between classes.

#### **3. Best Model Selection**

* The **KNN model trained on SMOTEENN-balanced data** was chosen as the best-performing model based on overall metrics:

  * **Accuracy (Val):** ~0.695
  * **ROC-AUC (Val):** ~0.73
  * **MCC (Val):** ~0.38
  * It offered consistent recall and precision across both classes, unlike deeper tree-based models that overfit.

#### **4. Calibration and Final Evaluation**

* Final test evaluation on unseen data showed:

  * **Accuracy:** 0.75
  * **ROC-AUC:** 0.843
  * **PR-AUC:** 0.651
  * **MCC:** 0.457
  * **Confusion Matrix:**

    * Non-diabetic: Precision = 0.82, Recall = 0.79
    * Diabetic: Precision = 0.63, Recall = 0.68
* The calibration improved probability reliability (Brier score ↓ from 0.174 → 0.165) even though overall accuracy remained stable.

#### **5. Key Takeaways**

* Calibration slightly improved probability estimates but did not significantly alter accuracy — consistent with expectations for small datasets.
* The final model generalizes well given the limited data size (768 samples).
* **KNN with SMOTEENN + Sigmoid Calibration** provides the best balance between interpretability, fairness, and reliability.

---

## Experiment Tracking

All experiments are tracked using **Weights & Biases (W&B)**.

### Logged Metrics:
- Accuracy, ROC-AUC, PR-AUC, MCC
- Precision & Recall (per class)
- Confusion Matrix
- ROC Curve
- Best hyperparameters

### W&B Dashboard:
Project: `diabetes_experiments`

**Groups:**
- `train`
- `val` 
- `test`

### Viewing Results:
```bash
# Access your W&B dashboard at:
https://wandb.ai/adham_ayman/pima_indians_diabetes_experiments
```
##  Usage

### 1. Clone/Download the Project

### 2. Prepare Your Data
Update the `DATA_DIR` variable with your dataset path:
```python
DATA_DIR = "/path/to/your/diabetes.csv"
```

### 3. Run the Notebook
Execute cells sequentially:
1. Data Loading
2. EDA
3. Preprocessing
4. Model Training (this takes time!)
5. Imbalanced Data Handling
6. Evaluation

### 4. Custom Evaluation
Use the `evaluate_binary_classification()` function:
```python
evaluate_binary_classification(
    model=your_model,
    X=X_test,
    y=y_test,
    model_name="custom_model",
    data_split="test", 
    save_model=True  # Optional: save best model
)
```

##  Future Work

### Planned Improvements:
- [X] Complete comprehensive imbalance handling experiments
- [ ] Feature engineering (polynomial features, interaction terms)
- [ ] Deep learning approaches (Neural Networks)
- [ ] Model interpretability (SHAP values, feature importance)
- [ ] Threshold optimization for clinical use
- [ ] Cross-dataset validation
- [ ] Deployment as web application
- [ ] A/B testing different preprocessing strategies

### Model Optimization:
- [ ] Bayesian hyperparameter optimization
- [ ] Ensemble learning with voting classifiers
- [ ] Time-based validation splits (if temporal data available)

##  Notes

- **Training Time:** Some models (KNN, XGBoost, LightGBM) can take significant time to train, especially with RandomizedSearchCV
- **Randomness:** All models use `random_state=21` for reproducibility


##  Acknowledgments

- Dataset: National Institute of Diabetes and Digestive and Kidney Diseases
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, W&B
---

**Last Updated:** 10/7/2025
**Status:**  In Progress - Completing imbalance handling experiments
