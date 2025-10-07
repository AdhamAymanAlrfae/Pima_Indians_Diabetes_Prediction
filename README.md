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

## 🔍 Exploratory Data Analysis

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

## 🛠️ Data Preprocessing

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
- **Scaling:** Applied StandardScaler for distance-based algorithms (SVM, KNN)
- **Invalid Values:** Replaced zeros with imputed values instead of dropping rows

## Models & Results

All models were trained with:
- **Hyperparameter Tuning:** RandomizedSearchCV with 5-fold cross-validation
- **Evaluation Metrics:** Accuracy, ROC-AUC, PR-AUC, MCC, Precision, Recall
- **Class Weighting:** Applied where applicable (Decision Tree, Random Forest, SVM)

### Models Evaluated:

#### 1. Decision Tree (dt)

Best Params → {'criterion': 'log_loss', 'max_depth': 5, 'min_samples_split': 2} 
Train → Acc: 0.746 | ROC-AUC: 0.897 | MCC: 0.558 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/z21hy3o2/overview)
Val → Acc: 0.626 | ROC-AUC: 0.697 | MCC: 0.258 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/pc3n3var/overview)


---

#### 2. K-Nearest Neighbors (knn)

Best Params → {'n_neighbors': 13, 'p': 1, 'weights': 'uniform'} 
Train → Acc: 0.797 | ROC-AUC: 0.853 | MCC: 0.532 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/f0pihdip/overview)
Val → Acc: 0.686 | ROC-AUC: 0.731 | MCC: 0.336 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/mjkyokdu/overview)

---

#### 3. Random Forest (rf)

Best Params → {'bootstrap': True, 'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 209} 
 
Train → Acc: 0.986 | ROC-AUC: 0.999 | MCC: 0.971 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/f45ghpca/overview)
Val → Acc: 0.678 | ROC-AUC: 0.745 | MCC: 0.311 | [Run](https://wandb.ai/adham_ayman/diabetes_experiments/runs/xsd7ux4e/overview)

---

#### 4. Support Vector Machine (svm)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

#### 5. XGBoost (xgb)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

#### 6. LightGBM (lgbm)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

#### 7. CatBoost (cat)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

#### 8. AdaBoost (ada)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

#### 9. Stacking Classifier (stack)

Best Params → {} 
 
Train → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()
Val → Acc: 0. | ROC-AUC: 0. | MCC: 0. | [Run]()

---

## Handling Imbalanced Dataset

### Approach: SMOTEENN (SMOTE + Edited Nearest Neighbors)

**Before Resampling:**
- Label '0' (Non-diabetic): XXX samples
- Label '1' (Diabetic): XXX samples

**After Resampling:**
- Label '0' (Non-diabetic): XXX samples
- Label '1' (Diabetic): XXX samples

### Models Retrained on Balanced Data:

#### Results Summary:

| Model           | Test Accuracy | Test ROC-AUC |  Test MCC   |
|-----------------|---------------|--------------|-------------|
| Decision Tree   | X.XX%         | X.XXX        | X.XXX       |
| Random Forest   | X.XX%         | X.XXX        | X.XXX       |
| SVM             | X.XX%         | X.XXX        | X.XXX       |
| XGBoost         | X.XX%         | X.XXX        | X.XXX       |
| Stacking        | X.XX%         | X.XXX        | X.XXX       |

### Other Techniques Explored:
>  **Work in Progress** - Additional imbalance handling techniques being evaluated:
> - SMOTE (Synthetic Minority Over-sampling Technique)
> - ADASYN (Adaptive Synthetic Sampling)
> - SMOTEN (SMOTE for Nominal features)
> - SVMSMOTE (SVM-based SMOTE)
> - Random Undersampling
> - Class Weight Adjustment

**Preliminary Findings:**
- [To be filled after experiments complete]

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
https://wandb.ai/adham_ayman/diabetes_experiments
```

## 💻 Usage

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

## 🔮 Future Work

### Planned Improvements:
- [ ] Complete comprehensive imbalance handling experiments
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

## 📝 Notes

- **Training Time:** Some models (SVM, XGBoost, LightGBM) can take significant time to train, especially with RandomizedSearchCV
- **Randomness:** All models use `random_state=21` for reproducibility


## 🙏 Acknowledgments

- Dataset: National Institute of Diabetes and Digestive and Kidney Diseases
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn, W&B
---

**Last Updated:** 10/7/2025
**Status:**  In Progress - Completing imbalance handling experiments
