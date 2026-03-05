# 🚢 Titanic Survival Prediction — Complete ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A complete, beginner-friendly machine learning pipeline predicting Titanic passenger survival — built step by step, with every decision explained.

---

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.693 | 0.712 | 0.759 |
| Decision Tree | 0.687 | 0.674 | 0.747 |
| **Random Forest** ⭐ | **0.715** | **0.727** | **0.765** |
| SVM | 0.687 | 0.702 | 0.750 |
| XGBoost | 0.709 | 0.718 | 0.761 |

---

## 📁 Project Structure
```
titanic-survival-prediction/
├── src/
│   ├── step1_load_data.py
│   ├── step2_eda.py
│   ├── step3_preprocessing.py
│   ├── step4_models.py
│   ├── step5_evaluation.py
│   ├── step6_tuning.py
│   ├── step7_feature_importance.py
│   └── step8_predict.py
├── data/
│   └── titanic.csv
├── models/
│   ├── best_random_forest_tuned.pkl
│   ├── scaler.pkl
│   ├── scale_cols.pkl
│   └── feature_cols.pkl
├── outputs/
│   ├── 01_eda_overview.png
│   ├── 02_model_evaluation.png
│   └── 03_feature_importance.png
├── run_all.py
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `train.csv` from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data), rename it to `titanic.csv`, and place it inside the `data/` folder.

### 5. Run the full pipeline
```bash
python run_all.py
```

Or run steps individually:
```bash
python src/step1_load_data.py
python src/step2_eda.py
python src/step3_preprocessing.py
python src/step4_models.py
python src/step5_evaluation.py
python src/step6_tuning.py
python src/step7_feature_importance.py
python src/step8_predict.py
```

---

## 🔧 Pipeline Overview

### Step 1 — Data Loading
- Load CSV and normalize all column names to lowercase
- Inspect shape, dtypes, and missing value counts

### Step 2 — Exploratory Data Analysis
- Survival rate by Sex, Pclass, Age, Embarked
- Fare distribution (raw vs log-transformed)
- Correlation heatmap and Pclass × Sex interaction

### Step 3 — Preprocessing & Feature Engineering

| Feature | Source | Why |
|---------|--------|-----|
| `title` | `name` (regex extract) | Captures sex + age-group + social status |
| `family_size` | `sibsp + parch + 1` | Combined family signal |
| `is_alone` | `family_size == 1` | Binary solo-traveler flag |
| `fare_log` | `log1p(fare)` | Corrects right-skewed distribution |
| `age_band` | `age` (pd.cut bins) | Captures non-linear age effect |
| `sex_encoded` | `sex` | Binary: female=1, male=0 |

**Rules applied:**
- ✅ Age imputed with **median by title group**, not global median
- ✅ Train/test split **before** scaling
- ✅ Scaler fitted **only** on training data
- ✅ One-hot encoding for all nominal categories

### Step 4 — Model Building
Five classifiers trained and compared:
- **Logistic Regression** — interpretable baseline
- **Decision Tree** — rule-based, `max_depth=5`
- **Random Forest** — 100-tree ensemble, best performer
- **XGBoost** — gradient boosting
- **SVM** — RBF kernel

### Step 5 — Evaluation
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix per model
- ROC curves overlaid for visual comparison
- 5-fold cross-validation scores

### Step 6 — Hyperparameter Tuning
GridSearchCV on Random Forest:
```python
param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2'],
}
```

### Step 7 — Feature Importance
Top features by Mean Decrease in Impurity:
1. `fare_log` (~0.21)
2. `age` (~0.20)
3. `sex_encoded` (~0.14)
4. `pclass` (~0.10)
5. `title_Mr` (~0.05)

### Step 8 — Prediction Pipeline
```python
passenger = {
    "pclass": 1, "sex": "female", "age": 25,
    "sibsp": 0, "parch": 0, "fare": 100, "embarked": "C"
}
# → SURVIVED (94.7% probability)
```

---

## 📦 Requirements
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
xgboost>=1.7.0
joblib>=1.2.0
```

---

## 💡 Key Learnings

| Concept | Lesson |
|---------|--------|
| Data Leakage | Split before scaling — never fit scaler on test data |
| Class Imbalance | 38% survival — use F1 + AUC, not accuracy alone |
| Feature Engineering | `title` from name outperformed switching algorithms |
| Missing Values | Group median imputation beats global median |
| Cross-Validation | More reliable than a single train/test split |
| Hyperparameter Tuning | Gains are modest (~1-2%) — features matter more |

---

## ⚠️ Common Mistakes Avoided

| Wrong | Right |
|-------|-------|
| `scaler.fit_transform(X_test)` | `scaler.transform(X_test)` |
| `np.log(fare)` | `np.log1p(fare)` — handles fare=0 |
| Label encode `embarked` | One-hot encode nominal categories |
| Global median for age | Median by title group |
| `eval_metric='logless'` | `eval_metric='logloss'` |
| `df.concat(...)` | `pd.concat(...)` |

---

## 🚀 What to Try Next

- [ ] Wrap preprocessing in `sklearn.Pipeline` + `ColumnTransformer`
- [ ] Handle class imbalance with SMOTE or `class_weight='balanced'`
- [ ] Try LightGBM or CatBoost
- [ ] Add SHAP values for individual prediction explanations
- [ ] Deploy with Flask or FastAPI
- [ ] Submit to the [Kaggle Titanic leaderboard](https://www.kaggle.com/competitions/titanic)

---

## 📄 License

MIT License — free to use, modify, and distribute.
