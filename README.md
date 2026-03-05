# 🚢 Titanic Survival Prediction — Complete ML Pipeline

An end-to-end machine learning pipeline predicting Titanic passenger survival.

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

## ✨ Features

- 🔍 **Exploratory Data Analysis** — survival breakdowns by sex, class, age, fare, and embarkation port with 6 auto-saved charts
- 🛠️ **Feature Engineering** — extracts title from passenger name, builds family size, log-transforms fare, and bins age into groups
- 🧹 **Smart Preprocessing** — group-based missing value imputation, one-hot encoding, stratified train/test split, and StandardScaler with zero data leakage
- 🤖 **5 Models Compared** — Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVM trained and benchmarked side by side
- 📈 **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves, and 5-fold cross-validation
- ⚙️ **Hyperparameter Tuning** — GridSearchCV with 5-fold CV on Random Forest across 240+ parameter combinations
- 📊 **Feature Importance** — bar chart of top 15 features with interpretation, validating EDA findings
- 🔮 **Prediction Pipeline** — predict survival for any new passenger using saved model artifacts (`.pkl` files)

---

## 📄 License

MIT License — free to use, modify, and distribute.
