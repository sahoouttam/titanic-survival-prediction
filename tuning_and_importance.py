import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV

from preprocessing import preprocess_pipeline

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def tune_random_forest(X_train, y_train):
    rf_baseline = RandomForestClassifier(random_state=42, n_estimators=100)
    baseline_cv = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring='f1')
    print(f"\n Baseline Random Forest (default params):")
    print(f" CV F1: {baseline_cv.mean():.4f} ± {baseline_cv.std():.4f}")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    total_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf']) *
        len(param_grid['max_features'])
    )
    print(f"\nSearching {total_combinations} combinations × 5 folds = {total_combinations * 5} fits")
    print("This may take a few minutes...\n")

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    print(f"\n Best parameters found:")

    for param, value in grid_search.best_params_.items():
        print(f" {param}: {value}")

    print(f"\n Best CV F1 (tuned): {grid_search.best_score_:.4f}")
    print(f" Improvement: {grid_search.best_score_ - baseline_cv.mean():.4f}")

    return grid_search, grid_search.best_estimator_


def compare_before_after(baseline_model, tuned_model, X_train, X_test, y_train, y_test):
    baseline_model.fit(X_train, y_train)
    tuned_model.fit(X_train, y_train)

    metrics = {}
    for label, model in [('Before Tuning', baseline_model), ('After Tuning', tuned_model)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics[label] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred)
        }
        print(f"\n{label}:")
        for m, v in metrics[label].items():
            print("f {m}: {v:.4f}")

    comparison_df = pd.DataFrame(metrics).T
    ax = comparison_df.plot(kind='bar', figsize=(8, 5), colormap='RdYlGn', edgecolor='white')
    plt.title('Before vs After Hyperparameter Tuning\n(Random Forest)', fontsize=13, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/tuning_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    return metrics


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)

    print(f"\n Top {top_n} most important features:")
    for _, row in feature_importance_df.iterrows():
        bar = '█' * int(row['Importance'] * 100)
        print(f" {row['Feature']:<25} {row['Importance']:.4f} {bar}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_importance_df)))[::-1]
    axes[0].barh(
        range(len(feature_importance_df)),
        feature_importance_df['Importance'],
        color=colors,
        edgecolor='white'
    )

    axes[0].set_yticks(range(len(feature_importance_df)))
    axes[0].set_yticklabels(feature_importance_df['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Important Score')
    axes[0].set_title('Feature Importance (Random Forest)\nMean Decrease in Impurity', fontweight='bold')

    top5 = feature_importance_df.head(5)
    others_importance = importances.sum() - top5['Importance'].sum()
    pie_data = list(top5['Importance']) + [others_importance]
    pie_labels = list(top5['Feature']) + ['Others']
    axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                colors=plt.cm.Set3(np.linspace(0, 1, len(pie_labels))),
                startangle=90)
    axes[1].set_title('Feature Importance Distribution\n(Top 5 + Others)', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n Interpretation:")
    top_feature = feature_importance_df.iloc[0]['Feature']
    print(f" '{top_feature}' is the most predictive feature")
    print(f" This aligns with our EDA finding that sex/title strongly predict survival")
    print(f" If a feature you expected to be important isn't — investigate why!")

    return feature_importance_df


if __name__ == "__main__":
    sys.path.append('.')
    df = pd.read_csv('data/titanic.csv')
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_pipeline(df)

    grid_search, best_rf = tune_random_forest(X_train, y_train)

    baseline_rf = RandomForestClassifier(random_state=42, n_estimators=100)
    metrics = compare_before_after(baseline_rf, best_rf, X_train, X_test, y_train, y_test)

    best_rf.fit(X_train, y_train)
    importance_df = plot_feature_importance(best_rf, feature_cols)

    joblib.dump(best_rf, 'models/best_random_forest_tuned.pkl')
    print("\n Tuned model saved to models/best_random_forest_tuned.pkl")



