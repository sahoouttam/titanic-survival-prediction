import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import preprocess_pipeline

warnings.filterwarnings('ignore')
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
sns.set_style("whitegrid")


try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")


def get_models():
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        ),

        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=None
        ),

        'SVM': SVC(
            random_state=42,
            probability=True,
            kernel='rbf',
            C=1.0
        )
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    results['CV F1 Mean'] = cv_scores.mean()
    results['CV F1 Std'] = cv_scores.std()

    return results, y_pred, y_prob


def train_all_models(X_train, X_test, y_train, y_test):
    models = get_models()
    all_results = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n Training {name}...", end=" ")
        results, y_pred, y_prob = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        all_results.append(results)
        trained_models[name] = model
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

        safe_name = name.replace(" ", "-").lower()
        joblib.dump(model, f'models/{safe_name}.pkl')
        print(f" Accuracy={results['Accuracy']:.3f}, "
              f"F1={results['F1 Score']:.3f}, AUC={results['ROC-AUC']:.3f}")

    results_df = pd.DataFrame(all_results).set_index('Model')
    results_df = results_df.sort_values('F1 Score', ascending=False)

    print(results_df.round(4).to_string())
    return results_df, trained_models, predictions


def plot_confusion_matrices(trained_models, predictions, y_test):
    n_models = len(trained_models)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (name, model) in enumerate(trained_models.items()):
        cm = confusion_matrix(y_test, predictions[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Died', 'Survived'],
                    yticklabels=['Died', 'Survived'])
        axes[i].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, predictions[name]["y_pred"]):.3f}')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

    plt.suptitle('Confusion matrices - All models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(trained_models, predictions, y_test):
    plt.figure(figsize=(10, 7))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, (name, model) in enumerate(trained_models.items()):
        y_prob = predictions[name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 linewidth=2, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison_bar(results_df):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    plot_data = results_df[metrics]

    plot_data.plot(kind='bar', figsize=(14, 6), colormap='Set2', edgecolor='white', linewidth=0.5)
    plt.title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_detailed_report(trained_models, predictions, y_test):
    for name, model in trained_models.items():
        print(f"\n{'─' * 40}")
        print(f"  {name.upper()}")
        print('─' * 40)
        print(classification_report(y_test, predictions[name]['y_pred'],
                                    target_names=['Died', 'Survived']))


if __name__ == "__main__":
    sys.path.append('src')
    df = pd.read_csv('data/titanic.csv')
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_pipeline(df)

    results_df, trained_models, predictions = train_all_models(X_train, X_test, y_train, y_test)
    plot_confusion_matrices(trained_models, predictions, y_test)
    plot_roc_curves(trained_models, predictions, y_test)
    plot_model_comparison_bar(results_df)
    print_detailed_report(trained_models, predictions, y_test)

    best_model_name = results_df['F1 Score'].idxmax()
    print(f"\nBEST MODEL: {best_model_name}")
    print(f" F1 Score: {results_df.loc[best_model_name, 'F1 Score']:.4f}")
    print(f" ROC-AUC: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")

