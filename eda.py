import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from load_data import load_titanic_data

os.makedirs("outputs", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_survival_by_sex(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x='Sex', hue='Survived', palette='Set2', ax=axes[0])
    axes[0].set_title('Survival Count By Sex')
    axes[0].set_xlabel('Sex')
    axes[0].set_ylabel('Count')
    axes[0].legend(['Died', 'Survived'])

    survival_rate = df.groupby('Sex')['Survived'].mean()
    survival_rate.plot(kind='bar', color=['#66c2a5', '#fc8d62'], ax=axes[1])
    axes[1].set_title('Survival Rate By Sex')
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(survival_rate):
        axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/survival_by_sex.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Survival rates by sex:\n{survival_rate.to_string()}")


def plot_survival_by_passenger_class(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set1', ax=axes[0])
    axes[0].set_title('Survival Count by Passenger Class')
    axes[0].set_xlabel('Passenger Class[1=First, 3=Third]')
    axes[0].legend(['Died', 'Survived'])

    survival_rate = df.groupby('Pclass')['Survived'].mean()
    survival_rate.plot(kind='bar', color=['#1f78b4', '#33a02c', '#e31a1c'], ax=axes[1])
    axes[1].set_title('Survival Rate by Passenger Class')
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(survival_rate):
        axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/survival_by_passenger_class.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Survival rates by passenger class:\n{survival_rate.to_string()}")


def plot_survival_by_age(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df[df['Survived'] == 0]['Age'].dropna().plot.kde(
        ax=axes[0], label='Died', color='red', linewidth=2)
    df[df['Survived'] == 1]['Age'].dropna().plot.kde(
        ax=axes[0], label='Survived', color='green', linewidth=2)
    axes[0].set_title('Age Distribution: Survivors Vs Non-Survivors')
    axes[0].set_xlabel('Age')
    axes[0].legend()
    axes[0].axvline(x=10, color='blue', linestyle='--', alpha=0.5, label='Age 10')

    sns.boxplot(data=df, x='Survived', y='Age', hue='Survived',
                palette={0: '#ff6b6b', 1: '#51cf66'}, legend=False, ax=axes[1])
    axes[1].set_title('Age Distribution by Survival (Boxplot)')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Died', 'Survived'])
    axes[1].set_ylabel('Age')

    plt.tight_layout()
    plt.savefig('outputs/survival_by_age.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f" Median age (died): {df[df['Survived'] == 0]['Age'].median():.1f}")
    print(f" Median age (survived): {df[df['Survived'] == 1]['Age'].median():.1f}")


def plot_survival_by_embarked(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x='Embarked', hue='Survived', palette='Set2',
                  ax=axes[0], order=['S', 'C', 'Q'])
    axes[0].set_title('Survival Count by Port of Embarkation')
    axes[0].set_xlabel('Embarked (S=Southampton, C=Cherbourg, Q=Queenstown)')
    axes[0].legend(['Died', 'Survived'])

    survival_rate = df.groupby('Embarked')['Survived'].mean().loc[['S', 'C', 'Q']]
    survival_rate.plot(kind='bar', color=['#4dac26', '#b8e186', '#d01c8b'], ax=axes[1])
    axes[1].set_title('Survival Rate by Port of Embarkation')
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=0)
    for i, v in enumerate(survival_rate):
        axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/survival_by_embarked.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f" Survival rates by embarked:\n{survival_rate.to_string()}")


def plot_fare_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df['Fare'].plot.hist(bins=50, ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Fare Distribution (Raw - Right Skewed)')
    axes[0].set_xlabel('Fare ($)')

    np.log1p(df['Fare']).plot.hist(bins=50, ax=axes[0], color='coral', edgecolor='white')
    axes[1].set_title('Fare Distribution (Log-Transformed)')
    axes[1].set_xlabel('log(Fare + 1)')

    plt.tight_layout()
    plt.savefig('outputs/fare_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Fare is right-skewed — consider log transform")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])

    temp_df = numeric_df.copy()
    if 'Sex' in df.columns:
        temp_df['Sex_encoded'] = (df['Sex'] == 'female').astype(int)

    plt.figure(figsize=(10, 8))
    correlation_matrix = temp_df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        mask=mask,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title('Feature Correlation Heatmap\n(Green = positive, Red = negative)', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heapmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nCorrelation with 'survived':")
    target_corr = correlation_matrix['Survived'].drop('Survived').sort_values(key=abs, ascending=False)
    print(target_corr.to_string())


def plot_combined_heatmap(df):
    pivot = df.pivot_table(values='Survived', index='Pclass', columns='Sex')

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2%',
        cmap='YlOrRd',
        cbar_kws={'label': 'Survival Rate'}
    )

    plt.title('Survival Rate: Passenger Class × Sex\n(The "Women & Children First" Effect)')
    plt.tight_layout()
    plt.savefig('outputs/pclass_sex_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    df = load_titanic_data("data/titanic.csv")
    plot_survival_by_sex(df)
    plot_survival_by_passenger_class(df)
    plot_survival_by_age(df)
    plot_survival_by_embarked(df)
    plot_fare_distribution(df)
    plot_correlation_heatmap(df)
    plot_combined_heatmap(df)