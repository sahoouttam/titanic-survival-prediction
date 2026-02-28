import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs("models", exist_ok=True)


def extract_title(df):
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        # Rare male titles → 'Rare'
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss',  # French for Miss
        'Countess': 'Mrs', 'Ms': 'Miss', 'Lady': 'Mrs',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare', 'Mme': 'Mrs'
    }

    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    return df


def create_family_features(df):
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_alone'] = (df['Family_size'] == 1).astype(int)

    df['Family_category'] = pd.cut(
        df['Family_size'],
        bins=[0, 1, 4, 20],
        labels=['alone', 'small', 'large']
    )
    return df


def handle_missing_values(df):
    df['Age'] = df.groupby('Title')['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    print(f" Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    df['Fare_log'] = np.log1p(df['Fare'])

    df['Age_band'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                            labels=['child', 'teen', 'young_adult', 'adult', 'senior'])
    return df


def encode_categorical(df):
    df['Sex_encoded'] = (df['Sex'] == 'female').astype(int)

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    title_dummies = pd.get_dummies(df['Title'], prefix='Title', drop_first=True)
    age_band_dummies = pd.get_dummies(df['Age_band'], prefix='Age_band', drop_first=True)
    family_cat_dummies = pd.get_dummies(df['Family_category'], prefix='Fam', drop_first=True)

    df = pd.concat([df, embarked_dummies, title_dummies, age_band_dummies, family_cat_dummies], axis=1)
    return df


def select_features(df):
    feature_columns = [
        'Pclass', 'Sex_encoded', 'Age', 'Fare_log',
        'SibSp', 'Parch', 'Family_size', 'Is_alone',
        # One-hot encoded columns
        'embarked_Q', 'embarked_S',
        'title_Miss', 'title_Mr', 'title_Mrs', 'title_Rare',
        'age_band_teen', 'age_band_young_adult', 'age_band_adult', 'age_band_senior',
        'fam_small', 'fam_large'
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    target = 'Survived'

    print(f" Selected {len(available_features)} features: {available_features}")
    return df[available_features], df[target]


def preprocess_pipeline(df, test_size=0.2, random_state=42):
    df = df.copy()

    print("\nExtracting title from name...")
    df = extract_title(df)

    print("Creating family size features...")
    df = create_family_features(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Engineering features (log fare, age bands)...")
    df = engineer_features(df)

    print("Encoding categorical variables...")
    df = encode_categorical(df)

    print("Selecting features...")
    X, y = select_features(df)

    print(f"\n Splitting data: {1-test_size:.0%} train / {test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f" Train size: {len(X_train)}, Test size: {len(X_test)}")

    scale_cols = ['Pclass', 'Age', 'Fare_log', 'Sibsp', 'Parch', 'Family_size']
    scale_cols = [col for col in scale_cols if col in X_train.columns]

    scaler = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(scale_cols, 'models/scale_cols.pkl')
    print("Scaler saved to models/scaler.pkl")

    print(f"\n Preprocessing complete!")
    print(f" X_train shape: {X_train.shape}")
    print(f" X_test shape: {X_test.shape}")
    print(f" Feature columns: {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test, scaler, list(X_train.columns)


if __name__ == "__main__":
    df = pd.read_csv("data/titanic.csv")
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_pipeline(df)
