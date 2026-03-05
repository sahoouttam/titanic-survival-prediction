import os.path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def preprocess_new_passenger(passenger_dict):
    df = pd.DataFrame([passenger_dict])

    if 'name' in df.columns and df['Name'].notna().any():
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Mrs', 'Ms': 'Miss', 'Lady': 'Mrs',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Sir': 'Rare', 'Capt': 'Rare', 'Mme': 'Mrs'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    else:
        if df['Sex'].iloc[0] == 'female':
            df['Title'] = 'Miss' if df['Age'].iloc[0] < 18 else 'Mrs'
        else:
            df['Title'] = 'Master' if df['Age'].iloc[0] < 15 else 'Mr'

    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_alone'] = (df['Family_size'] == 1).astype(int)
    df['Family_category'] = pd.cut(df['Family_size'], bins=[0, 1, 4, 20],
                                   labels=['alone', 'small', 'large'])
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Age_band'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                            labels=['child', 'teen', 'young_adult', 'adult', 'senior'])

    df['Sex_encoded'] = (df['Sex'] == 'female').astype(int)

    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='embarked')
    title_dummies = pd.get_dummies(df['Title'], prefix='title')
    age_band_dummies = pd.get_dummies(df['Age_band'], prefix='age_band')
    family_cat_dummies = pd.get_dummies(df['Family_category'], prefix='fam')

    df = pd.concat([df, embarked_dummies, title_dummies, age_band_dummies, family_cat_dummies], axis=1)
    return df


def align_features(df, expected_feature_cols):
    for col in expected_feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_feature_cols]


def predict_passenger(passenger_dict, model_path='models/best_random_forest_tuned.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load('models/scaler.pkl')
    scale_cols = joblib.load('models/scale_cols.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')

    df = preprocess_new_passenger(passenger_dict)

    df_aligned = align_features(df, feature_cols)

    valid_scale_cols = [c for c in scale_cols if c in df_aligned.columns]
    df_aligned[valid_scale_cols] = scaler.transform(df_aligned[valid_scale_cols])

    prediction = model.predict(df_aligned)[0]
    probability = model.predict_proba(df_aligned)[0][1]

    return prediction, probability


def print_prediction_result(passenger_dict, prediction, probability):
    print("\n Passenger Details:")
    for key, value in passenger_dict.items():
        print(f" {key}: {value}")
    survived = "✅ SURVIVED" if prediction == 1 else "❌ DID NOT SURVIVE"
    print(f"   Prediction: {survived}")
    print(f"   Survival Probability: {probability:.1%}")

    if probability > 0.8:
        confidence = "HIGH confidence"
    elif probability > 0.6:
        confidence = "MODERATE confidence"
    elif probability > 0.4:
        confidence = "LOW confidence"
    else:
        confidence = "HIGH confidence (did not survive)"

    print(f" Confidence: {confidence}")


def run_example_predictions():
    test_passengers = [
        {
            'Name': '25-year-old 1st class female, traveling alone',
            'data': {
                'Pclass': 1, 'Sex': 'female', 'Age': 25,
                'SibSp': 0, 'Parch': 0, 'Fare': 100, 'Embarked': 'C'
            }
        },
        {
            'Name': '35-year-old 3rd class male, traveling alone',
            'data': {
                'Pclass': 3, 'Sex': 'male', 'Age': 35,
                'SibSp': 0, 'Parch': 0, 'Fare': 8, 'Embarked': 'S'
            }
        },
        {
            'Name': '5-year-old child, 2nd class, with parents',
            'data': {
                'Pclass': 2, 'Sex': 'male', 'Age': 5,
                'SibSp': 1, 'Parch': 2, 'Fare': 30, 'Embarked': 'S'
            }
        },
        {
            'Name': '50-year-old wealthy 1st class male',
            'data': {
                'Pclass': 1, 'Sex': 'male', 'Age': 50,
                'SibSp': 1, 'Parch': 0, 'Fare': 200, 'Embarked': 'C'
            }
        },
        {
            'Name': '22-year-old female, 3rd class, large family',
            'data': {
                'Pclass': 3, 'Sex': 'female', 'Age': 22,
                'SibSp': 3, 'Parch': 2, 'Fare': 15, 'Embarked': 'S'
            }
        },
    ]

    results = []
    for passenger in test_passengers:
        try:
            pred, prob = predict_passenger(passenger['data'])
            print_prediction_result(passenger['data'], pred, prob)
            results.append({
                'Passenger': passenger['name'],
                'Prediction': 'Survived' if pred == 1 else 'Died',
                'Probability': f"{prob:.1%}"
            })
        except Exception as e:
            print(f"f Error predicting for {passenger['Name']}: {e}")

    if results:
        print("\n SUMMARY TABLE:")
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    if not os.path.exists('models/best_random_forest_tuned.pkl'):
        print("No trained model found!")
    else:
        run_example_predictions()