import pandas as pd


def load_titanic_data(path="data/titanic.csv"):
    df = pd.read_csv(path)
    print(f" Dataset loaded successfully!")
    print(f" Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


def inspect(df):
    print("\n--- BASIC INFO ---")
    print(df.info())

    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

    print("\n--- STATISTICAL SUMMARY ---")

    print(df.describe())

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    }).query('`Missing Count` > 0').sort_values('Missing %', ascending=False)
    print(missing_df)

    print("\n--- COLUMN DATA TYPES ---")
    print(df.dtypes)

    print("\n--- TARGET VARIABLE (survived) DISTRIBUTION ---")
    print(df['Survived'].value_counts())
    print(f"   Survival rate: {df['Survived'].mean():.2%}")

    return missing_df


if __name__ == "__main__":
    df = load_titanic_data()
    missing_info = inspect(df)