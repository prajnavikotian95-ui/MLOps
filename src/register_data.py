from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "Unnamed: 0",
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_and_validate_data(csv_path: str = "data/tourism.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    print("=== DATA SUMMARY ===")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column check: PASSED")
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("\nTarget (ProdTaken) counts:")
    print(df["ProdTaken"].value_counts().to_string())
    return df


if __name__ == "__main__":
    register_and_validate_data()
    