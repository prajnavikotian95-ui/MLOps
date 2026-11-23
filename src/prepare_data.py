from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    df = pd.read_csv("data/tourism.csv")

    drop_cols = [col for col in ["Unnamed: 0", "CustomerID"] if col in df.columns]
    df_clean = df.drop(columns=drop_cols).copy()

    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    if "Gender" in df_clean.columns:
        df_clean["Gender"] = df_clean["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df_clean.columns:
        df_clean["MaritalStatus"] = df_clean["MaritalStatus"].replace({"Single": "Unmarried"})
    if "Occupation" in df_clean.columns:
        df_clean["Occupation"] = df_clean["Occupation"].replace({"Free Lancer": "Freelancer"})

    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.2,
        random_state=42,
        stratify=df_clean["ProdTaken"],
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved train.csv {train_df.shape} and test.csv {test_df.shape} to {out_dir}")


if __name__ == "__main__":
    main()
