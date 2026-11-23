import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_splits(processed_dir: str = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test splits from the artifact/processed folder."""
    base = Path(processed_dir)
    train_path = base / "train.csv"
    test_path = base / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv inside {base.resolve()}. "
            "Run src/prepare_data.py or download the workflow artifact first."
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def train_and_evaluate(
    processed_dir: str = "data/processed",
    model_output_path: str = "models/best_model.joblib",
    experiment_name: str = "Visit_With_Us_Wellness_Tourism",
) -> None:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    train_df, test_df = load_splits(processed_dir)

    target_col = "ProdTaken"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' must be present in train and test sets.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    preprocessor = build_preprocessor(X_train)

    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [8, 15, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="RandomForest_GridSearchCV"):
        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "cv_best_roc_auc": float(grid_search.best_score_),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        out_file = Path(model_output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, out_file)

        mlflow.sklearn.log_model(
            best_model,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )
        mlflow.log_artifact(str(out_file), artifact_path="serialized_model")

        print("=== BEST PARAMETERS ===")
        for k, v in best_params.items():
            print(f"{k}: {v}")

        print("\n=== TEST SET EVALUATION ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print(f"\nSaved best model to: {out_file.resolve()}")


if __name__ == "__main__":
    train_and_evaluate()
