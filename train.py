from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERICAL_FEATURES = FEATURE_COLUMNS[1:]
CATEGORICAL_FEATURES = ["Type"]

THRESHOLD = 0.87


def train():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["Machine failure"]

    # Reproduce the original 80/20 test split
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    # Final model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    # Handle class imbalance
    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # Train
    model.fit(
        X_train,
        y_train,
        classifier__sample_weight=weights,
    )

    # Save model and metadata
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "threshold": THRESHOLD,
        "features": FEATURE_COLUMNS,
    }

    joblib.dump(artifact, MODEL_PATH)

    print("Model trained successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Classification threshold: {THRESHOLD}")


if __name__ == "__main__":
    train()