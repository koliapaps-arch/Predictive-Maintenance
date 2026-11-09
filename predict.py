from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


class PredictiveMaintenanceModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Run python train.py first."
            )

        artifact = joblib.load(MODEL_PATH)

        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.features = artifact["features"]

    def predict(self, equipment_data: dict):
        # Check required input features
        missing = [
            feature
            for feature in self.features
            if feature not in equipment_data
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )

        # Convert input into DataFrame
        input_df = pd.DataFrame(
            [equipment_data],
            columns=self.features
        )

        # Predict failure probability
        probability = float(
            self.model.predict_proba(input_df)[0, 1]
        )

        # Apply optimized threshold
        prediction = int(
            probability >= self.threshold
        )

        return {
            "machine_failure": prediction,
            "failure_probability": round(probability, 4),
            "threshold": self.threshold,
            "status": (
                "Failure predicted"
                if prediction == 1
                else "Normal operation"
            )
        }


if __name__ == "__main__":
    predictor = PredictiveMaintenanceModel()

    sample_equipment = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0
    }

    result = predictor.predict(sample_equipment)

    print("Prediction result:")
    print(result)