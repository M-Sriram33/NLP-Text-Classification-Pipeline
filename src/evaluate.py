import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    project_root = Path(__file__).resolve().parent.parent

    model_path = project_root / "models" / "sentiment_model.joblib"
    test_path = project_root / "data" / "processed" / "test_clean.csv"

    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading test data...")
    test_df = pd.read_csv(test_path)

    X_test = test_df["text"]
    y_test = test_df["label"]

    print("Running predictions...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"]
        )
    )

    print("=== Confusion Matrix ===")
    print(cm)


if __name__ == "__main__":
    main()