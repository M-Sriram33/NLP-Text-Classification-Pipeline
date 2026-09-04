import joblib
import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def main():

    # ---------------------------------------------------------
    # 1. Define project paths
    # ---------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    train_path = (
        project_root
        / "data"
        / "processed"
        / "train_clean.csv"
    )

    model_dir = (
        project_root
        / "models"
    )

    model_path = (
        model_dir
        / "sentiment_model.joblib"
    )

    # ---------------------------------------------------------
    # 2. Check that training data exists
    # ---------------------------------------------------------

    if not train_path.exists():

        raise FileNotFoundError(
            f"Training data not found at: {train_path}"
        )

    # Create model directory if necessary

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 3. Load training data
    # ---------------------------------------------------------

    print("Loading training data...")

    train_df = pd.read_csv(
        train_path
    )

    # Check required columns

    required_columns = {
        "text",
        "label"
    }

    missing_columns = (
        required_columns
        - set(train_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    X_train = train_df["text"]
    y_train = train_df["label"]

    print(
        f"Training samples: {len(train_df):,}"
    )

    print(
        "Class distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------
    # 4. Build the complete NLP pipeline
    # ---------------------------------------------------------

    print("\nBuilding TF-IDF + Linear SVM pipeline...")

    model = Pipeline([
        (
            "tfidf",

            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        ),

        (
            "svm",

            LinearSVC(
                C=1.0
            )
        )
    ])

    # ---------------------------------------------------------
    # 5. Train the model
    # ---------------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ---------------------------------------------------------
    # 6. Display feature information
    # ---------------------------------------------------------

    tfidf = model.named_steps["tfidf"]

    feature_count = len(
        tfidf.get_feature_names_out()
    )

    print(
        f"TF-IDF features: {feature_count:,}"
    )

    # ---------------------------------------------------------
    # 7. Save the complete pipeline
    # ---------------------------------------------------------

    print(
        "\nSaving trained model..."
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to:\n{model_path}"
    )

    print(
        "\nTraining pipeline completed successfully."
    )


if __name__ == "__main__":
    main()