import sys
import joblib
from pathlib import Path


def load_model():
    """Load the trained sentiment classification model."""
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "sentiment_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}"
        )

    return joblib.load(model_path)


def predict_sentiment(text):
    """Predict sentiment for a single piece of text."""
    model = load_model()

    prediction = model.predict([text])[0]

    if prediction == 1:
        return "POSITIVE"
    return "NEGATIVE"


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "Your review text here"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    sentiment = predict_sentiment(text)

    print(f"Prediction: {sentiment}")


if __name__ == "__main__":
    main()