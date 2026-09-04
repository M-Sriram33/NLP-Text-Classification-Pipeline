from datasets import load_dataset
import pandas as pd
from pathlib import Path


def main():
    print("Downloading IMDb dataset...")

    dataset = load_dataset("stanfordnlp/imdb")

    print("\nDataset downloaded successfully!")
    print(dataset)

    # Create raw data directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert train and test splits to pandas
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Save them
    train_path = output_dir / "imdb_train.csv"
    test_path = output_dir / "imdb_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTraining data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    print("\nTraining shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTraining class distribution:")
    print(train_df["label"].value_counts())

    print("\nExample review:")
    print(train_df.iloc[0]["text"])


if __name__ == "__main__":
    main()