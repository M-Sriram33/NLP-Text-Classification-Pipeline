# IMDb Sentiment Analysis — End-to-End NLP Text Classification

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Tests](https://img.shields.io/badge/Tests-2%20passed-brightgreen)

An end-to-end Natural Language Processing (NLP) project for classifying IMDb movie reviews as **Positive** or **Negative**.

The project covers the complete machine learning workflow:

- Dataset loading
- Data validation
- Duplicate detection
- Train/test leakage detection
- Text preprocessing
- Exploratory Data Analysis (EDA)
- TF-IDF feature extraction
- Multiple model comparison
- Linear SVM training
- Model evaluation
- Confusion matrix
- Error analysis
- Model serialization
- Command-line prediction
- REST API using FastAPI
- Automated tests
- Git/GitHub project organization

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Dataset](#3-dataset)
- [4. Dataset Validation](#4-dataset-validation)
- [5. Data Preprocessing](#5-data-preprocessing)
- [6. Exploratory Data Analysis](#6-exploratory-data-analysis)
- [7. NLP Feature Engineering](#7-nlp-feature-engineering)
- [8. Model Training](#8-model-training)
- [9. Model Comparison](#9-model-comparison)
- [10. Final Model](#10-final-model)
- [11. Evaluation Results](#11-evaluation-results)
- [12. Confusion Matrix](#12-confusion-matrix)
- [13. Error Analysis](#13-error-analysis)
- [14. Model Serialization](#14-model-serialization)
- [15. Command Line Interface](#15-command-line-interface)
- [16. REST API](#16-rest-api)
- [17. Automated Testing](#17-automated-testing)
- [18. Project Architecture](#18-project-architecture)
- [19. Project Structure](#19-project-structure)
- [20. Installation](#20-installation)
- [21. Running the Project](#21-running-the-project)
- [22. Technologies Used](#22-technologies-used)
- [23. Results Summary](#23-results-summary)
- [24. Limitations](#24-limitations)
- [25. Future Improvements](#25-future-improvements)
- [26. Key Learnings](#26-key-learnings)
- [27. Project Status](#27-project-status)

---

# 1. Project Overview

This project implements a complete NLP text classification system using the **IMDb movie review dataset**.

The goal is to predict whether a movie review expresses a:

- `Positive` sentiment
- `Negative` sentiment

The project uses classical machine learning techniques rather than a transformer-based architecture.

The main pipeline is:

```text
IMDb Reviews
      |
      v
Data Validation
      |
      v
Duplicate / Leakage Detection
      |
      v
Text Cleaning
      |
      v
Exploratory Data Analysis
      |
      v
TF-IDF Feature Extraction
      |
      v
Model Comparison
      |
      +--------------------+
      |                    |
      v                    v
Logistic Regression    Naive Bayes
      |
      +--------------------+
               |
               v
          Linear SVM
               |
               v
          Evaluation
               |
        +------+------+
        |             |
        v             v
       CLI           REST API
```

---

# 2. Problem Statement

Sentiment analysis is an NLP classification task where a model determines the emotional polarity of a piece of text.

For this project, the input is an IMDb movie review:

```text
"The movie was absolutely fantastic and very enjoyable."
```

The expected output is:

```text
Positive
```

For example:

```text
Input:
"This was one of the worst movies I have ever seen."

Output:
Negative
```

The task is therefore formulated as a binary classification problem:

```text
Input  -> Movie Review Text
Output -> 0 or 1
```

Where:

```text
0 = Negative
1 = Positive
```

---

# 3. Dataset

## Dataset Used

The project uses the IMDb movie review dataset.

The dataset contains:

```text
Training reviews: 25,000
Testing reviews:  25,000
Total reviews:    50,000
```

Each record contains two main columns:

| Column | Description |
|---|---|
| `text` | Movie review |
| `label` | Sentiment label |

The labels are:

```text
0 = Negative
1 = Positive
```

---

## Raw Dataset Shape

```text
Training:
(25000, 2)

Testing:
(25000, 2)
```

---

# 4. Dataset Validation

Before training the model, the dataset was checked for common data-quality problems.

The following checks were performed:

- Missing values
- Duplicate rows
- Duplicate reviews
- Train/test overlap
- Class distribution
- Review length

---

## Missing Values

```text
Training missing values:
0

Testing missing values:
0
```

Therefore, no missing-value imputation was required.

---

## Duplicate Detection

The training dataset contained:

```text
96 duplicate reviews
```

The testing dataset contained:

```text
199 duplicate reviews
```

These duplicates were removed.

---

## Train/Test Leakage

Exact text overlap between training and testing data was checked.

Initially:

```text
Exact train/test overlap:
123 reviews
```

These overlapping test reviews were removed to prevent data leakage.

After cleaning:

```text
Remaining train/test overlap:
0
```

This ensures that the final evaluation is performed on reviews that were not present in the training set.

---

## Final Dataset Size

After duplicate removal and leakage prevention:

```text
Training:
24,904 reviews

Testing:
24,678 reviews
```

Final class distribution:

```text
Training:
Positive: 12,472
Negative: 12,432

Testing:
Positive: 12,412
Negative: 12,266
```

The classes are therefore reasonably balanced.

---

# 5. Data Preprocessing

The text preprocessing pipeline performs lightweight cleaning before feature extraction.

The following operations are applied:

```text
Raw Review
    |
    v
Remove HTML tags
    |
    v
Normalize whitespace
    |
    v
Remove leading/trailing spaces
    |
    v
Clean Review
```

Example:

```text
Before:

"<br />This movie was GREAT!!!"


After:

"This movie was GREAT!!!"
```

The cleaning function is:

```python
def clean_text(text):
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces
    text = text.strip()

    return text
```

The cleaned datasets are saved as:

```text
data/processed/train_clean.csv
data/processed/test_clean.csv
```

---

# 6. Exploratory Data Analysis

EDA was performed before model training to understand the dataset.

---

## 6.1 Class Distribution

The dataset contains approximately equal numbers of positive and negative reviews.

Training distribution:

```text
Negative: 12,432
Positive: 12,472
```

Testing distribution:

```text
Negative: 12,266
Positive: 12,412
```

This indicates that the dataset does not have a major class imbalance.

---

## 6.2 Review Length

The approximate review length statistics are:

```text
Mean:       233.8 words
Std:        173.7 words
Minimum:    10 words
Median:     174 words
75th %ile:  284 words
Maximum:    2470 words
```

This shows that review length varies considerably.

Some reviews are very short while others contain thousands of words.

---

# 7. NLP Feature Engineering

The text is converted into numerical features using **TF-IDF**.

TF-IDF stands for:

```text
Term Frequency - Inverse Document Frequency
```

It assigns higher importance to words that are informative within a document but less common across the entire dataset.

---

## TF-IDF Configuration

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
```

### Parameters

| Parameter | Value | Purpose |
|---|---:|---|
| `lowercase` | `True` | Converts text to lowercase |
| `ngram_range` | `(1, 2)` | Uses unigrams and bigrams |
| `min_df` | `2` | Removes extremely rare terms |
| `max_df` | `0.95` | Removes extremely common terms |

The model uses both:

```text
Unigrams:
good
bad
excellent
boring

Bigrams:
very good
not good
really bad
highly recommended
```

This allows the classifier to capture some contextual information.

The resulting TF-IDF representation contains approximately:

```text
435,769 features
```

---

# 8. Model Training

Three traditional machine learning models were compared:

1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

All models use the same TF-IDF representation.

---

## 8.1 Logistic Regression

```python
LogisticRegression(
    max_iter=1000
)
```

Performance:

```text
Accuracy:  89.01%
Precision: 88.61%
Recall:    89.66%
F1-score:  89.14%
```

---

## 8.2 Linear SVM

```python
LinearSVC(
    C=1.0
)
```

Performance:

```text
Accuracy:  90.19%
Precision: 90.46%
Recall:    89.99%
F1-score:  90.22%
```

---

## 8.3 Multinomial Naive Bayes

```python
MultinomialNB()
```

Performance:

```text
Accuracy:  86.83%
Precision: 90.23%
Recall:    82.79%
F1-score:  86.35%
```

---

# 9. Model Comparison

The models were evaluated using:

```text
Accuracy
Precision
Recall
F1-score
```

Final comparison:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Linear SVM** | **90.19%** | **90.46%** | **89.99%** | **90.22%** |
| Logistic Regression | 89.01% | 88.61% | 89.66% | 89.14% |
| Naive Bayes | 86.83% | 90.23% | 82.79% | 86.35% |

The comparison is also saved to:

```text
results/model_comparison.csv
```

---

# 10. Final Model

The best-performing model was:

```text
TF-IDF
   +
Linear SVM
```

The complete model is implemented as a scikit-learn Pipeline:

```python
Pipeline([
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
        LinearSVC(C=1.0)
    )
])
```

Using a pipeline ensures that the same feature extraction process is automatically applied during inference.

---

# 11. Evaluation Results

The final Linear SVM model achieved:

```text
Accuracy:
90.19%

Precision:
90.46%

Recall:
89.99%

F1-score:
90.22%
```

These results show that the model correctly classifies approximately 9 out of every 10 IMDb reviews.

---

## Classification Report

```text
              precision    recall    f1-score

Negative        ~0.90       ~0.90       ~0.90
Positive        ~0.90       ~0.90       ~0.90
```

The model performs similarly across both sentiment classes.

---

# 12. Confusion Matrix

The final confusion matrix is:

```text
                 Predicted
              Negative Positive

Actual
Negative       11088     1178

Positive        1243    11169
```

Therefore:

```text
True Negatives  = 11088
False Positives = 1178

False Negatives = 1243
True Positives  = 11169
```

Total correct predictions:

```text
22,257
```

Total incorrect predictions:

```text
2,421
```

The confusion matrix visualization is saved at:

```text
results/confusion_matrix.png
```

---

# 13. Error Analysis

Error analysis was performed using the Linear SVM decision function.

The SVM produces a decision score indicating how close an example is to the classification boundary.

Examples of uncertain predictions had scores such as:

```text
-0.000005
 0.000098
 0.000267
-0.000273
-0.000369
```

Scores close to zero indicate that the model is uncertain.

One example was:

```text
"A good x evil film with tastes of "James Bond"..."
```

The actual label was:

```text
Positive
```

but the model predicted:

```text
Negative
```

with a decision score very close to zero.

This suggests that difficult examples may contain:

- Mixed sentiment
- Sarcasm
- Ambiguous language
- Complex sentence structure
- Context that cannot be fully captured by TF-IDF
- Words whose sentiment changes depending on context

Traditional TF-IDF models are strong baselines but do not understand language context in the same way as transformer-based models.

---

# 14. Model Serialization

The complete trained pipeline is saved using `joblib`.

File:

```text
models/sentiment_model.joblib
```

The saved object contains:

```text
TF-IDF Vectorizer
        +
Linear SVM
```

This means the application can directly load the model and make predictions without retraining.

Example:

```python
import joblib

model = joblib.load(
    "models/sentiment_model.joblib"
)

prediction = model.predict([
    "This movie was fantastic!"
])

print(prediction)
```

Output:

```text
[1]
```

---

# 15. Command Line Interface

A command-line interface was created using:

```text
src/predict.py
```

The CLI loads the serialized model and predicts sentiment for user-provided text.

---

## Positive Example

```bash
python -m src.predict "This movie was absolutely fantastic!"
```

Output:

```text
Prediction: POSITIVE
```

---

## Negative Example

```bash
python -m src.predict "This was one of the worst movies I have ever seen."
```

Output:

```text
Prediction: NEGATIVE
```

---

# 16. REST API

A REST API was implemented using:

```text
FastAPI
```

The API is located at:

```text
src/api.py
```

Start the server with:

```bash
uvicorn src.api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

---

## API Endpoints

### GET `/`

Returns API status.

Example:

```json
{
    "message": "IMDb Sentiment Classification API",
    "status": "running"
}
```

---

### POST `/predict`

Accepts a movie review and returns its predicted sentiment.

Request:

```json
{
    "text": "This movie was fantastic!"
}
```

Response:

```json
{
    "sentiment": "positive"
}
```

---

## Swagger Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface can be used to test the `/predict` endpoint interactively.

---

# 17. Automated Testing

Automated tests were created using:

```text
pytest
```

Test file:

```text
tests/test_model.py
```

The tests verify:

```text
Positive review -> Positive prediction
Negative review -> Negative prediction
```

Run:

```bash
pytest
```

Current result:

```text
2 passed
```

This confirms that the saved model can successfully perform basic positive and negative sentiment predictions.

---

# 18. Project Architecture

The project follows this architecture:

```text
                  ┌──────────────────────┐
                  │    IMDb Dataset      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Validation       │
                  │                      │
                  │ • Missing values     │
                  │ • Duplicates         │
                  │ • Leakage            │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Text Preprocessing   │
                  │                      │
                  │ • HTML removal       │
                  │ • Whitespace cleanup │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ TF-IDF              │
                  │ Unigrams + Bigrams  │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       Model Comparison      │
              │                              │
              │ Logistic Regression          │
              │ Linear SVM                   │
              │ Naive Bayes                  │
              └──────────────┬───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Best Model           │
                  │ Linear SVM           │
                  └──────────┬───────────┘
                             │
                  ┌──────────┴──────────┐
                  │                     │
                  ▼                     ▼
           ┌──────────────┐      ┌──────────────┐
           │ CLI          │      │ FastAPI      │
           │ Prediction   │      │ REST API     │
           └──────────────┘      └──────────────┘
```

---

# 19. Project Structure

```text
NLP text classification/
│
├── data/
│   ├── raw/
│   │   ├── imdb_train.csv
│   │   └── imdb_test.csv
│   │
│   └── processed/
│       ├── train_clean.csv
│       └── test_clean.csv
│
├── models/
│   └── sentiment_model.joblib
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
├── src/
│   ├── api.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   └── test_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 20. Installation

Clone the repository:

```bash
git clone https://github.com/M-Sriram33/NLP-Text-Classification-Pipeline.git
```

Move into the project:

```bash
cd "NLP text classification"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 21. Running the Project

## Train the Model

```bash
python -m src.train
```

This trains the TF-IDF + Linear SVM pipeline and saves the trained model to:

```text
models/sentiment_model.joblib
```

---


## Run Evaluation

```bash
python -m src.evaluate
```

Expected:

```text
=== Evaluation Results ===
Accuracy: 0.9019
```

---

## Run CLI Prediction

```bash
python -m src.predict "This movie was amazing and enjoyable."
```

Expected:

```text
Prediction: POSITIVE
```

Negative example:

```bash
python -m src.predict "This movie was boring and terrible."
```

Expected:

```text
Prediction: NEGATIVE
```

---

## Run API

```bash
uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Run Tests

```bash
pytest
```

Expected:

```text
2 passed
```

---

# 22. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| Linear SVM | Final classifier |
| Logistic Regression | Baseline model |
| Naive Bayes | Baseline model |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| Joblib | Model serialization |
| FastAPI | REST API |
| Uvicorn | API server |
| Pytest | Automated testing |
| Jupyter Notebook | EDA and experimentation |
| Git | Version control |
| GitHub | Repository hosting |

---

# 23. Results Summary

The final system achieved:

| Metric | Result |
|---|---:|
| Accuracy | **90.19%** |
| Precision | **90.46%** |
| Recall | **89.99%** |
| F1-score | **90.22%** |

Best model:

```text
Linear SVM
```

Feature representation:

```text
TF-IDF
```

Features:

```text
Unigrams + Bigrams
```

Number of approximate features:

```text
435,769
```

---

# 24. Limitations

Although the model performs well, it has several limitations.

## 1. Traditional NLP representation

TF-IDF does not understand deep semantic relationships between words.

For example:

```text
"I expected this movie to be bad, but it was actually excellent."
```

requires contextual understanding.

---

## 2. Sarcasm

Sarcastic statements can be difficult for TF-IDF models.

Example:

```text
"Wow, what a masterpiece... if you enjoy falling asleep."
```

The model may incorrectly classify this review.

---

## 3. Long-range context

TF-IDF primarily represents word and n-gram frequency.

It does not model long-range dependencies like transformer architectures.

---

## 4. Domain specificity

The model was trained on IMDb movie reviews.

Therefore, performance may decrease on text from different domains such as:

```text
Twitter
News
Product reviews
Financial text
Medical text
Customer support messages
```

---

# 25. Future Improvements

Possible improvements include:

### Transformer-based models

Replace TF-IDF + SVM with:

```text
BERT
RoBERTa
DistilBERT
ALBERT
```

---

### Hyperparameter tuning

Tune:

```text
SVM C
TF-IDF min_df
TF-IDF max_df
ngram_range
```

---

### Cross-validation

Use stratified cross-validation for more robust evaluation.

---

### Better preprocessing

Investigate:

```text
Lemmatization
Stop-word handling
Negation detection
Advanced tokenization
```

---

### More detailed error analysis

Analyze errors based on:

```text
Review length
Sarcasm
Negation
Mixed sentiment
Rare vocabulary
Decision confidence
```

---

### Production deployment

The FastAPI service could be deployed using:

```text
Docker
AWS
Azure
Google Cloud
Render
Railway
```

---

# 26. Key Learnings

This project demonstrates the complete lifecycle of an NLP classification system.

### Data Quality

Data leakage can artificially inflate model performance.

The project therefore explicitly checked:

```text
Duplicates
Train/test overlap
Missing values
Class distribution
```

---

### Feature Engineering

TF-IDF with unigrams and bigrams provides a strong classical NLP baseline.

---

### Model Selection

Different algorithms perform differently on the same feature representation.

In this experiment:

```text
Linear SVM
>
Logistic Regression
>
Naive Bayes
```

in terms of F1-score.

---

### Evaluation

Accuracy alone is not sufficient.

Precision, recall, F1-score and the confusion matrix provide a more complete view of model performance.

---

### Deployment

The trained model was not only evaluated but also made usable through:

```text
CLI
+
REST API
```

---

### Testing

Automated tests verify that the saved model produces expected sentiment predictions.

---

# 27. Project Status

```text
[✓] Dataset loaded
[✓] Data validation completed
[✓] Duplicate detection completed
[✓] Train/test leakage checked
[✓] Text preprocessing completed
[✓] Exploratory Data Analysis completed
[✓] TF-IDF feature extraction implemented
[✓] Logistic Regression trained
[✓] Linear SVM trained
[✓] Naive Bayes trained
[✓] Model comparison completed
[✓] Reproducible training script implemented
[✓] Final Linear SVM selected
[✓] Model evaluated
[✓] Confusion matrix generated
[✓] Error analysis completed
[✓] Model serialized
[✓] CLI implemented
[✓] FastAPI implemented
[✓] API tested
[✓] Automated tests implemented
[✓] README documentation completed
[✓] Git repository initialized
[✓] GitHub repository published
```

---

# Conclusion

This project implements a complete end-to-end NLP sentiment classification pipeline using IMDb movie reviews.

The final system uses:

```text
IMDb Reviews
      ↓
Data Cleaning
      ↓
Leakage Prevention
      ↓
TF-IDF
      ↓
Linear SVM
      ↓
90.19% Accuracy
      ↓
Model Serialization
      ↓
CLI + REST API
      ↓
Automated Tests
```

The final Linear SVM model achieved an accuracy of **90.19%** and an F1-score of **90.22%**.

The project demonstrates practical skills in:

```text
NLP
Machine Learning
Text Classification
Data Cleaning
Feature Engineering
Model Evaluation
Error Analysis
Model Deployment
REST APIs
Testing
Git/GitHub
```

---

## GitHub Repository

https://github.com/M-Sriram33/NLP-Text-Classification-Pipeline.git

---

## Author

**Sriram**

