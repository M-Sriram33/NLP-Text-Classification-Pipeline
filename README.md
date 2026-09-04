# 🎬 IMDb Sentiment Analysis — End-to-End NLP Classification

> An end-to-end Natural Language Processing (NLP) text classification project that predicts whether an IMDb movie review is **positive** or **negative** using **TF-IDF + Linear SVM**, with both a **CLI** and **FastAPI REST API** interface.

---

<p align="center">

**90.19% Accuracy** • **90.22% F1 Score** • **TF-IDF + Linear SVM** • **FastAPI** • **Pytest**

</p>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Project Objectives](#-project-objectives)
- [Dataset](#-dataset)
- [Data Quality & Cleaning](#-data-quality--cleaning)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [NLP Pipeline](#-nlp-pipeline)
- [Feature Engineering](#-feature-engineering)
- [Model Comparison](#-model-comparison)
- [Final Model](#-final-model)
- [Evaluation](#-evaluation)
- [Error Analysis](#-error-analysis)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Command-Line Interface](#-command-line-interface)
- [REST API](#-rest-api)
- [API Example](#-api-example)
- [Testing](#-testing)
- [Reproducibility](#-reproducibility)
- [Technologies](#-technologies)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

# 🎯 Project Overview

Sentiment analysis is a Natural Language Processing task used to determine the emotional polarity of text.

In this project, the goal is to classify IMDb movie reviews into two categories:

| Label | Sentiment |
|---:|---|
| `0` | Negative |
| `1` | Positive |

For example:

```text
Input:
"This movie was absolutely fantastic!"

Output:
POSITIVE
