# Emotion Classifier
# 🎭 Emotion Classifier (GoEmotions • Multi-Label Baseline)

> Classifies emotions from short text using the **GoEmotions (simplified)** dataset and a **TF-IDF + One-Vs-Rest Logistic Regression** baseline.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-success)

---

## 🚀 Overview

- **Task:** Multi-label emotion classification (a text can have multiple emotions).
- **Dataset:** GoEmotions (simplified label set) via Hugging Face `datasets`.
- **Baseline model:** `TfidfVectorizer` → `OneVsRestClassifier(LogisticRegression)`.
- **Notebook:** `notebooks/01_baseline_goemotions_multilabel.ipynb`.

This repo shows a solid end-to-end baseline: environment setup, data loading, model training, threshold tuning, evaluation, and saved artifacts.

---

## 🧩 Tech Stack

- Python 3.11
- scikit-learn, pandas, numpy
- datasets (Hugging Face)
- Jupyter (VS Code)
- Conda/Miniforge for environments

---

## 📊 Results (Baseline)

**Evaluation style:** multi-label, reported with *samples-average* metrics.  
**Threshold tuning:** swept `t ∈ [0.20 … 0.50]`, picked best by samples-F1.

| Setting | Samples-F1 | Samples-Precision | Samples-Recall |
|--------:|:-----------:|:-----------------:|:--------------:|
| **Threshold** `t = 0.20` | **0.54** | 0.54 | 0.58 |
| Top-k `k = 2` | ~0.54 | (varies) | (varies) |

> Notes:
> - Thresholding (t=0.20) reflects variable number of emotions per text and gave the best samples-F1 in this baseline.
> - Top-k returns exactly k emotions per text (nice for demos); thresholding is better for research-style evaluation.

---
## 📁 Project Structure

emotion-classifier/
│
├── notebooks/
│ └── 01_baseline_goemotions_multilabel.ipynb
│
├── models/ # saved artifacts (gitignored)
│ └── tfidf_logreg_goemotions_multilabel.joblib
│
├── src/ # optional scripts
│ ├── train.py
│ ├── predict.py
│ └── utils.py
│
├── reports/ # txt reports/metrics (optional)
├── README.md
└── .gitignore

---
## ⚙️ Reproduce Locally

```bash
# create & activate env
conda create -n emotion python=3.11 -y
conda activate emotion
conda install -y numpy pandas scikit-learn matplotlib jupyter ipykernel
python -m ipykernel install --user --name emotion --display-name "Python (Emotion ML Env)"

# open the notebook in VS Code and select the "Python (Emotion ML Env)" kernel
code .
BEST_THRESHOLD = 0.20
proba = model.predict_proba(["i'm nervous but excited"])
pred = (proba >= BEST_THRESHOLD).astype(int)
# if no labels pass threshold, fallback to top-1
# always returns k labels per input
k = 2
# pick the two highest-probability emotions per text

---

### 🏷️ (Optional) Repo polish on GitHub
- **Description:** “Multi-label emotion classification baseline on GoEmotions (scikit-learn).”
- **Topics/Tags:** `nlp`, `machine-learning`, `text-classification`, `goemotions`, `multilabel`, `scikit-learn`, `python`.
- **Default branch:** main (if not already).

---


